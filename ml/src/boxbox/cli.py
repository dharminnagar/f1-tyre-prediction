"""``boxbox`` command line: download -> build -> train -> evaluate -> export (per profile)."""

import logging
from typing import Annotated

import joblib
import typer

from boxbox.config import Config, Profile, load_config
from boxbox.paths import dataset_paths

app = typer.Typer(add_completion=False, no_args_is_help=True)
ProfileOption = Annotated[str, typer.Option("--profile", "-p", help="subset | full")]


@app.callback()
def main() -> None:
    """BoxBox tyre degradation and pit-window pipeline."""


def _config(value: str) -> Config:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    profile: Profile
    if value == "subset":
        profile = "subset"
    elif value == "full":
        profile = "full"
    else:
        raise typer.BadParameter("profile must be 'subset' or 'full'")
    return load_config(profile)


def _stage_file(cfg: Config, name: str) -> str:
    path = dataset_paths().processed / cfg.profile / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


@app.command()
def download(
    profile: ProfileOption = "subset",
    force: bool = False,
    with_f1db: Annotated[bool, typer.Option("--with-f1db")] = False,
) -> None:
    """Fetch raw sources into dataset/ (idempotent; records the source used per season)."""
    from boxbox.ingestion.download import download as run_download

    for record in run_download(_config(profile), force=force, with_f1db=with_f1db):
        typer.echo(f"{record.season}: tables={record.tables_source} tyres={record.tyre_source}")


@app.command()
def build(profile: ProfileOption = "subset") -> None:
    """Build the clean lap dataset and the feature table."""
    from boxbox.features.build import build_features
    from boxbox.preprocessing.build import build_lap_dataset, dataset_hash, write_lap_dataset

    cfg = _config(profile)
    dataset = build_lap_dataset(cfg)
    write_lap_dataset(cfg, dataset)
    features = build_features(dataset.laps, cfg)
    features.to_parquet(dataset_paths().processed_file(cfg.profile, "features"), index=False)
    rate = float(dataset.stint_verification["matched"].mean())
    typer.echo(
        f"{len(dataset.laps)} laps ({int(dataset.laps['is_clean'].sum())} clean), "
        f"hash {dataset_hash(dataset.laps)}, stint verification {rate:.1%}"
    )


@app.command()
def train(profile: ProfileOption = "subset") -> None:
    """Fit all models on the training races."""
    import pandas as pd

    from boxbox.models.train import fit_all

    cfg = _config(profile)
    features = pd.read_parquet(dataset_paths().processed_file(cfg.profile, "features"))
    joblib.dump(fit_all(features, cfg), _stage_file(cfg, "models"))
    typer.echo("models fitted")


@app.command()
def evaluate(profile: ProfileOption = "subset") -> None:
    """Evaluate on validation (selection) and test (reported once)."""
    import pandas as pd

    from boxbox.evaluation.evaluate import evaluate as run_evaluate
    from boxbox.preprocessing.build import read_lap_dataset

    cfg = _config(profile)
    dataset = read_lap_dataset(cfg)
    features = pd.read_parquet(dataset_paths().processed_file(cfg.profile, "features"))
    result = run_evaluate(
        dataset.laps,
        features,
        joblib.load(_stage_file(cfg, "models")),
        cfg,
        dataset.stint_verification,
    )
    joblib.dump(result, _stage_file(cfg, "evaluation"))
    for row in result.metric_rows:
        if row["split"] in {"val", "test"} and row["target"] in {
            "next_lap",
            f"next_{cfg.horizon_n}_mean",
        }:
            typer.echo(
                f"{row['model']:>15} {row['split']:>4} {row['target']:>12} MAE {row['mae']:.3f}"
            )
    typer.echo(f"selected: {result.selected_model}; pit loss {result.pit_loss.median_s:.2f} s")


@app.command()
def export(profile: ProfileOption = "subset") -> None:
    """Write the artifact tree consumed by the API."""
    import pandas as pd

    from boxbox.export.artifacts import export_all
    from boxbox.preprocessing.build import read_lap_dataset

    cfg = _config(profile)
    features = pd.read_parquet(dataset_paths().processed_file(cfg.profile, "features"))
    paths = export_all(
        cfg, read_lap_dataset(cfg), features, joblib.load(_stage_file(cfg, "evaluation"))
    )
    typer.echo(f"artifacts written to {paths.profile_dir}")


@app.command(name="all")
def run_all(profile: ProfileOption = "subset") -> None:
    """Download -> build -> train -> evaluate -> export."""
    download(profile)
    build(profile)
    train(profile)
    evaluate(profile)
    export(profile)
