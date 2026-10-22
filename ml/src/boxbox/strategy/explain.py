"""Model-agnostic decomposition of the next-lap prediction and plain-language explanations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boxbox.models.forecaster import Forecaster
from boxbox.strategy.pit_window import PitWindowResult
from boxbox.strategy.remaining_life import RemainingLife
from boxbox.strategy.rollout import fuel_terms, predict_segments, stay_out_design


@dataclass(frozen=True)
class Decomposition:
    """Additive parts of the predicted next lap time (they sum to ``total_s``).

    tyre = f(x) - f(x at reference tyre age); driver = f(x_ref) - f(x_ref with zero driver
    offset/quali gap); context = the remaining model output; field_ref and fuel are exact.
    """

    field_ref_s: float
    driver_offset_s: float
    tyre_s: float
    fuel_s: float
    context_s: float

    @property
    def total_s(self) -> float:
        return self.field_ref_s + self.driver_offset_s + self.tyre_s + self.fuel_s + self.context_s


def decompose_next_lap(
    model: Forecaster, state: pd.Series, reference_age: int, fuel_s_per_lap: float
) -> Decomposition:
    """Decompose the next-lap prediction of ``model`` at the decision ``state``."""
    design = stay_out_design(state, int(state["lap"]) + 1)
    reference = design.assign(tyre_age=float(reference_age), stint_lap=float(reference_age))
    neutral_driver = reference.assign(driver_offset_s=0.0, quali_gap_s=0.0)
    full, at_ref, neutral = (
        float(p.y_s[0]) for p in predict_segments(model, state, [design, reference, neutral_driver])
    )
    laps = design["lap_number"].to_numpy(dtype=np.int64)
    return Decomposition(
        field_ref_s=float(state["s_field_ref_next_s"]),
        driver_offset_s=at_ref - neutral,
        tyre_s=full - at_ref,
        fuel_s=float(fuel_terms(state, laps, fuel_s_per_lap)[0]),
        context_s=neutral,
    )


def explanation_lines(
    state: pd.Series,
    decomposition: Decomposition,
    remaining: RemainingLife,
    window: PitWindowResult | None,
    reference_age: int,
) -> list[str]:
    """Human-readable reasons behind the degradation, remaining-life and window estimates."""
    lines = [
        f"{state['compound']} tyre, {int(state['tyre_age'])} laps old "
        f"(stint lap {int(state['stint_lap'])}).",
        f"Tyre age costs {decomposition.tyre_s:+.2f} s vs a {reference_age}-lap-old "
        f"set in the same race context.",
    ]
    slope = state.get("s_deg_slope_5")
    if slope is not None and pd.notna(slope):
        lines.append(f"Recent clean laps degrade {float(slope):+.3f} s/lap (5-lap slope).")
    lines.append(
        f"Driver pace offset {decomposition.driver_offset_s:+.2f} s vs field; "
        f"fuel load still costs {decomposition.fuel_s:.2f} s."
    )
    if remaining.value == 0 and not remaining.capped_by_race_end:
        lines.append(
            f"The tyre-attributable loss already exceeds {remaining.threshold_s:.2f} s: "
            "the set is past its competitive life."
        )
    elif remaining.capped_by_race_end:
        lines.append(
            f"Predicted to stay within {remaining.threshold_s:.2f} s of reference pace to the flag."
        )
    else:
        lines.append(
            f"About {remaining.value} competitive laps left before the tyre loss exceeds "
            f"{remaining.threshold_s:.2f} s."
        )
    if window is not None:
        if window.window is not None:
            lines.append(
                f"Pitting for {window.replacement_compound} pays back the "
                f"{window.pit_loss_s:.1f} s pit loss beyond model uncertainty on laps "
                f"{window.window.start_lap}-{window.window.end_lap}."
            )
        else:
            lines.append(
                f"No lap where a stop for {window.replacement_compound} beats staying out by "
                f"more than the uncertainty margin."
            )
        if not window.stay_out_feasible:
            lines.append("Staying out to the end would break the two-dry-compound rule.")
    return lines
