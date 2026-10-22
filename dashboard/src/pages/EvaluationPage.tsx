import { useState } from "react";

import type { GroupBy, MetricRow, ModelName, Profile, ProfileMetrics } from "../api/client";
import {
  useDrivers,
  useErrorGroups,
  useMetrics,
  usePitTiming,
  usePredictions,
  useRaces,
  useThresholdSensitivity,
} from "../api/queries";
import { ErrorDistributionChart } from "../components/charts/ErrorDistributionChart";
import { HorizonChart } from "../components/charts/HorizonChart";
import { PredVsActualChart } from "../components/charts/PredVsActualChart";
import { AssumptionsPanel } from "../components/explain/AssumptionsPanel";
import { Card, DataTable, Field, inputClass, StatusMessage } from "../components/ui";
import { MODEL_ORDER } from "../lib/colors";
import { formatPercent, formatSeconds, modelLabel } from "../lib/format";
import { type DecisionMode, thresholdTable } from "../lib/transform";

const DECISION_LABELS: Record<DecisionMode, string> = {
  stint_lap_5: "Stint lap 5 (one per stint)",
  all: "Every lap from stint lap 4",
};

function laps(value: number | null | undefined, signed = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}`;
}

/** Remaining-life MAE (bias) per threshold and model for one decision mode. */
export function ThresholdTable({
  rows,
  decision,
  selectedModel,
}: {
  rows: Parameters<typeof thresholdTable>[0];
  decision: DecisionMode;
  selectedModel: ModelName;
}) {
  const table = thresholdTable(rows, decision);
  return (
    <DataTable
      caption="Threshold sensitivity"
      columns={[
        { key: "threshold", label: "Threshold", align: "right" },
        ...MODEL_ORDER.map((m) => ({ key: m, label: `${modelLabel(m)} MAE (bias)`, align: "right" as const })),
        { key: "n", label: "Uncensored / censored", align: "right" },
        { key: "consistency", label: "Censored consistency", align: "right" },
      ]}
      rows={table.map(({ threshold, byModel }) => {
        const selected = byModel.get(selectedModel);
        return {
          threshold: `${threshold.toFixed(2)} s`,
          ...Object.fromEntries(
            MODEL_ORDER.map((m) => {
              const row = byModel.get(m);
              return [m, row ? `${laps(row.mae_laps)} (${laps(row.bias_laps, true)})` : "—"];
            }),
          ),
          n: selected ? `${selected.n_uncensored} / ${selected.n_censored}` : "—",
          consistency: formatPercent(selected?.censored_consistency, 0),
        };
      })}
    />
  );
}

/** Calibrated-window outcome per held-out split (stops not excluded). */
export function PitTimingSummaryTable({ metrics }: { metrics: ProfileMetrics }) {
  const splits = Object.entries(metrics.pit_timing_summary).sort(([a], [b]) => (a === "val" ? -1 : b === "val" ? 1 : 0));
  return (
    <DataTable
      caption="Pit-timing summary"
      columns={[
        { key: "split", label: "Split" },
        { key: "kept", label: "Stops kept", align: "right" },
        { key: "rate", label: "In window", align: "right" },
        { key: "width", label: "Median width", align: "right" },
        { key: "bias", label: "Stint-1 optimum − actual", align: "right" },
        { key: "pastCap", label: "Near-optimum past cap", align: "right" },
        { key: "excluded", label: "Excluded" },
      ]}
      rows={splits.map(([split, s]) => ({
        split,
        kept: `${s.n_kept}/${s.n}`,
        rate: formatPercent(s.in_window_rate_kept, 0),
        width: s.median_width_kept === null || s.median_width_kept === undefined ? "—" : `${s.median_width_kept.toFixed(0)} laps`,
        bias: `${laps(s.optimum_bias_kept_by_stint?.["1"], true)} laps`,
        pastCap: formatPercent(s.near_optimum_past_cap_share_kept, 0),
        excluded:
          Object.entries(s.excluded)
            .map(([reason, n]) => `${reason.replace("_", " ")} ${n}`)
            .join(", ") || "none",
      }))}
    />
  );
}

function rate(value: number | null | undefined): string {
  return formatPercent(value, 0);
}

/** Empirical coverage and mean width of the served 80% intervals (val and test). */
export function IntervalCoverageTable({ metrics }: { metrics: ProfileMetrics }) {
  const rows = metrics.interval_coverage;
  const find = (model: string, target: string, split: string) =>
    rows.find((r) => r.model === model && r.target === target && r.split === split);
  const targets = [...new Set(rows.map((r) => r.target))];
  return (
    <DataTable
      caption="Interval coverage"
      columns={[
        { key: "model", label: "Model" },
        { key: "target", label: "Target" },
        { key: "method", label: "Method" },
        { key: "val", label: "Val coverage (width)", align: "right" },
        { key: "test", label: "Test coverage (width)", align: "right" },
      ]}
      rows={MODEL_ORDER.flatMap((model) =>
        targets.flatMap((target) => {
          const val = find(model, target, "val");
          const test = find(model, target, "test");
          if (val === undefined && test === undefined) return [];
          const cell = (r: typeof val) =>
            r === undefined ? "—" : `${rate(r.coverage)} (${formatSeconds(r.mean_width_s, 2)})`;
          return [
            {
              model: modelLabel(model),
              target: target.replaceAll("_", " "),
              method: (val ?? test)?.method === "quantile" ? "quantile HGB" : "±1.28σ",
              val: cell(val),
              test: cell(test),
            },
          ];
        }),
      )}
    />
  );
}

/** LSTM vs GRU cell with identical hyper-parameters (the GRU is evaluation-only). */
export function SequenceCellTable({ metrics }: { metrics: ProfileMetrics }) {
  return (
    <DataTable
      caption="LSTM vs GRU"
      columns={[
        { key: "cell", label: "Cell" },
        { key: "split", label: "Split" },
        { key: "next", label: "Next lap", align: "right" },
        { key: "mean", label: "Next-n mean", align: "right" },
        { key: "k", label: "k-lap MAE", align: "right" },
        { key: "slope", label: "Held-out slope", align: "right" },
        { key: "time", label: "Train time", align: "right" },
      ]}
      rows={metrics.sequence_cell_comparison.map((r) => ({
        cell: modelLabel(r.cell),
        split: r.split,
        next: formatSeconds(r.next_lap_mae),
        mean: `${formatSeconds(r.next_n_mean_mae)} (n=${r.n})`,
        k: `${formatSeconds(r.horizon_k_mae)} (k=${r.horizon_k})`,
        slope: r.rollout_s_per_lap_held_out === null ? "—" : `${r.rollout_s_per_lap_held_out.toFixed(3)} s/lap`,
        time: `${r.train_seconds.toFixed(0)} s (best epoch ${r.best_epoch})`,
      }))}
    />
  );
}

/** Pit-timing outcome of every model, each with its own validation-calibrated margin. */
export function PitTimingByModelTable({ metrics }: { metrics: ProfileMetrics }) {
  return (
    <DataTable
      caption="Pit timing by model"
      columns={[
        { key: "model", label: "Model" },
        { key: "margin", label: "Margin", align: "right" },
        { key: "val", label: "Val in window (width)", align: "right" },
        { key: "test", label: "Test in window (width)", align: "right" },
        { key: "bias", label: "Val stint-1 optimum − actual", align: "right" },
      ]}
      rows={metrics.pit_timing_by_model.map((r) => {
        const split = (name: string) => {
          const s = r.summary[name];
          if (s === undefined) return "—";
          const width = s.median_width_kept === null || s.median_width_kept === undefined ? "—" : `${s.median_width_kept.toFixed(0)} laps`;
          return `${rate(s.in_window_rate_kept)} (${width})`;
        };
        return {
          model: modelLabel(r.model),
          margin: `${r.calibrated_margin_sigma.toFixed(2)}σ`,
          val: split("val"),
          test: split("test"),
          bias: `${laps(r.summary.val?.optimum_bias_kept_by_stint?.["1"], true)} laps`,
        };
      })}
    />
  );
}

const TARGETS = [
  { target: "next_lap", label: "Next lap" },
  { target: "next_5_mean", label: "Next-5 mean" },
] as const;

function maeOf(rows: readonly MetricRow[], model: string, split: string, target: string): number | null {
  return rows.find((r) => r.model === model && r.split === split && r.target === target)?.mae ?? null;
}

/** Model × target MAE with one column pair (val/test) per training profile. */
export function MetricsTable({ profiles }: { profiles: readonly ProfileMetrics[] }) {
  const columns = [
    { key: "model", label: "Model" },
    { key: "target", label: "Target" },
    ...profiles.flatMap((p) => [
      { key: `${p.profile}-val`, label: `${p.profile} val`, align: "right" as const },
      { key: `${p.profile}-test`, label: `${p.profile} test`, align: "right" as const },
    ]),
  ];
  const rows = MODEL_ORDER.flatMap((model) =>
    TARGETS.map(({ target, label }) => ({
      model: modelLabel(model),
      target: label,
      ...Object.fromEntries(
        profiles.flatMap((p) => [
          [`${p.profile}-val`, `${formatSeconds(maeOf(p.metrics, model, "val", target))}${p.selected_model === model && target === "next_lap" ? " ★" : ""}`],
          [`${p.profile}-test`, formatSeconds(maeOf(p.metrics, model, "test", target))],
        ]),
      ),
    })),
  );
  return <DataTable caption="MAE by model, target and profile" columns={columns} rows={rows} />;
}

export function EvaluationPage({ profile }: { profile: Profile }) {
  const metrics = useMetrics();
  const current = metrics.data?.profiles.find((p) => p.profile === profile);
  const [split, setSplit] = useState<"val" | "test">("val");
  const [groupBy, setGroupBy] = useState<GroupBy>("compound");
  const [model, setModel] = useState<ModelName | undefined>(undefined);
  const effectiveModel = model ?? current?.selected_model;
  const errors = useErrorGroups(profile, groupBy, effectiveModel);
  const threshold = useThresholdSensitivity(profile);
  const [decision, setDecision] = useState<DecisionMode>("stint_lap_5");
  const timing = usePitTiming(profile);

  const races = useRaces(profile);
  const heldOut = (races.data ?? []).filter((r) => r.split !== "train");
  const [raceId, setRaceId] = useState<string | undefined>(undefined);
  const race = raceId ?? heldOut[0]?.race_id;
  const drivers = useDrivers(profile, race);
  const [driverId, setDriverId] = useState<string | undefined>(undefined);
  const driver = drivers.data?.find((d) => d.driver_id === driverId) ?? drivers.data?.[0];
  const [stintId, setStintId] = useState<number | undefined>(undefined);
  const longestStint = driver?.stints.reduce<(typeof driver.stints)[number] | undefined>(
    (best, s) => (best === undefined || s.end_lap - s.start_lap > best.end_lap - best.start_lap ? s : best),
    undefined,
  );
  const stint = driver?.stints.find((s) => s.stint_id === stintId)?.stint_id ?? longestStint?.stint_id;
  const predictions = usePredictions(profile, race, driver?.driver_id, stint, effectiveModel);

  if (metrics.data === undefined || current === undefined) {
    return (
      <Card>
        <StatusMessage loading={metrics.isPending} error={metrics.error} empty={metrics.data ? `No metrics for ${profile}` : undefined} />
      </Card>
    );
  }

  return (
    <>
      <Card
        title="Held-out accuracy (MAE, seconds)"
        subtitle="Same validation (2024) and test (2025) races for both profiles; ★ = model selected on validation. Test races are scored once."
      >
        <MetricsTable profiles={metrics.data.profiles} />
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Error vs forecast horizon"
          subtitle={`Stay-out rollout MAE k laps ahead (${profile})`}
          actions={
            <Field label="Split">
              <select className={inputClass} value={split} onChange={(e) => { setSplit(e.target.value === "test" ? "test" : "val"); }}>
                <option value="val">Validation 2024</option>
                <option value="test">Test 2025</option>
              </select>
            </Field>
          }
        >
          <HorizonChart rows={current.horizon} split={split} />
        </Card>
        <Card
          title="Error distribution"
          subtitle="Absolute next-lap error on held-out races: median bar, IQR whiskers"
          actions={
            <div className="flex gap-2">
              <Field label="Group by">
                <select className={inputClass} value={groupBy} onChange={(e) => { setGroupBy(e.target.value as GroupBy); }}>
                  <option value="compound">Compound</option>
                  <option value="race">Race</option>
                  <option value="stint_length">Stint length</option>
                </select>
              </Field>
              <Field label="Model">
                <select className={inputClass} value={effectiveModel} onChange={(e) => { setModel(e.target.value as ModelName); }}>
                  {MODEL_ORDER.map((m) => (
                    <option key={m} value={m}>{modelLabel(m)}</option>
                  ))}
                </select>
              </Field>
            </div>
          }
        >
          {errors.data ? <ErrorDistributionChart rows={errors.data} /> : <StatusMessage loading={errors.isPending} error={errors.error} />}
        </Card>
      </div>

      <Card
        title="Predicted vs actual on a held-out stint"
        subtitle={`One-step-ahead predictions of ${modelLabel(effectiveModel ?? "")}`}
        actions={
          <div className="flex flex-wrap gap-2">
            <Field label="Race">
              <select className={inputClass} value={race ?? ""} onChange={(e) => { setRaceId(e.target.value); setDriverId(undefined); setStintId(undefined); }}>
                {heldOut.map((r) => (
                  <option key={r.race_id} value={r.race_id}>{r.season} ({r.split})</option>
                ))}
              </select>
            </Field>
            <Field label="Driver">
              <select className={inputClass} value={driver?.driver_id ?? ""} onChange={(e) => { setDriverId(e.target.value); setStintId(undefined); }}>
                {(drivers.data ?? []).map((d) => (
                  <option key={d.driver_id} value={d.driver_id}>{d.code}</option>
                ))}
              </select>
            </Field>
            <Field label="Stint">
              <select className={inputClass} value={stint ?? ""} onChange={(e) => { setStintId(Number(e.target.value)); }}>
                {(driver?.stints ?? []).map((s) => (
                  <option key={s.stint_id} value={s.stint_id}>{s.stint_id} · {s.compound} (laps {s.start_lap}–{s.end_lap})</option>
                ))}
              </select>
            </Field>
          </div>
        }
      >
        {predictions.data ? <PredVsActualChart rows={predictions.data} /> : <StatusMessage loading={predictions.isPending} error={predictions.error} />}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Remaining competitive life"
          subtitle="Laps until the tyre loss exceeds the threshold on 2 consecutive clean laps; MAE and bias (predicted − actual) on uncensored held-out stints. Censored consistency = share of stints that never crossed where the model also lasts to the stint end."
          actions={
            <Field label="Decision">
              <select
                className={inputClass}
                value={decision}
                onChange={(e) => {
                  setDecision(e.target.value === "all" ? "all" : "stint_lap_5");
                }}
              >
                {(Object.keys(DECISION_LABELS) as DecisionMode[]).map((d) => (
                  <option key={d} value={d}>{DECISION_LABELS[d]}</option>
                ))}
              </select>
            </Field>
          }
        >
          {threshold.data ? (
            <ThresholdTable rows={threshold.data} decision={decision} selectedModel={current.selected_model} />
          ) : (
            <StatusMessage loading={threshold.isPending} error={threshold.error} />
          )}
        </Card>
        <Card
          title="LORO cross-validation and ablations"
          subtitle="Leave-one-training-race-out next-lap MAE; degradation-feature ablation; fuel sensitivity"
        >
          <DataTable
            caption="LORO-CV summary"
            columns={[
              { key: "model", label: "Model" },
              { key: "mean", label: "LORO mean", align: "right" },
              { key: "std", label: "± std", align: "right" },
            ]}
            rows={current.loro_cv_summary.map((r) => ({
              model: modelLabel(r.model),
              mean: r.status === "not_run" ? "not run" : formatSeconds(r.mean),
              std: r.status === "not_run" ? "—" : formatSeconds(r.std),
            }))}
          />
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <DataTable
              caption="Ablation"
              columns={[{ key: "split", label: "GBM + deg features" }, { key: "mae", label: "MAE", align: "right" }]}
              rows={current.ablation_deg_features.filter((r) => r.split !== "train").map((r) => ({ split: r.split, mae: formatSeconds(r.mae) }))}
            />
            <DataTable
              caption="Fuel sensitivity"
              columns={[
                { key: "fuel", label: "Fuel s/lap" },
                { key: "model", label: "Model" },
                { key: "split", label: "Split" },
                { key: "mae", label: "MAE", align: "right" },
              ]}
              rows={current.fuel_sensitivity.map((r) => ({
                fuel: r.fuel_s_per_lap.toFixed(2),
                model: modelLabel(r.model),
                split: r.split,
                mae: formatSeconds(r.mae),
              }))}
            />
          </div>
          {current.normalisation_sensitivity.length > 0 && (
            <div className="mt-3">
              <DataTable
                caption="Normalisation sensitivity"
                columns={[
                  { key: "normalisation", label: "Reference pace" },
                  { key: "model", label: "Model" },
                  { key: "val", label: "Val MAE", align: "right" },
                  { key: "test", label: "Test MAE", align: "right" },
                  { key: "slope", label: "Held-out rollout slope", align: "right" },
                ]}
                rows={current.normalisation_sensitivity.map((r) => ({
                  normalisation: r.normalisation === "field" ? "Field median" : "Teammate",
                  model: modelLabel(r.model),
                  val: formatSeconds(r.val_mae),
                  test: formatSeconds(r.test_mae),
                  slope: r.held_out_rollout_s_per_lap === null ? "—" : `${r.held_out_rollout_s_per_lap.toFixed(3)} s/lap`,
                }))}
              />
            </div>
          )}
        </Card>
      </div>

      <Card
        title="Pit-timing check (descriptive only)"
        subtitle={`Window computed at stint lap 5 vs the actual in-lap; margin ${current.calibrated_margin_sigma === null ? "not calibrated" : `${current.calibrated_margin_sigma.toFixed(2)}σ, the smallest reaching ${formatPercent(current.window_target_coverage, 0)} of validation stops`}. Rates exclude safety-car, short-stint and retirement stops; ${formatPercent(current.pit_timing_in_window_rate)} of all stops fell inside.`}
      >
        <PitTimingSummaryTable metrics={current} />
        {timing.data ? (
          <div className="mt-3 max-h-80 overflow-y-auto">
            <DataTable
              caption="Pit timing"
              columns={[
                { key: "race", label: "Race" },
                { key: "driver", label: "Driver" },
                { key: "stint", label: "Stint", align: "right" },
                { key: "actual", label: "Actual in-lap", align: "right" },
                { key: "window", label: "Window", align: "right" },
                { key: "optimal", label: "Optimal", align: "right" },
                { key: "inside", label: "Inside" },
                { key: "excluded", label: "Excluded" },
              ]}
              rows={timing.data.map((r) => ({
                race: r.race_id,
                driver: r.driver_id,
                stint: r.stint_id,
                actual: r.actual_pit_lap,
                window: r.window_start !== null && r.window_end !== null ? `${r.window_start}–${r.window_end}` : "none",
                optimal: r.optimal_lap ?? "—",
                inside: r.actual_in_window ? "yes" : "no",
                excluded: r.excluded_reason?.replace("_", " ") ?? "",
              }))}
            />
          </div>
        ) : (
          <StatusMessage loading={timing.isPending} error={timing.error} />
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="80% interval coverage"
          subtitle="Share of held-out lap times inside the served interval (target 75–85%). σ is fitted on validation rollouts, so its validation coverage is in-sample; test is the fair comparison."
        >
          {current.interval_coverage.length > 0 ? <IntervalCoverageTable metrics={current} /> : <StatusMessage empty="No interval coverage in these artifacts" />}
        </Card>
        <Card
          title="Sequence model"
          subtitle="Encoder–decoder over lap history; the GRU twin is trained for comparison only and is not selectable. Laps beyond the decoder horizon follow the tyre prior."
        >
          {current.sequence_cell_comparison.length > 0 ? <SequenceCellTable metrics={current} /> : <StatusMessage empty="No sequence model in these artifacts" />}
        </Card>
      </div>

      {current.pit_timing_by_model.length > 0 && (
        <Card
          title="Pit timing by model"
          subtitle="Same descriptive check for every model, each with the smallest margin reaching the validation target (kept stops only)"
        >
          <PitTimingByModelTable metrics={current} />
        </Card>
      )}

      <Card title="Assumptions">
        <AssumptionsPanel metrics={current} />
      </Card>
    </>
  );
}
