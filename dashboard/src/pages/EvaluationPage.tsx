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
        <Card title="Competitive-threshold sensitivity" subtitle="Remaining-life MAE in laps on uncensored held-out decisions">
          {threshold.data ? (
            <DataTable
              caption="Threshold sensitivity"
              columns={[
                { key: "threshold", label: "Threshold", align: "right" },
                ...MODEL_ORDER.map((m) => ({ key: m, label: modelLabel(m), align: "right" as const })),
                { key: "n", label: "Uncensored", align: "right" },
              ]}
              rows={[...new Set(threshold.data.map((r) => r.threshold_s))].map((t) => {
                const at = threshold.data.filter((r) => r.threshold_s === t);
                return {
                  threshold: `${t.toFixed(2)} s`,
                  ...Object.fromEntries(MODEL_ORDER.map((m) => [m, at.find((r) => r.model === m)?.mae_laps?.toFixed(1) ?? "—"])),
                  n: at[0] ? `${at[0].n_uncensored}/${at[0].n_total}` : "—",
                };
              })}
            />
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
            rows={current.loro_cv_summary.map((r) => ({ model: modelLabel(r.model), mean: formatSeconds(r.mean), std: formatSeconds(r.std) }))}
          />
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <DataTable
              caption="Ablation"
              columns={[{ key: "split", label: "GBM + deg features" }, { key: "mae", label: "MAE", align: "right" }]}
              rows={current.ablation_deg_features.filter((r) => r.split !== "train").map((r) => ({ split: r.split, mae: formatSeconds(r.mae) }))}
            />
            <DataTable
              caption="Fuel sensitivity"
              columns={[{ key: "fuel", label: "Fuel s/lap" }, { key: "split", label: "Split" }, { key: "mae", label: "MAE", align: "right" }]}
              rows={current.fuel_sensitivity.map((r) => ({ fuel: r.fuel_s_per_lap.toFixed(2), split: r.split, mae: formatSeconds(r.mae) }))}
            />
          </div>
        </Card>
      </div>

      <Card
        title="Pit-timing check (descriptive only)"
        subtitle={`Window computed at stint lap 5 vs the actual in-lap; ${formatPercent(current.pit_timing_in_window_rate)} of stops fell inside`}
      >
        {timing.data ? (
          <div className="max-h-80 overflow-y-auto">
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
              ]}
              rows={timing.data.map((r) => ({
                race: r.race_id,
                driver: r.driver_id,
                stint: r.stint_id,
                actual: r.actual_pit_lap,
                window: r.window_start !== null && r.window_end !== null ? `${r.window_start}–${r.window_end}` : "none",
                optimal: r.optimal_lap ?? "—",
                inside: r.actual_in_window ? "yes" : "no",
              }))}
            />
          </div>
        ) : (
          <StatusMessage loading={timing.isPending} error={timing.error} />
        )}
      </Card>

      <Card title="Assumptions">
        <AssumptionsPanel metrics={current} />
      </Card>
    </>
  );
}
