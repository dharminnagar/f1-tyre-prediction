import { useMemo, useState } from "react";

import type { Profile } from "../api/client";
import {
  useCounterfactual,
  useDrivers,
  useLaps,
  useModels,
  usePitWindow,
  useRaces,
  useTyreState,
} from "../api/queries";
import { CounterfactualChart } from "../components/charts/CounterfactualChart";
import { DegradationSlopeChart } from "../components/charts/DegradationSlopeChart";
import { PitCostChart } from "../components/charts/PitCostChart";
import { StintPaceChart } from "../components/charts/StintPaceChart";
import { WhyPanel } from "../components/explain/WhyPanel";
import { KpiTiles } from "../components/kpi/KpiTiles";
import { Selectors } from "../components/selectors/Selectors";
import { Card, Field, inputClass, StatusMessage } from "../components/ui";
import { formatDelta } from "../lib/format";
import { useSelection } from "../lib/selection";
import { defaultLap, planLabel } from "../lib/transform";

export function PitWallPage({ profile }: { profile: Profile }) {
  const [selection, update] = useSelection();
  const races = useRaces(profile);
  const raceId = selection.raceId ?? races.data?.[0]?.race_id;
  const drivers = useDrivers(profile, raceId);
  const driverId = selection.driverId ?? drivers.data?.[0]?.driver_id;
  const laps = useLaps(profile, raceId, driverId);
  const models = useModels(profile);
  const race = races.data?.find((r) => r.race_id === raceId);
  const totalLaps = race?.total_laps ?? 2;
  const lastLap = laps.data?.[laps.data.length - 1]?.lap ?? totalLaps;
  const lap = Math.min(selection.lap ?? defaultLap(laps.data ?? []) ?? 2, Math.max(2, Math.min(lastLap, totalLaps - 1)));
  // Strategy views default to the model with the best multi-lap (not next-lap) accuracy.
  const strategyModel = models.data?.find((m) => m.selected_for_strategy)?.name;
  const model = selection.model ?? strategyModel;

  const decision =
    raceId !== undefined && driverId !== undefined && laps.data !== undefined && models.data !== undefined
      ? { profile, raceId, driverId, lap, model }
      : undefined;
  const state = useTyreState(decision);
  const pitWindow = usePitWindow(
    decision && {
      race_id: decision.raceId,
      driver_id: decision.driverId,
      lap,
      model: model ?? null,
      profile,
    },
  );

  const [pitLapOverride, setPitLapOverride] = useState<number | undefined>(undefined);
  const optimal = pitWindow.data?.optimal_lap ?? undefined;
  const pitLap = Math.min(Math.max(pitLapOverride ?? optimal ?? lap + 1, lap + 1), totalLaps - 1);
  // The counterfactual fits the set the optimal plan chooses at its first stop.
  const planCompound = pitWindow.data?.plan[0]?.compound ?? null;
  const counterfactual = useCounterfactual(
    decision && pitLap > lap && pitLap < totalLaps
      ? {
          race_id: decision.raceId,
          driver_id: decision.driverId,
          lap,
          pit_lap: pitLap,
          model: model ?? null,
          profile,
          replacement_compound: planCompound,
        }
      : undefined,
  );

  const finalDelta = useMemo(() => counterfactual.data?.cumulative_delta_s.at(-1), [counterfactual.data]);

  return (
    <>
      <Card>
        {races.data && models.data ? (
          <Selectors
            selection={{ ...selection, raceId, driverId, lap }}
            update={(patch) => {
              setPitLapOverride(undefined);
              update(patch);
            }}
            races={races.data}
            drivers={drivers.data ?? []}
            models={models.data}
            defaultModel={strategyModel}
            maxLap={totalLaps}
          />
        ) : (
          <StatusMessage loading={races.isPending} error={races.error ?? models.error} />
        )}
      </Card>

      {state.data ? (
        <KpiTiles state={state.data} window={pitWindow.data} />
      ) : (
        <Card>
          <StatusMessage loading={state.isPending && decision !== undefined} error={state.error} />
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-5">
        <Card
          className="lg:col-span-3"
          title="Stint pace and forecast"
          subtitle="Actual lap times, the decision lap and the next-lap forecast with its ~80% interval"
        >
          {laps.data ? (
            <StintPaceChart laps={laps.data} state={state.data} decisionLap={lap} />
          ) : (
            <StatusMessage loading={laps.isPending} error={laps.error} />
          )}
        </Card>
        <Card className="lg:col-span-2" title="Why" subtitle="What drives the estimate at this lap">
          {state.data ? <WhyPanel state={state.data} /> : <StatusMessage loading={state.isPending} error={state.error} />}
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card
          title="Pit cost curve"
          subtitle={
            pitWindow.data
              ? `Seconds lost vs the best pit lap · first stop on ${pitWindow.data.replacement_compound} · pit loss ${pitWindow.data.pit_loss_s.toFixed(1)} s${pitWindow.data.stay_out_feasible ? "" : " · must stop (two-compound rule)"}`
              : "Seconds lost vs the best pit lap"
          }
        >
          {pitWindow.data ? (
            <>
              <PitCostChart result={pitWindow.data} />
              <p className="tabular mt-1 text-xs text-[var(--text-muted)]">
                {pitWindow.data.plan.length > 0
                  ? `Best plan: ${planLabel(pitWindow.data.plan)} (${pitWindow.data.plan.length} stop${pitWindow.data.plan.length > 1 ? "s" : ""} from here)`
                  : "No pit stop left to plan."}
                {` · margin ${pitWindow.data.assumptions.margin_sigma.toFixed(2)}σ (${pitWindow.data.assumptions.margin_source})`}
              </p>
            </>
          ) : (
            <StatusMessage loading={pitWindow.isPending} error={pitWindow.error} />
          )}
        </Card>
        <Card
          title="Counterfactual: box on lap X vs stay out"
          subtitle="Cumulative time of the pit branch minus staying out (below zero = the stop has paid back)"
          actions={
            <Field label="Pit lap">
              <input
                type="number"
                className={`${inputClass} w-20`}
                min={lap + 1}
                max={totalLaps - 1}
                value={pitLap}
                onChange={(e) => {
                  setPitLapOverride(Number(e.target.value));
                }}
              />
            </Field>
          }
        >
          {counterfactual.data ? (
            <>
              <CounterfactualChart data={counterfactual.data} pitLap={pitLap} />
              <p className="tabular mt-1 text-xs text-[var(--text-muted)]">
                At the flag: {formatDelta(finalDelta ?? null, 1)} for boxing on lap {pitLap} (new {counterfactual.data.replacement_compound}).
              </p>
            </>
          ) : (
            <StatusMessage loading={counterfactual.isPending && decision !== undefined} error={counterfactual.error} />
          )}
        </Card>
      </div>

      <Card title="Degradation slope over the stint" subtitle="Causal 5-lap slope of clean-lap relative pace (s per lap of tyre age)">
        {laps.data && state.data ? (
          <DegradationSlopeChart laps={laps.data} stintId={state.data.stint_id} />
        ) : (
          <StatusMessage loading={laps.isPending} error={laps.error} />
        )}
      </Card>
    </>
  );
}
