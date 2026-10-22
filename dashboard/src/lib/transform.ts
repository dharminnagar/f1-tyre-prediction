import type {
  CounterfactualResponse,
  CurvePoint,
  LapRecord,
  PitCandidate,
  TyreState,
} from "../api/client";

export interface PacePoint {
  lap: number;
  clean: number | null;
  excluded: number | null;
  forecast: number | null;
  band: [number, number] | null;
  compound: string;
  reason: string | null;
}

/** Actual laps (clean vs excluded) plus the next-N forecast band from the decision lap. */
export function pacePoints(laps: readonly LapRecord[], state: TyreState | undefined): PacePoint[] {
  const forecast = new Map((state?.next_laps ?? []).map((i) => [i.lap, i]));
  return laps.map((lap) => {
    const predicted = forecast.get(lap.lap);
    return {
      lap: lap.lap,
      clean: lap.is_clean ? lap.lap_time_s : null,
      excluded: lap.is_clean ? null : lap.lap_time_s,
      forecast: predicted?.pred_s ?? null,
      band: predicted ? [predicted.lo_s, predicted.hi_s] : null,
      compound: lap.compound,
      reason: lap.exclusion_reason,
    };
  });
}

/** Y-domain that focuses on racing pace (clean-lap range with padding). */
export function paceDomain(points: readonly PacePoint[]): [number, number] {
  const values = points.flatMap((p) => [p.clean, p.forecast, ...(p.band ?? [])]).filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  if (values.length === 0) return [0, 1];
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = Math.max(0.3, (high - low) * 0.08);
  return [Math.floor((low - pad) * 10) / 10, Math.ceil((high + pad) * 10) / 10];
}

export interface CostPoint {
  pitLap: number;
  costVsBest: number;
  margin: number;
  inWindow: boolean;
  advantage: number;
}

/** Pit cost relative to the cheapest candidate (seconds lost by boxing on that lap instead). */
export function costCurve(candidates: readonly PitCandidate[]): CostPoint[] {
  if (candidates.length === 0) return [];
  const best = Math.min(...candidates.map((c) => c.cost_pit_s));
  return candidates.map((c) => ({
    pitLap: c.pit_lap,
    costVsBest: c.cost_pit_s - best,
    margin: c.margin_s,
    inWindow: c.in_window,
    advantage: c.advantage_s,
  }));
}

export interface CounterfactualPoint {
  lap: number;
  delta: number;
  stayOut: number;
  pit: number;
  actual: number | null;
}

export function counterfactualPoints(cf: CounterfactualResponse): CounterfactualPoint[] {
  return cf.laps.map((lap, i) => ({
    lap,
    delta: cf.cumulative_delta_s[i] ?? 0,
    stayOut: cf.stay_out[i] ?? 0,
    pit: cf.pit[i] ?? 0,
    actual: cf.actual[i] ?? null,
  }));
}

export interface CurveRow {
  tyreAge: number;
  [series: string]: number | [number, number] | null;
}

/** Pivot curve points to one row per tyre age with per-compound median/IQR/fit columns. */
export function pivotCurves(points: readonly CurvePoint[], minCount = 5): CurveRow[] {
  const rows = new Map<number, CurveRow>();
  for (const p of points) {
    if (p.n < minCount) continue;
    const row = rows.get(p.tyre_age) ?? { tyreAge: p.tyre_age };
    row[`${p.compound}_median`] = p.median_s;
    row[`${p.compound}_iqr`] = [p.q25_s, p.q75_s];
    row[`${p.compound}_fit`] = p.fitted_s;
    rows.set(p.tyre_age, row);
  }
  return [...rows.values()].sort((a, b) => a.tyreAge - b.tyreAge);
}

/** Default decision lap: a few laps into the driver's first stint. */
export function defaultLap(laps: readonly LapRecord[]): number | undefined {
  const firstStint = laps.filter((l) => l.stint_id === 1);
  const candidate = firstStint.find((l) => l.stint_lap === 8) ?? laps[Math.min(8, laps.length - 2)];
  return candidate?.lap;
}
