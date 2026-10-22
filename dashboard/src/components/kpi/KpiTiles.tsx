import type { ReactNode } from "react";

import type { PitWindowResponse, TyreState } from "../../api/client";
import { formatDelta, formatLapTime } from "../../lib/format";
import { CompoundBadge } from "../ui";
import { Sparkline } from "./Sparkline";

function Tile({ label, children, hint }: { label: string; children: ReactNode; hint?: string | undefined }) {
  return (
    <div className="min-w-0 rounded-xl border border-[var(--border)] bg-[var(--surface-1)] p-3">
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
      <div className="tabular mt-1 text-lg font-semibold text-[var(--text-primary)]">{children}</div>
      {hint !== undefined && <div className="mt-0.5 text-xs text-[var(--text-muted)]">{hint}</div>}
    </div>
  );
}

export function KpiTiles({ state, window }: { state: TyreState; window: PitWindowResponse | undefined }) {
  const life = state.remaining_competitive_laps;
  const next = state.next_lap;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
      <Tile label="Tyre" hint={`stint ${state.stint_id}, stint lap ${state.stint_lap}`}>
        <span className="flex items-center gap-2">
          <CompoundBadge compound={state.compound} />
          <span className="text-sm font-normal text-[var(--text-secondary)]">{state.tyre_age.toFixed(0)} laps</span>
        </span>
      </Tile>
      <Tile
        label="Degradation"
        hint={`recent laps ${state.degradation.empirical_s_per_lap === null ? "—" : formatDelta(state.degradation.empirical_s_per_lap, 3)}`}
      >
        {formatDelta(state.degradation.model_s_per_lap, 3)}/lap
      </Tile>
      <Tile label={`Next lap (${next.lap})`} hint={`80% ${next.method === "quantile" ? "quantile" : "±σ"}: ${formatLapTime(next.lo_s)} – ${formatLapTime(next.hi_s)}`}>
        {formatLapTime(next.pred_s)}
      </Tile>
      <Tile label={`Next ${state.next_laps.length} laps`} hint={state.next_laps.some((i) => i.extrapolated) ? "beyond training tyre ages" : undefined}>
        <Sparkline points={state.next_laps} />
      </Tile>
      <Tile
        label="Competitive laps left"
        hint={`within ${life.threshold_s.toFixed(2)} s of a 3-lap-old set`}
      >
        {life.capped_by_race_end ? `${life.value} (to flag)` : `~${life.value}`}
      </Tile>
      <Tile
        label="Pit window"
        hint={window ? `optimal lap ${window.optimal_lap ?? "—"} · ${window.replacement_compound}` : undefined}
      >
        {window === undefined ? "…" : window.window === null ? "none" : `Lap ${window.window.start_lap}–${window.window.end_lap}`}
      </Tile>
    </div>
  );
}
