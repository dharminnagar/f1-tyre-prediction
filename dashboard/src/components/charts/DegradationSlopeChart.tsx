import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { LapRecord } from "../../api/client";
import { formatDelta } from "../../lib/format";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { TooltipBox, TooltipRow } from "./chartTheme";

/** Causal 5-lap degradation slope over the current stint (acceleration shown on hover). */
export function DegradationSlopeChart({ laps, stintId }: { laps: readonly LapRecord[]; stintId: number }) {
  const points = useMemo(
    () =>
      laps
        .filter((l) => l.stint_id === stintId)
        .map((l) => ({ age: l.tyre_age, slope: l.deg_slope_5, accel: l.deg_accel, lap: l.lap })),
    [laps, stintId],
  );
  const byAge = useMemo(() => new Map(points.map((p) => [p.age, p])), [points]);
  return (
    <div className="h-48">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="age" type="number" domain={["dataMin", "dataMax"]} {...AXIS_PROPS} />
          <YAxis domain={[-0.5, 0.5]} allowDataOverflow width={44} tickFormatter={(v: number) => v.toFixed(1)} {...AXIS_PROPS} />
          <ReferenceLine y={0} stroke="var(--text-muted)" />
          <Line dataKey="slope" stroke="var(--accent)" strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
          <Tooltip
            cursor={{ stroke: "var(--border)" }}
            content={({ active, label }) => {
              const p = typeof label === "number" ? byAge.get(label) : undefined;
              if (!active || p === undefined) return null;
              return (
                <TooltipBox title={`Tyre age ${p.age} (lap ${p.lap})`}>
                  <TooltipRow color="var(--accent)" label="Slope" value={`${formatDelta(p.slope, 3)}/lap`} />
                  <TooltipRow label="Acceleration" value={`${formatDelta(p.accel, 3)}/lap²`} />
                </TooltipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
