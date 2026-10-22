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

import type { CounterfactualResponse } from "../../api/client";
import { formatDelta, formatLapTime } from "../../lib/format";
import { counterfactualPoints } from "../../lib/transform";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { TooltipBox, TooltipRow } from "./chartTheme";

/** Cumulative time of "box on lap X" minus "stay out": below zero means the stop is ahead. */
export function CounterfactualChart({ data, pitLap }: { data: CounterfactualResponse; pitLap: number }) {
  const points = useMemo(() => counterfactualPoints(data), [data]);
  const byLap = useMemo(() => new Map(points.map((p) => [p.lap, p])), [points]);
  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="lap" type="number" domain={["dataMin", "dataMax"]} {...AXIS_PROPS} />
          <YAxis width={44} tickFormatter={(v: number) => `${v.toFixed(0)}s`} {...AXIS_PROPS} />
          <ReferenceLine y={0} stroke="var(--text-muted)" />
          <ReferenceLine x={pitLap} stroke="var(--compound-soft)" strokeDasharray="4 3" />
          <Line dataKey="delta" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
          <Tooltip
            cursor={{ stroke: "var(--border)" }}
            content={({ active, label }) => {
              const p = typeof label === "number" ? byLap.get(label) : undefined;
              if (!active || p === undefined) return null;
              return (
                <TooltipBox title={`Lap ${p.lap}`}>
                  <TooltipRow color="var(--accent)" label="Pit − stay (cum.)" value={formatDelta(p.delta, 1)} />
                  <TooltipRow label="Stay out" value={formatLapTime(p.stayOut)} />
                  <TooltipRow label="Pit branch" value={formatLapTime(p.pit)} />
                  <TooltipRow label="Actual" value={formatLapTime(p.actual)} />
                </TooltipBox>
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
