import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PredictionRow } from "../../api/client";
import { formatDelta, formatLapTime } from "../../lib/format";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { Legend, LegendItem, TooltipBox, TooltipRow } from "./chartTheme";

/** One-step-ahead predictions with the ~80% interval against actual lap times over a stint. */
export function PredVsActualChart({ rows }: { rows: readonly PredictionRow[] }) {
  const data = useMemo(
    () => rows.map((r) => ({ lap: r.lap, actual: r.actual_s, pred: r.pred_s, band: [r.lo_s, r.hi_s] as [number, number] })),
    [rows],
  );
  const byLap = useMemo(() => new Map(data.map((d) => [d.lap, d])), [data]);
  return (
    <div>
      <Legend>
        <LegendItem color="var(--text-primary)" label="Actual (clean laps)" />
        <LegendItem color="var(--accent)" label="Predicted ± 80% band" />
      </Legend>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="lap" type="number" domain={["dataMin", "dataMax"]} {...AXIS_PROPS} />
            <YAxis domain={["auto", "auto"]} width={56} tickFormatter={(v: number) => formatLapTime(v).slice(0, -2)} {...AXIS_PROPS} />
            <Area dataKey="band" stroke="none" fill="var(--window)" isAnimationActive={false} />
            <Line dataKey="pred" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line
              dataKey="actual"
              stroke="none"
              dot={{ r: 3, fill: "var(--text-primary)" }}
              activeDot={false}
              isAnimationActive={false}
            />
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, label }) => {
                const d = typeof label === "number" ? byLap.get(label) : undefined;
                if (!active || d === undefined) return null;
                return (
                  <TooltipBox title={`Lap ${d.lap}`}>
                    <TooltipRow label="Actual" value={formatLapTime(d.actual)} />
                    <TooltipRow color="var(--accent)" label="Predicted" value={formatLapTime(d.pred)} />
                    <TooltipRow label="Error" value={formatDelta(d.pred - d.actual, 3)} />
                  </TooltipBox>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
