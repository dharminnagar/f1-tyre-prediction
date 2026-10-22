import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { PitWindowResponse } from "../../api/client";
import { formatDelta } from "../../lib/format";
import { costCurve } from "../../lib/transform";
import { TableView } from "../ui";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { Legend, LegendItem, TooltipBox, TooltipRow } from "./chartTheme";

/**
 * Race-time cost of boxing on each candidate lap relative to the best lap, window shaded.
 * Candidates whose plan runs a tyre past the model's training age cap are overdrawn in a
 * warning colour: their cost relies on extrapolated pace.
 */
export function PitCostChart({ result }: { result: PitWindowResponse }) {
  const points = useMemo(() => costCurve(result.candidates), [result.candidates]);
  const byLap = useMemo(() => new Map(points.map((p) => [p.pitLap, p])), [points]);
  const upper = Math.min(60, Math.max(5, ...points.map((p) => p.costVsBest)));
  const anyPastCap = points.some((p) => p.costPastCap !== null);

  return (
    <div>
      <Legend>
        <LegendItem color="var(--accent)" label="Cost vs best pit lap" />
        <LegendItem color="var(--text-muted)" label="Uncertainty margin" dashed />
        <LegendItem color="var(--text-primary)" label="Optimal lap" />
        <LegendItem color="var(--compound-soft)" label="Actual stop" dashed />
        {anyPastCap && <LegendItem color="var(--series-2)" label="Plan past tyre-age cap (extrapolated)" />}
      </Legend>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="pitLap" type="number" domain={["dataMin", "dataMax"]} {...AXIS_PROPS} />
            <YAxis
              domain={[0, upper]}
              allowDataOverflow
              width={44}
              tickFormatter={(v: number) => `${v.toFixed(0)}s`}
              {...AXIS_PROPS}
            />
            {result.window !== null && (
              <ReferenceArea
                x1={result.window.start_lap}
                x2={result.window.end_lap}
                fill="var(--window)"
                stroke="none"
                label={{ value: "pit window", position: "insideTop", fill: "var(--text-secondary)", fontSize: 11 }}
              />
            )}
            <Line dataKey="margin" stroke="var(--text-muted)" strokeDasharray="4 3" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Line dataKey="costVsBest" stroke="var(--accent)" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="costPastCap" stroke="var(--series-2)" strokeWidth={3} dot={false} connectNulls={false} isAnimationActive={false} />
            {result.optimal_lap !== null && (
              <ReferenceLine x={result.optimal_lap} stroke="var(--text-primary)" strokeWidth={1.5} />
            )}
            {result.actual_pit_laps.map((lap) => (
              <ReferenceLine key={lap} x={lap} stroke="var(--compound-soft)" strokeDasharray="4 3" />
            ))}
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, label }) => {
                const p = typeof label === "number" ? byLap.get(label) : undefined;
                if (!active || p === undefined) return null;
                return (
                  <TooltipBox title={`Box on lap ${p.pitLap}${p.inWindow ? " · in window" : ""}`}>
                    <TooltipRow color="var(--accent)" label="Cost vs best" value={formatDelta(p.costVsBest, 1)} />
                    <TooltipRow color="var(--text-muted)" label="Margin" value={formatDelta(p.margin, 1)} />
                    <TooltipRow label="Gain vs no stop" value={formatDelta(p.advantage, 1)} />
                    {p.lapsPastCap > 0 && (
                      <TooltipRow color="var(--series-2)" label="Laps past tyre-age cap" value={String(p.lapsPastCap)} />
                    )}
                  </TooltipBox>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <TableView
        caption="Pit cost by lap"
        columns={[
          { key: "lap", label: "Pit lap", align: "right" },
          { key: "cost", label: "Cost vs best", align: "right" },
          { key: "margin", label: "Margin", align: "right" },
          { key: "adv", label: "Gain vs no stop", align: "right" },
          { key: "pastCap", label: "Laps past cap", align: "right" },
          { key: "window", label: "In window" },
        ]}
        rows={points.map((p) => ({
          lap: p.pitLap,
          cost: formatDelta(p.costVsBest, 1),
          margin: formatDelta(p.margin, 1),
          adv: formatDelta(p.advantage, 1),
          pastCap: p.lapsPastCap > 0 ? p.lapsPastCap : "",
          window: p.inWindow ? "yes" : "",
        }))}
      />
    </div>
  );
}
