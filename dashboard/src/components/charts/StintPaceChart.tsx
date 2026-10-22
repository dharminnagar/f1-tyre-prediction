import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { LapRecord, TyreState } from "../../api/client";
import { compoundColor, COMPOUND_LETTER } from "../../lib/colors";
import { formatLapTime } from "../../lib/format";
import { paceDomain, pacePoints, type PacePoint } from "../../lib/transform";
import { TableView } from "../ui";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { Legend, LegendItem, TooltipBox, TooltipRow } from "./chartTheme";

const COMPOUNDS = ["SOFT", "MEDIUM", "HARD"] as const;

export function StintPaceChart({
  laps,
  state,
  decisionLap,
}: {
  laps: readonly LapRecord[];
  state: TyreState | undefined;
  decisionLap: number;
}) {
  const points = useMemo(() => pacePoints(laps, state), [laps, state]);
  const byLap = useMemo(() => new Map(points.map((p) => [p.lap, p])), [points]);
  const domain = useMemo(() => paceDomain(points), [points]);
  const pitLaps = laps.filter((l) => l.pit_in).map((l) => l.lap);

  return (
    <div>
      <Legend>
        <LegendItem color="var(--text-secondary)" label="Clean lap (dot colour = compound)" />
        <LegendItem color="var(--text-muted)" label="Excluded lap (hollow)" />
        <LegendItem color="var(--accent)" label="Forecast ± 80% band" dashed />
        <LegendItem color="var(--text-muted)" label="Actual pit stop" dashed />
      </Legend>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 20, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="lap" type="number" domain={["dataMin", "dataMax"]} {...AXIS_PROPS} />
            <YAxis
              domain={domain}
              allowDataOverflow
              width={56}
              tickFormatter={(v: number) => formatLapTime(v).slice(0, -2)}
              {...AXIS_PROPS}
            />
            {pitLaps.map((lap) => (
              <ReferenceLine key={lap} x={lap + 0.5} stroke="var(--text-muted)" strokeDasharray="4 3" />
            ))}
            <ReferenceLine
              x={decisionLap}
              stroke="var(--text-primary)"
              label={{ value: "now", position: "top", fill: "var(--text-secondary)", fontSize: 11 }}
            />
            <Area dataKey="band" stroke="none" fill="var(--window)" isAnimationActive={false} />
            <Line
              dataKey="forecast"
              stroke="var(--accent)"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={false}
              isAnimationActive={false}
              connectNulls={false}
            />
            {COMPOUNDS.map((compound) => (
              <Line
                key={compound}
                dataKey={(p: PacePoint) => (p.compound === compound ? p.clean : null)}
                name={compound}
                stroke="none"
                dot={{ r: 3.5, fill: compoundColor(compound), stroke: "var(--surface-1)", strokeWidth: 1 }}
                activeDot={false}
                isAnimationActive={false}
              />
            ))}
            <Line
              dataKey={(p: PacePoint) => (p.excluded !== null && p.excluded <= domain[1] ? p.excluded : null)}
              name="excluded"
              stroke="none"
              dot={{ r: 3.5, fill: "none", stroke: "var(--text-muted)", strokeWidth: 1.5 }}
              activeDot={false}
              isAnimationActive={false}
            />
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, label }) => {
                const point = typeof label === "number" ? byLap.get(label) : undefined;
                if (!active || point === undefined) return null;
                const time = point.clean ?? point.excluded;
                return (
                  <TooltipBox title={`Lap ${point.lap} · ${COMPOUND_LETTER[point.compound] ?? ""} ${point.compound}`}>
                    <TooltipRow label="Actual" value={formatLapTime(time)} />
                    {point.forecast !== null && (
                      <TooltipRow color="var(--accent)" label="Forecast" value={formatLapTime(point.forecast)} />
                    )}
                    {point.reason !== null && <TooltipRow label="Excluded" value={point.reason} />}
                  </TooltipBox>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <TableView
        caption="Lap times"
        columns={[
          { key: "lap", label: "Lap", align: "right" },
          { key: "compound", label: "Compound" },
          { key: "time", label: "Lap time", align: "right" },
          { key: "forecast", label: "Forecast", align: "right" },
          { key: "reason", label: "Excluded because" },
        ]}
        rows={points.map((p) => ({
          lap: p.lap,
          compound: p.compound,
          time: formatLapTime(p.clean ?? p.excluded),
          forecast: formatLapTime(p.forecast),
          reason: p.reason ?? "",
        }))}
      />
    </div>
  );
}
