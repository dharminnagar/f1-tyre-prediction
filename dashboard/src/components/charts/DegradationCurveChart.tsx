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

import type { CurvePoint } from "../../api/client";
import { compoundColor, COMPOUND_LETTER } from "../../lib/colors";
import { formatDelta } from "../../lib/format";
import { pivotCurves } from "../../lib/transform";
import { TableView } from "../ui";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { Legend, LegendItem, TooltipBox, TooltipRow } from "./chartTheme";

const COMPOUNDS = ["SOFT", "MEDIUM", "HARD"] as const;

/** Median relative pace (driver-adjusted) by tyre age, IQR band and model fit per compound. */
export function DegradationCurveChart({ points }: { points: readonly CurvePoint[] }) {
  const rows = useMemo(() => pivotCurves(points), [points]);
  const present = COMPOUNDS.filter((c) => rows.some((r) => r[`${c}_median`] !== undefined));
  const byAge = useMemo(() => new Map(rows.map((r) => [r.tyreAge, r])), [rows]);

  return (
    <div>
      <Legend>
        {present.map((c) => (
          <LegendItem key={c} color={compoundColor(c)} label={`${COMPOUND_LETTER[c] ?? ""} ${c} median`} />
        ))}
        <LegendItem color="var(--text-secondary)" label="Model fit" dashed />
      </Legend>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis
              dataKey="tyreAge"
              type="number"
              domain={["dataMin", "dataMax"]}
              label={{ value: "Tyre age (laps)", position: "insideBottom", offset: -2, fill: "var(--text-muted)", fontSize: 11 }}
              {...AXIS_PROPS}
            />
            <YAxis width={44} tickFormatter={(v: number) => `${v.toFixed(1)}s`} {...AXIS_PROPS} />
            {present.map((c) => (
              <Area key={`${c}-iqr`} dataKey={`${c}_iqr`} stroke="none" fill={compoundColor(c)} fillOpacity={0.12} isAnimationActive={false} connectNulls />
            ))}
            {present.map((c) => (
              <Line key={`${c}-median`} dataKey={`${c}_median`} stroke={compoundColor(c)} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} connectNulls />
            ))}
            {present.map((c) => (
              <Line key={`${c}-fit`} dataKey={`${c}_fit`} stroke={compoundColor(c)} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} connectNulls />
            ))}
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, label }) => {
                const row = typeof label === "number" ? byAge.get(label) : undefined;
                if (!active || row === undefined) return null;
                return (
                  <TooltipBox title={`Tyre age ${row.tyreAge}`}>
                    {present.map((c) => {
                      const median = row[`${c}_median`];
                      const fit = row[`${c}_fit`];
                      if (typeof median !== "number") return null;
                      return (
                        <TooltipRow
                          key={c}
                          color={compoundColor(c)}
                          label={c}
                          value={`${formatDelta(median)} (fit ${typeof fit === "number" ? formatDelta(fit) : "—"})`}
                        />
                      );
                    })}
                  </TooltipBox>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <TableView
        caption="Degradation curves"
        columns={[
          { key: "compound", label: "Compound" },
          { key: "age", label: "Tyre age", align: "right" },
          { key: "median", label: "Median", align: "right" },
          { key: "iqr", label: "IQR", align: "right" },
          { key: "fit", label: "Model fit", align: "right" },
          { key: "n", label: "Laps", align: "right" },
        ]}
        rows={points.map((p) => ({
          compound: p.compound,
          age: p.tyre_age,
          median: formatDelta(p.median_s),
          iqr: `${formatDelta(p.q25_s)} … ${formatDelta(p.q75_s)}`,
          fit: formatDelta(p.fitted_s),
          n: p.n,
        }))}
      />
    </div>
  );
}
