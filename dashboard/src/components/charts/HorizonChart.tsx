import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { HorizonRow, ModelName } from "../../api/client";
import { MODEL_COLOR, MODEL_ORDER } from "../../lib/colors";
import { formatSeconds, modelLabel } from "../../lib/format";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { Legend, LegendItem, TooltipBox, TooltipRow } from "./chartTheme";

/** Forecast MAE versus laps ahead for each model on one split. */
export function HorizonChart({ rows, split }: { rows: readonly HorizonRow[]; split: "val" | "test" }) {
  const data = useMemo(() => {
    const byK = new Map<number, Partial<Record<ModelName, number>> & { k: number }>();
    for (const r of rows.filter((row) => row.split === split)) {
      const entry = byK.get(r.k) ?? { k: r.k };
      entry[r.model] = r.mae;
      byK.set(r.k, entry);
    }
    return [...byK.values()].sort((a, b) => a.k - b.k);
  }, [rows, split]);
  const byK = useMemo(() => new Map(data.map((d) => [d.k, d])), [data]);
  return (
    <div>
      <Legend>
        {MODEL_ORDER.map((m) => (
          <LegendItem key={m} color={MODEL_COLOR[m]} label={modelLabel(m)} />
        ))}
      </Legend>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="k" type="number" domain={[1, "dataMax"]} {...AXIS_PROPS} />
            <YAxis width={44} domain={[0, "auto"]} tickFormatter={(v: number) => v.toFixed(2)} {...AXIS_PROPS} />
            {MODEL_ORDER.map((m) => (
              <Line key={m} dataKey={m} stroke={MODEL_COLOR[m]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            ))}
            <Tooltip
              cursor={{ stroke: "var(--border)" }}
              content={({ active, label }) => {
                const d = typeof label === "number" ? byK.get(label) : undefined;
                if (!active || d === undefined) return null;
                return (
                  <TooltipBox title={`${label} lap${label === 1 ? "" : "s"} ahead`}>
                    {MODEL_ORDER.map((m) => (
                      <TooltipRow key={m} color={MODEL_COLOR[m]} label={modelLabel(m)} value={formatSeconds(d[m])} />
                    ))}
                  </TooltipBox>
                );
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
