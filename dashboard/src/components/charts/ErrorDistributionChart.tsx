import { useMemo } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ErrorGroupRow } from "../../api/client";
import { formatSeconds } from "../../lib/format";
import { TableView } from "../ui";
import { AXIS_PROPS, GRID_PROPS } from "./chartStyle";
import { TooltipBox, TooltipRow } from "./chartTheme";

/** Median absolute next-lap error per group with the interquartile range as whiskers. */
export function ErrorDistributionChart({ rows }: { rows: readonly ErrorGroupRow[] }) {
  const data = useMemo(
    () =>
      rows.map((r) => ({
        group: r.group,
        median: r.q50_s,
        iqr: [r.q50_s - r.q25_s, r.q75_s - r.q50_s] as [number, number],
        row: r,
      })),
    [rows],
  );
  const byGroup = useMemo(() => new Map(data.map((d) => [d.group, d.row])), [data]);
  return (
    <div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid {...GRID_PROPS} />
            <XAxis dataKey="group" type="category" {...AXIS_PROPS} />
            <YAxis width={44} tickFormatter={(v: number) => v.toFixed(2)} {...AXIS_PROPS} />
            <Bar dataKey="median" fill="var(--accent)" radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false}>
              <ErrorBar dataKey="iqr" stroke="var(--text-secondary)" width={6} />
            </Bar>
            <Tooltip
              cursor={{ fill: "var(--surface-2)" }}
              content={({ active, label }) => {
                const r = typeof label === "string" ? byGroup.get(label) : undefined;
                if (!active || r === undefined) return null;
                return (
                  <TooltipBox title={`${r.group} · ${r.n} laps`}>
                    <TooltipRow label="Median |error|" value={formatSeconds(r.q50_s)} />
                    <TooltipRow label="IQR" value={`${formatSeconds(r.q25_s)} – ${formatSeconds(r.q75_s)}`} />
                    <TooltipRow label="90th pct" value={formatSeconds(r.q90_s)} />
                  </TooltipBox>
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <TableView
        caption="Error quantiles"
        columns={[
          { key: "group", label: "Group" },
          { key: "q25", label: "Q25", align: "right" },
          { key: "q50", label: "Median", align: "right" },
          { key: "q75", label: "Q75", align: "right" },
          { key: "q90", label: "Q90", align: "right" },
          { key: "n", label: "Laps", align: "right" },
        ]}
        rows={rows.map((r) => ({
          group: r.group,
          q25: formatSeconds(r.q25_s),
          q50: formatSeconds(r.q50_s),
          q75: formatSeconds(r.q75_s),
          q90: formatSeconds(r.q90_s),
          n: r.n,
        }))}
      />
    </div>
  );
}
