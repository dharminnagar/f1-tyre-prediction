/** Shared axis/grid styling for Recharts charts (token-driven, recessive). */
export const AXIS_PROPS = {
  stroke: "var(--text-muted)",
  tick: { fill: "var(--text-muted)", fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: "var(--border)" },
} as const;

export const GRID_PROPS = { stroke: "var(--grid)", strokeDasharray: "0", vertical: false } as const;
