import type { Interval } from "../../api/client";
import { formatLapTime } from "../../lib/format";

/** Tiny inline line of the next-N predicted lap times (no axes; values in the title). */
export function Sparkline({ points, width = 120, height = 32 }: { points: readonly Interval[]; width?: number; height?: number }) {
  if (points.length < 2) return null;
  const values = points.map((p) => p.pred_s);
  const low = Math.min(...values);
  const span = Math.max(Math.max(...values) - low, 0.05);
  const step = width / (points.length - 1);
  const path = values
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(height - 4 - ((v - low) / span) * (height - 8)).toFixed(1)}`)
    .join(" ");
  const label = points.map((p) => `lap ${p.lap}: ${formatLapTime(p.pred_s)}`).join(", ");
  return (
    <svg width={width} height={height} role="img" aria-label={`Next laps forecast: ${label}`}>
      <title>{label}</title>
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" />
    </svg>
  );
}
