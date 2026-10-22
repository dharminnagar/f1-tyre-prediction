/** Lap time as m:ss.sss (e.g. 1:18.432). */
export function formatLapTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(3).padStart(6, "0")}`;
}

/** Signed seconds with a fixed number of decimals (e.g. +0.42 s). */
export function formatDelta(seconds: number | null | undefined, digits = 2): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const sign = seconds > 0 ? "+" : seconds < 0 ? "−" : "±";
  return `${sign}${Math.abs(seconds).toFixed(digits)} s`;
}

/** Plain seconds (e.g. 0.463 s). */
export function formatSeconds(seconds: number | null | undefined, digits = 3): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  return `${seconds.toFixed(digits)} s`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export const MODEL_LABELS: Record<string, string> = {
  baseline_curve: "Baseline curve",
  poly_ridge: "Polynomial ridge",
  hgb: "Gradient boosting",
  hgb_deg_ablation: "GBM + deg features",
};

export function modelLabel(name: string): string {
  return MODEL_LABELS[name] ?? name;
}
