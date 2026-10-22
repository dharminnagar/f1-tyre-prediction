import type { ModelName } from "../api/client";

/** Compound identity colours (CSS tokens validated for CVD in light and dark). */
export const COMPOUND_COLOR: Record<string, string> = {
  SOFT: "var(--compound-soft)",
  MEDIUM: "var(--compound-medium)",
  HARD: "var(--compound-hard)",
};

/** Single-letter compound code: secondary encoding so identity is never colour alone. */
export const COMPOUND_LETTER: Record<string, string> = { SOFT: "S", MEDIUM: "M", HARD: "H" };

export function compoundColor(compound: string): string {
  return COMPOUND_COLOR[compound] ?? "var(--text-muted)";
}

/** Fixed model order -> fixed categorical slot (colour follows the entity, never its rank). */
export const MODEL_COLOR: Record<ModelName, string> = {
  baseline_curve: "var(--series-3)",
  poly_ridge: "var(--series-1)",
  hgb: "var(--series-2)",
  lstm: "var(--series-4)",
};

export const MODEL_ORDER: readonly ModelName[] = ["baseline_curve", "poly_ridge", "hgb", "lstm"];
