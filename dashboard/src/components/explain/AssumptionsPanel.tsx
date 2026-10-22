import type { ProfileMetrics } from "../../api/client";
import { formatPercent, formatSeconds, modelLabel } from "../../lib/format";

/** Modelling assumptions and data provenance that qualify every number on the page. */
function windowRule(metrics: ProfileMetrics): string {
  const k = metrics.calibrated_margin_sigma;
  const margin = k === null ? "the configured σ multiple (not calibrated)" : `${k.toFixed(2)}σ (calibrated on validation stops)`;
  return `within ${margin} of the best pit lap and ahead of staying out by more than the same margin (cumulative σ)`;
}

export function AssumptionsPanel({ metrics }: { metrics: ProfileMetrics }) {
  const pit = metrics.pit_loss;
  return (
    <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
      {[
        ["Selected model", `${modelLabel(metrics.selected_model)} (${metrics.selection_rule})`],
        ["Training races", `${metrics.train_races} Spanish GPs; validation 2024, test 2025`],
        ["Pit loss", `${formatSeconds(pit.median_s, 1)} median of ${pit.n_stops} green-flag stops (IQR ${pit.q25_s.toFixed(1)}–${pit.q75_s.toFixed(1)} s)`],
        ["Stint verification", `${formatPercent(metrics.stint_verification.match_rate)} of ${metrics.stint_verification.n_stops} pit stops matched a tyre change`],
        ["Target", "lap time − fuel (0.05 s/lap remaining) − causal field median"],
        ["Competitive threshold", "pace loss vs a 3-lap-old set ≤ 1.0 s (sensitivity below)"],
        ["Window rule", windowRule(metrics)],
        ["Intervals", "80%: quantile gradient boosting for HGB, ±1.28σ of validation errors for the others"],
        ["Pit timing check", "descriptive only — real stops also react to traffic, safety cars and rivals"],
      ].map(([term, detail]) => (
        <div key={term}>
          <dt className="text-xs text-[var(--text-muted)]">{term}</dt>
          <dd className="text-[var(--text-primary)]">{detail}</dd>
        </div>
      ))}
    </dl>
  );
}
