import type { ProfileMetrics } from "../../api/client";
import { formatPercent, formatSeconds, modelLabel } from "../../lib/format";

/** Modelling assumptions and data provenance that qualify every number on the page. */
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
        ["Window rule", "within 1σ of the best pit lap and ahead of staying out by > 1σ (cumulative)"],
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
