import type { TyreState } from "../../api/client";
import { formatDelta, formatLapTime } from "../../lib/format";

const PARTS = [
  { key: "tyre_s", label: "Tyre age" },
  { key: "driver_offset_s", label: "Driver pace" },
  { key: "context_s", label: "Race context" },
] as const;

/** Explanation lines plus the additive decomposition of the next-lap prediction. */
export function WhyPanel({ state }: { state: TyreState }) {
  const d = state.decomposition;
  const scale = Math.max(0.5, ...PARTS.map((p) => Math.abs(d[p.key])));
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <ul className="space-y-1.5 text-sm text-[var(--text-secondary)]">
        {state.explanation.map((line) => (
          <li key={line} className="flex gap-2">
            <span aria-hidden className="text-[var(--text-muted)]">›</span>
            {line}
          </li>
        ))}
      </ul>
      <div>
        <p className="mb-2 text-xs text-[var(--text-muted)]">
          Next lap {formatLapTime(d.total_s)} = field reference {formatLapTime(d.field_ref_s)} + fuel {formatDelta(d.fuel_s)} + the parts below
        </p>
        <div className="space-y-2">
          {PARTS.map((part) => {
            const value = d[part.key];
            const width = `${((Math.abs(value) / scale) * 50).toFixed(1)}%`;
            return (
              <div key={part.key} className="grid grid-cols-[6.5rem_1fr_4.5rem] items-center gap-2 text-xs">
                <span className="text-[var(--text-secondary)]">{part.label}</span>
                <div className="relative h-3 rounded bg-[var(--surface-2)]">
                  <div aria-hidden className="absolute top-0 bottom-0 left-1/2 w-px bg-[var(--border)]" />
                  <div
                    aria-hidden
                    className="absolute top-0 bottom-0 rounded"
                    style={{
                      background: value >= 0 ? "var(--compound-soft)" : "var(--status-good)",
                      width,
                      left: value >= 0 ? "50%" : undefined,
                      right: value < 0 ? "50%" : undefined,
                    }}
                  />
                </div>
                <span className="tabular text-right text-[var(--text-primary)]">{formatDelta(value)}</span>
              </div>
            );
          })}
        </div>
        <p className="mt-2 text-xs text-[var(--text-muted)]">Positive = slower than the field reference.</p>
      </div>
    </div>
  );
}
