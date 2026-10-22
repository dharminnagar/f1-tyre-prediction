import type { ReactNode } from "react";

export interface TooltipEntry {
  name?: string | number;
  value?: unknown;
  color?: string;
  dataKey?: unknown;
}

export function TooltipBox({ title, children }: { title: ReactNode; children: ReactNode }) {
  return (
    <div className="tabular rounded-md border border-[var(--border)] bg-[var(--surface-1)] px-2.5 py-2 text-xs shadow-sm">
      <div className="mb-1 font-semibold text-[var(--text-primary)]">{title}</div>
      <div className="space-y-0.5 text-[var(--text-secondary)]">{children}</div>
    </div>
  );
}

export function TooltipRow({ color, label, value }: { color?: string; label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="inline-flex items-center gap-1.5">
        {color !== undefined && (
          <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
        )}
        {label}
      </span>
      <span className="text-[var(--text-primary)]">{value}</span>
    </div>
  );
}

export function LegendItem({ color, label, dashed = false }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
      <svg width="16" height="8" aria-hidden>
        <line
          x1="0"
          y1="4"
          x2="16"
          y2="4"
          stroke={color}
          strokeWidth="2"
          strokeDasharray={dashed ? "4 3" : undefined}
        />
      </svg>
      {label}
    </span>
  );
}

export function Legend({ children }: { children: ReactNode }) {
  return <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">{children}</div>;
}
