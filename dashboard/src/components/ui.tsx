import type { ReactNode } from "react";

import { compoundColor, COMPOUND_LETTER } from "../lib/colors";

export function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-[var(--border)] bg-[var(--surface-1)] p-4 ${className}`}
    >
      {(title !== undefined || actions !== undefined) && (
        <header className="mb-3 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title !== undefined && (
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">{title}</h2>
            )}
            {subtitle !== undefined && (
              <p className="mt-0.5 text-xs text-[var(--text-muted)]">{subtitle}</p>
            )}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatusMessage({
  loading,
  error,
  empty,
}: {
  loading?: boolean | undefined;
  error?: Error | null | undefined;
  empty?: string | undefined;
}) {
  if (loading === true) {
    return <p className="py-6 text-center text-sm text-[var(--text-muted)]">Loading…</p>;
  }
  if (error) {
    return (
      <p role="alert" className="py-6 text-center text-sm text-[var(--status-critical)]">
        {error.message}
      </p>
    );
  }
  if (empty !== undefined) {
    return <p className="py-6 text-center text-sm text-[var(--text-muted)]">{empty}</p>;
  }
  return null;
}

export function CompoundBadge({ compound }: { compound: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm font-medium">
      <span
        aria-hidden
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border-[3px] text-[10px] font-bold"
        style={{ borderColor: compoundColor(compound), color: "var(--text-primary)" }}
      >
        {COMPOUND_LETTER[compound] ?? "?"}
      </span>
      {compound}
    </span>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-xs text-[var(--text-secondary)]">
      <span>{label}</span>
      {children}
    </label>
  );
}

export const inputClass =
  "h-9 min-w-0 rounded-md border border-[var(--border)] bg-[var(--surface-1)] px-2 text-sm text-[var(--text-primary)] focus:outline-2 focus:outline-[var(--accent)]";

export function DataTable({
  columns,
  rows,
  caption,
}: {
  columns: readonly { key: string; label: string; align?: "left" | "right" }[];
  rows: readonly Record<string, ReactNode>[];
  caption?: string;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="tabular w-full border-collapse text-xs">
        {caption !== undefined && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-[var(--border)] text-[var(--text-muted)]">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={`px-2 py-1.5 font-medium ${c.align === "right" ? "text-right" : "text-left"}`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-[var(--border)] last:border-0">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={`px-2 py-1.5 text-[var(--text-primary)] ${c.align === "right" ? "text-right" : "text-left"}`}
                >
                  {row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Collapsible table view behind every chart (identity never depends on colour alone). */
export function TableView(props: Parameters<typeof DataTable>[0]) {
  return (
    <details className="mt-2 text-xs text-[var(--text-muted)]">
      <summary className="cursor-pointer select-none">Table view</summary>
      <div className="mt-2 max-h-64 overflow-y-auto">
        <DataTable {...props} />
      </div>
    </details>
  );
}
