import { useEffect, useState, type ReactNode } from "react";
import { NavLink, useSearchParams } from "react-router";

import type { ProfileInfo } from "../../api/client";
import { inputClass } from "../ui";

type Theme = "light" | "dark" | "system";

function readTheme(): Theme {
  try {
    const stored = localStorage.getItem("boxbox-theme");
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    return "system";
  }
}

export function AppShell({ profiles, children }: { profiles: readonly ProfileInfo[]; children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(readTheme);
  const [params, setParams] = useSearchParams();
  const fallback = profiles.find((p) => p.is_default)?.profile ?? "subset";
  const profile = params.get("profile") ?? fallback;

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("boxbox-theme", theme);
    } catch {
      // Storage may be unavailable (private mode); the theme still applies for this session.
    }
  }, [theme]);

  const query = `?${new URLSearchParams({ profile }).toString()}`;
  const link = ({ isActive }: { isActive: boolean }) =>
    `rounded-md px-3 py-1.5 text-sm ${isActive ? "bg-[var(--surface-2)] font-semibold text-[var(--text-primary)]" : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"}`;

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b border-[var(--border)] bg-[var(--surface-1)]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-3 px-4 py-2">
          <span className="mr-2 text-base font-bold tracking-tight">BoxBox</span>
          <nav className="flex gap-1" aria-label="Pages">
            <NavLink to={{ pathname: "/", search: query }} className={link} end>
              Pit wall
            </NavLink>
            <NavLink to={{ pathname: "/degradation", search: query }} className={link}>
              Degradation
            </NavLink>
            <NavLink to={{ pathname: "/evaluation", search: query }} className={link}>
              Evaluation
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <label className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
              Profile
              <select
                className={inputClass}
                value={profile}
                onChange={(e) => {
                  const next = new URLSearchParams(params);
                  next.set("profile", e.target.value);
                  setParams(next, { replace: true });
                }}
              >
                {profiles.map((p) => (
                  <option key={p.profile} value={p.profile}>
                    {p.profile} (train {p.train_seasons[0]}–{p.train_seasons[p.train_seasons.length - 1]})
                  </option>
                ))}
              </select>
            </label>
            <select
              aria-label="Theme"
              className={inputClass}
              value={theme}
              onChange={(e) => {
                setTheme(e.target.value as Theme);
              }}
            >
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-4">{children}</main>
    </div>
  );
}
