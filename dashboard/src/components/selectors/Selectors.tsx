import type { DriverSummary, ModelInfo, ModelName, RaceSummary } from "../../api/client";
import { modelLabel } from "../../lib/format";
import type { Selection } from "../../lib/selection";
import { Field, inputClass } from "../ui";

type Update = (patch: Partial<Record<keyof Selection, string | number | undefined>>) => void;

export function Selectors({
  selection,
  update,
  races,
  drivers,
  models,
  defaultModel,
  maxLap,
}: {
  selection: Selection;
  update: Update;
  races: readonly RaceSummary[];
  drivers: readonly DriverSummary[];
  models: readonly ModelInfo[];
  /** Model shown when none is chosen (the page's API default). */
  defaultModel: ModelName | undefined;
  maxLap: number;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Field label="Race">
        <select
          className={inputClass}
          value={selection.raceId ?? ""}
          onChange={(e) => {
            update({ raceId: e.target.value, driverId: undefined, lap: undefined });
          }}
        >
          {races.map((r) => (
            <option key={r.race_id} value={r.race_id}>
              {r.season} {r.name} ({r.split})
            </option>
          ))}
        </select>
      </Field>
      <Field label="Driver">
        <select
          className={inputClass}
          value={selection.driverId ?? ""}
          onChange={(e) => {
            update({ driverId: e.target.value, lap: undefined });
          }}
        >
          {drivers.map((d) => (
            <option key={d.driver_id} value={d.driver_id}>
              {d.code} · {d.name} ({d.team})
            </option>
          ))}
        </select>
      </Field>
      <Field label={`Decision lap (2–${maxLap - 1})`}>
        <div className="flex items-center gap-2">
          <input
            type="range"
            aria-label="Decision lap slider"
            min={2}
            max={Math.max(2, maxLap - 1)}
            value={selection.lap ?? 2}
            onChange={(e) => {
              update({ lap: Number(e.target.value) });
            }}
            className="min-w-0 flex-1 accent-[var(--accent)]"
          />
          <input
            type="number"
            aria-label="Decision lap"
            min={2}
            max={maxLap - 1}
            value={selection.lap ?? ""}
            onChange={(e) => {
              update({ lap: e.target.value === "" ? undefined : Number(e.target.value) });
            }}
            className={`${inputClass} w-16`}
          />
        </div>
      </Field>
      <Field label="Model">
        <select
          className={inputClass}
          value={selection.model ?? defaultModel ?? ""}
          onChange={(e) => {
            update({ model: e.target.value });
          }}
        >
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {modelLabel(m.name)}
              {m.selected_for_strategy ? " (strategy pick)" : ""}
              {m.strategy_eligible ? "" : " (forecast only)"}
              {m.selected ? " (best next-lap)" : ""}
            </option>
          ))}
        </select>
      </Field>
    </div>
  );
}
