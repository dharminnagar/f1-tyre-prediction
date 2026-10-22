import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router";

import type { ModelName, Profile } from "../api/client";

export interface Selection {
  profile: Profile | undefined;
  raceId: string | undefined;
  driverId: string | undefined;
  lap: number | undefined;
  model: ModelName | undefined;
}

type SelectionKey = keyof Selection;

const MODELS: readonly string[] = ["baseline_curve", "poly_ridge", "hgb"];

export function parseProfile(value: string | null): Profile | undefined {
  return value === "subset" || value === "full" ? value : undefined;
}

export function parseModel(value: string | null): ModelName | undefined {
  return value !== null && MODELS.includes(value) ? (value as ModelName) : undefined;
}

export function parseLap(value: string | null): number | undefined {
  if (value === null) return undefined;
  const lap = Number.parseInt(value, 10);
  return Number.isInteger(lap) && lap > 0 ? lap : undefined;
}

/** Race/driver/lap/model/profile selection persisted in the URL query string. */
export function useSelection(): [Selection, (patch: Partial<Record<SelectionKey, string | number | undefined>>) => void] {
  const [params, setParams] = useSearchParams();
  const selection = useMemo<Selection>(
    () => ({
      profile: parseProfile(params.get("profile")),
      raceId: params.get("race") ?? undefined,
      driverId: params.get("driver") ?? undefined,
      lap: parseLap(params.get("lap")),
      model: parseModel(params.get("model")),
    }),
    [params],
  );
  const update = useCallback(
    (patch: Partial<Record<SelectionKey, string | number | undefined>>) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          const names: Record<SelectionKey, string> = {
            profile: "profile",
            raceId: "race",
            driverId: "driver",
            lap: "lap",
            model: "model",
          };
          for (const [key, value] of Object.entries(patch) as [SelectionKey, string | number | undefined][]) {
            if (value === undefined || value === "") next.delete(names[key]);
            else next.set(names[key], String(value));
          }
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );
  return [selection, update];
}
