import { keepPreviousData, useQuery } from "@tanstack/react-query";

import {
  api,
  unwrap,
  type CounterfactualRequest,
  type GroupBy,
  type ModelName,
  type PitWindowRequest,
  type Profile,
} from "./client";

const STATIC = { staleTime: Infinity } as const;

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: async () => unwrap(await api.GET("/api/health")),
    ...STATIC,
  });
}

export function useProfiles() {
  return useQuery({
    queryKey: ["profiles"],
    queryFn: async () => unwrap(await api.GET("/api/profiles")),
    ...STATIC,
  });
}

export function useRaces(profile: Profile | undefined) {
  return useQuery({
    queryKey: ["races", profile],
    queryFn: async () =>
      unwrap(await api.GET("/api/races", { params: { query: { profile: profile ?? null } } })),
    enabled: profile !== undefined,
    ...STATIC,
  });
}

export function useDrivers(profile: Profile | undefined, raceId: string | undefined) {
  return useQuery({
    queryKey: ["drivers", profile, raceId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/races/{race_id}/drivers", {
          params: { path: { race_id: raceId ?? "" }, query: { profile: profile ?? null } },
        }),
      ),
    enabled: profile !== undefined && raceId !== undefined,
    ...STATIC,
  });
}

export function useLaps(
  profile: Profile | undefined,
  raceId: string | undefined,
  driverId: string | undefined,
) {
  return useQuery({
    queryKey: ["laps", profile, raceId, driverId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/races/{race_id}/drivers/{driver_id}/laps", {
          params: {
            path: { race_id: raceId ?? "", driver_id: driverId ?? "" },
            query: { profile: profile ?? null },
          },
        }),
      ),
    enabled: profile !== undefined && raceId !== undefined && driverId !== undefined,
    ...STATIC,
  });
}

export function useModels(profile: Profile | undefined) {
  return useQuery({
    queryKey: ["models", profile],
    queryFn: async () =>
      unwrap(await api.GET("/api/models", { params: { query: { profile: profile ?? null } } })),
    enabled: profile !== undefined,
    ...STATIC,
  });
}

export interface DecisionKey {
  profile: Profile;
  raceId: string;
  driverId: string;
  lap: number;
  model: ModelName | undefined;
}

export function useTyreState(key: DecisionKey | undefined) {
  return useQuery({
    queryKey: ["state", key],
    queryFn: async () => {
      if (key === undefined) throw new Error("no decision selected");
      return unwrap(
        await api.GET("/api/state", {
          params: {
            query: {
              race_id: key.raceId,
              driver_id: key.driverId,
              lap: key.lap,
              model: key.model ?? null,
              profile: key.profile,
            },
          },
        }),
      );
    },
    enabled: key !== undefined,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
}

export function usePitWindow(request: PitWindowRequest | undefined) {
  return useQuery({
    queryKey: ["pit-window", request],
    queryFn: async () => {
      if (request === undefined) throw new Error("no decision selected");
      return unwrap(await api.POST("/api/strategy/pit-window", { body: request }));
    },
    enabled: request !== undefined,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
}

export function useCounterfactual(request: CounterfactualRequest | undefined) {
  return useQuery({
    queryKey: ["counterfactual", request],
    queryFn: async () => {
      if (request === undefined) throw new Error("no decision selected");
      return unwrap(await api.POST("/api/strategy/counterfactual", { body: request }));
    },
    enabled: request !== undefined,
    placeholderData: keepPreviousData,
    ...STATIC,
  });
}

export function useMetrics() {
  return useQuery({
    queryKey: ["metrics"],
    queryFn: async () => unwrap(await api.GET("/api/evaluation/metrics")),
    ...STATIC,
  });
}

export function usePredictions(
  profile: Profile | undefined,
  raceId: string | undefined,
  driverId: string | undefined,
  stintId: number | undefined,
  model: ModelName | undefined,
) {
  return useQuery({
    queryKey: ["predictions", profile, raceId, driverId, stintId, model],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/evaluation/predictions", {
          params: {
            query: {
              race_id: raceId ?? "",
              driver_id: driverId ?? "",
              stint_id: stintId ?? 0,
              model: model ?? null,
              profile: profile ?? null,
            },
          },
        }),
      ),
    enabled:
      profile !== undefined &&
      raceId !== undefined &&
      driverId !== undefined &&
      stintId !== undefined,
    ...STATIC,
  });
}

export function useErrorGroups(
  profile: Profile | undefined,
  groupBy: GroupBy,
  model: ModelName | undefined,
) {
  return useQuery({
    queryKey: ["errors", profile, groupBy, model],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/evaluation/errors", {
          params: { query: { group_by: groupBy, model: model ?? null, profile: profile ?? null } },
        }),
      ),
    enabled: profile !== undefined,
    ...STATIC,
  });
}

export function useThresholdSensitivity(profile: Profile | undefined) {
  return useQuery({
    queryKey: ["threshold", profile],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/evaluation/threshold-sensitivity", {
          params: { query: { profile: profile ?? null } },
        }),
      ),
    enabled: profile !== undefined,
    ...STATIC,
  });
}

export function usePitTiming(profile: Profile | undefined) {
  return useQuery({
    queryKey: ["pit-timing", profile],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/evaluation/pit-timing", {
          params: { query: { profile: profile ?? null } },
        }),
      ),
    enabled: profile !== undefined,
    ...STATIC,
  });
}

export function useDegradationCurves(profile: Profile | undefined, raceId: string) {
  return useQuery({
    queryKey: ["curves", profile, raceId],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/degradation/curves", {
          params: { query: { race_id: raceId, profile: profile ?? null } },
        }),
      ),
    enabled: profile !== undefined,
    ...STATIC,
  });
}
