import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

type Schemas = components["schemas"];

export type Health = Schemas["Health"];
export type ProfileInfo = Schemas["ProfileInfo"];
export type RaceSummary = Schemas["RaceSummary"];
export type DriverSummary = Schemas["DriverSummary"];
export type LapRecord = Schemas["LapRecord"];
export type ModelInfo = Schemas["ModelInfo"];
export type TyreState = Schemas["TyreState"];
export type Interval = Schemas["Interval"];
export type PitWindowRequest = Schemas["PitWindowRequest"];
export type PitWindowResponse = Schemas["PitWindowResponse"];
export type PitCandidate = Schemas["PitCandidateOut"];
export type CounterfactualRequest = Schemas["CounterfactualRequest"];
export type CounterfactualResponse = Schemas["CounterfactualResponse"];
export type MetricsResponse = Schemas["MetricsResponse"];
export type ProfileMetrics = Schemas["ProfileMetrics"];
export type MetricRow = Schemas["MetricRow"];
export type HorizonRow = Schemas["HorizonRow"];
export type PredictionRow = Schemas["PredictionRow"];
export type ErrorGroupRow = Schemas["ErrorGroupRow"];
export type ThresholdRow = Schemas["ThresholdRow"];
export type PitTimingRow = Schemas["PitTimingRow"];
export type CurvePoint = Schemas["CurvePoint"];
export type Compound = Schemas["TyreState"]["compound"];
export type ModelName = Schemas["ModelInfo"]["name"];
export type Profile = Schemas["ProfileInfo"]["profile"];
export type GroupBy = "compound" | "race" | "stint_length";

export const api = createClient<paths>({ baseUrl: window.location.origin });

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function detailOf(error: unknown): string {
  if (typeof error === "object" && error !== null && "detail" in error) {
    const detail = error.detail;
    return typeof detail === "string" ? detail : JSON.stringify(detail);
  }
  return "request failed";
}

/** Unwrap an openapi-fetch result, throwing ApiError on non-2xx responses. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.data === undefined) {
    throw new ApiError(result.response.status, detailOf(result.error));
  }
  return result.data;
}
