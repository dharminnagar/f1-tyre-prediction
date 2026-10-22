import { laps, pitWindow, state } from "../test/fixtures";
import { costCurve, defaultLap, paceDomain, pacePoints, pivotCurves } from "./transform";

describe("transforms", () => {
  it("splits clean and excluded laps and attaches the forecast band", () => {
    const points = pacePoints(laps, state);
    const lap1 = points.find((p) => p.lap === 1);
    const lap9 = points.find((p) => p.lap === 9);
    expect(lap1?.clean).toBeNull();
    expect(lap1?.excluded).toBeCloseTo(80.08);
    expect(lap1?.reason).toBe("lap_one");
    expect(lap9?.forecast).toBeCloseTo(80.72);
    expect(lap9?.band).toEqual([80.1, 81.3]);
  });

  it("pads the pace domain around clean laps and forecasts", () => {
    const [low, high] = paceDomain(pacePoints(laps, state));
    expect(low).toBeLessThan(80.16);
    expect(high).toBeGreaterThan(81.6);
  });

  it("expresses pit cost relative to the best candidate", () => {
    const curve = costCurve(pitWindow.candidates);
    const best = curve.find((p) => p.pitLap === 14);
    expect(best?.costVsBest).toBe(0);
    expect(curve.find((p) => p.pitLap === 17)?.costVsBest).toBeCloseTo(0.9);
    expect(curve.filter((p) => p.inWindow).map((p) => p.pitLap)).toEqual([11, 12, 13, 14, 15, 16, 17]);
  });

  it("pivots curve points per compound and hides thin ages", () => {
    const rows = pivotCurves([
      { compound: "SOFT", tyre_age: 2, median_s: 0.1, q25_s: 0, q75_s: 0.2, n: 10, fitted_s: 0.12 },
      { compound: "HARD", tyre_age: 2, median_s: -0.2, q25_s: -0.3, q75_s: 0, n: 10, fitted_s: -0.1 },
      { compound: "SOFT", tyre_age: 40, median_s: 2, q25_s: 1, q75_s: 3, n: 2, fitted_s: 1.5 },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.SOFT_median).toBe(0.1);
    expect(rows[0]?.HARD_iqr).toEqual([-0.3, 0]);
  });

  it("defaults the decision lap into the first stint", () => {
    expect(defaultLap(laps)).toBe(8);
  });
});
