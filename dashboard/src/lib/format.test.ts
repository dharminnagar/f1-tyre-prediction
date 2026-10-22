import { formatDelta, formatLapTime, formatPercent, modelLabel } from "./format";

describe("format", () => {
  it("formats lap times as m:ss.sss", () => {
    expect(formatLapTime(78.4321)).toBe("1:18.432");
    expect(formatLapTime(65.05)).toBe("1:05.050");
    expect(formatLapTime(null)).toBe("—");
    expect(formatLapTime(Number.NaN)).toBe("—");
  });

  it("formats signed deltas", () => {
    expect(formatDelta(0.4211)).toBe("+0.42 s");
    expect(formatDelta(-1.5, 1)).toBe("−1.5 s");
    expect(formatDelta(0)).toBe("±0.00 s");
  });

  it("formats percentages and model labels", () => {
    expect(formatPercent(0.9901)).toBe("99.0%");
    expect(modelLabel("hgb")).toBe("Gradient boosting");
    expect(modelLabel("unknown")).toBe("unknown");
  });
});
