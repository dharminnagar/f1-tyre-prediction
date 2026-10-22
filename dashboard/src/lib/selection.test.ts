import { parseLap, parseModel, parseProfile } from "./selection";

describe("url selection parsing", () => {
  it("accepts only known profiles and models", () => {
    expect(parseProfile("full")).toBe("full");
    expect(parseProfile("other")).toBeUndefined();
    expect(parseModel("hgb")).toBe("hgb");
    expect(parseModel("lstm")).toBe("lstm");
    expect(parseModel("gru")).toBeUndefined();
  });

  it("parses positive integer laps", () => {
    expect(parseLap("12")).toBe(12);
    expect(parseLap("0")).toBeUndefined();
    expect(parseLap("abc")).toBeUndefined();
    expect(parseLap(null)).toBeUndefined();
  });
});
