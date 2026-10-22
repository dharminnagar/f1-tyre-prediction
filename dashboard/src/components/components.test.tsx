import { render, screen, within } from "@testing-library/react";

import { pitWindow, profileMetrics, state } from "../test/fixtures";
import { MetricsTable } from "../pages/EvaluationPage";
import { WhyPanel } from "./explain/WhyPanel";
import { KpiTiles } from "./kpi/KpiTiles";

describe("KpiTiles", () => {
  it("shows compound, forecast, remaining life and the pit window", () => {
    render(<KpiTiles state={state} window={pitWindow} />);
    expect(screen.getByText("SOFT")).toBeInTheDocument();
    expect(screen.getByText("1:20.720")).toBeInTheDocument();
    expect(screen.getByText("~7")).toBeInTheDocument();
    expect(screen.getByText("Lap 11–17")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Next laps forecast/ })).toBeInTheDocument();
  });

  it("says when no window beats the margin", () => {
    render(<KpiTiles state={state} window={{ ...pitWindow, window: null }} />);
    expect(screen.getByText("none")).toBeInTheDocument();
  });
});

describe("WhyPanel", () => {
  it("lists explanations and decomposition parts", () => {
    render(<WhyPanel state={state} />);
    expect(screen.getByText("About 7 competitive laps left.")).toBeInTheDocument();
    expect(screen.getByText("Tyre age")).toBeInTheDocument();
    expect(screen.getByText("+0.40 s")).toBeInTheDocument();
  });
});

describe("MetricsTable", () => {
  it("puts both profiles side by side", () => {
    render(<MetricsTable profiles={[profileMetrics("subset", 0), profileMetrics("full", 0.01)]} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("subset val")).toBeInTheDocument();
    expect(within(table).getByText("full test")).toBeInTheDocument();
    expect(within(table).getByText("0.463 s ★")).toBeInTheDocument();
    expect(within(table).getByText("0.511 s")).toBeInTheDocument();
  });
});
