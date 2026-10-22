import { render, screen, within } from "@testing-library/react";

import { pitWindow, profileMetrics, state, thresholdRows } from "../test/fixtures";
import {
  IntervalCoverageTable,
  MetricsTable,
  PitTimingSummaryTable,
  SequenceCellTable,
  ThresholdTable,
} from "../pages/EvaluationPage";
import { AssumptionsPanel } from "./explain/AssumptionsPanel";
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

describe("ThresholdTable", () => {
  it("shows MAE with bias for the chosen decision mode only", () => {
    render(<ThresholdTable rows={thresholdRows} decision="stint_lap_5" selectedModel="poly_ridge" />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("3.2 (+0.5)")).toBeInTheDocument();
    expect(within(table).queryByText("4.4 (+0.5)")).toBeNull();
    expect(within(table).getAllByText("20 / 60")).toHaveLength(2);
    expect(within(table).getAllByText("80%")).toHaveLength(2);
  });
});

describe("PitTimingSummaryTable", () => {
  it("summarises the calibrated windows per split", () => {
    render(<PitTimingSummaryTable metrics={profileMetrics("subset", 0)} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("40/42")).toBeInTheDocument();
    expect(within(table).getByText("62%")).toBeInTheDocument();
    expect(within(table).getByText("9 laps")).toBeInTheDocument();
    expect(within(table).getByText("+4.0 laps")).toBeInTheDocument();
    expect(within(table).getByText("neutralised 2")).toBeInTheDocument();
  });
});

describe("AssumptionsPanel", () => {
  it("states the calibrated window margin instead of a fixed 1σ", () => {
    render(<AssumptionsPanel metrics={profileMetrics("subset", 0)} />);
    expect(screen.getByText(/within 0\.50σ \(calibrated on validation stops\)/)).toBeInTheDocument();
  });

  it("says when the margin was not calibrated", () => {
    render(<AssumptionsPanel metrics={{ ...profileMetrics("subset", 0), calibrated_margin_sigma: null }} />);
    expect(screen.getByText(/not calibrated/)).toBeInTheDocument();
  });
});

describe("IntervalCoverageTable", () => {
  it("shows coverage and width per split with the interval method", () => {
    render(<IntervalCoverageTable metrics={profileMetrics("subset", 0)} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("quantile HGB")).toBeInTheDocument();
    expect(within(table).getByText("80% (1.46 s)")).toBeInTheDocument();
    expect(within(table).getByText("70% (1.28 s)")).toBeInTheDocument();
  });
});

describe("SequenceCellTable", () => {
  it("compares the LSTM and GRU cells", () => {
    render(<SequenceCellTable metrics={profileMetrics("subset", 0)} />);
    const table = screen.getByRole("table");
    expect(within(table).getByText("LSTM sequence")).toBeInTheDocument();
    expect(within(table).getByText("GRU sequence")).toBeInTheDocument();
    expect(within(table).getAllByText("0.680 s (k=20)")).toHaveLength(2);
  });
});
