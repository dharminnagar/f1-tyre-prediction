import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";

import {
  counterfactual,
  drivers,
  laps,
  models,
  pitWindow,
  profileMetrics,
  profiles,
  races,
  state,
} from "./test/fixtures";

const ROUTES: [RegExp, unknown][] = [
  [/\/api\/profiles/, profiles],
  [/\/api\/races\/[^/]+\/drivers\/[^/]+\/laps/, laps],
  [/\/api\/races\/[^/]+\/drivers/, drivers],
  [/\/api\/races/, races],
  [/\/api\/models/, models],
  [/\/api\/state/, state],
  [/\/api\/strategy\/pit-window/, pitWindow],
  [/\/api\/strategy\/counterfactual/, counterfactual],
  [/\/api\/evaluation\/metrics/, { profiles: [profileMetrics("subset", 0)] }],
];

function mockFetch(input: RequestInfo | URL): Promise<Response> {
  const url = input instanceof Request ? input.url : String(input);
  const match = ROUTES.find(([pattern]) => pattern.test(url));
  const body = match ? JSON.stringify(match[1]) : JSON.stringify({ detail: "not found" });
  return Promise.resolve(
    new Response(body, { status: match ? 200 : 404, headers: { "Content-Type": "application/json" } }),
  );
}

async function renderApp(path: string) {
  vi.resetModules();
  vi.stubGlobal("fetch", vi.fn(mockFetch));
  const { App } = await import("./App");
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the pit wall with state and window from the API", async () => {
    await renderApp("/?profile=subset");
    expect(await screen.findByText("Lap 11–17")).toBeInTheDocument();
    expect(await screen.findByText("Pit cost curve")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Evaluation" })).toBeInTheDocument();
  });

  it("renders the evaluation page metrics table", async () => {
    await renderApp("/evaluation?profile=subset");
    expect(await screen.findByText("Held-out accuracy (MAE, seconds)")).toBeInTheDocument();
    expect(await screen.findByText("0.463 s ★")).toBeInTheDocument();
  });
});
