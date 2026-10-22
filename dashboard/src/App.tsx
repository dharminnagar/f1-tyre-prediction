import { Route, Routes } from "react-router";

import type { Profile } from "./api/client";
import { useProfiles } from "./api/queries";
import { AppShell } from "./components/layout/AppShell";
import { Card, StatusMessage } from "./components/ui";
import { parseProfile, useSelection } from "./lib/selection";
import { DegradationPage } from "./pages/DegradationPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { PitWallPage } from "./pages/PitWallPage";

export function App() {
  const profiles = useProfiles();
  const [selection] = useSelection();
  if (profiles.data === undefined) {
    return (
      <div className="mx-auto max-w-xl p-8">
        <Card title="BoxBox">
          <StatusMessage loading={profiles.isPending} error={profiles.error} />
        </Card>
      </div>
    );
  }
  const fallback = profiles.data.find((p) => p.is_default)?.profile;
  const requested = selection.profile ?? parseProfile(fallback ?? null);
  const profile: Profile = profiles.data.some((p) => p.profile === requested)
    ? (requested ?? "subset")
    : (profiles.data[0]?.profile ?? "subset");
  return (
    <AppShell profiles={profiles.data}>
      <Routes>
        <Route path="/" element={<PitWallPage key={profile} profile={profile} />} />
        <Route path="/degradation" element={<DegradationPage profile={profile} />} />
        <Route path="/evaluation" element={<EvaluationPage profile={profile} />} />
      </Routes>
    </AppShell>
  );
}
