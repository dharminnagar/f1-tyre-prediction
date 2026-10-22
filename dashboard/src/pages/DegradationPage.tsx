import type { Profile } from "../api/client";
import { useDegradationCurves, useRaces } from "../api/queries";
import { DegradationCurveChart } from "../components/charts/DegradationCurveChart";
import { Card, Field, inputClass, StatusMessage } from "../components/ui";
import { useSelection } from "../lib/selection";

export function DegradationPage({ profile }: { profile: Profile }) {
  const [selection, update] = useSelection();
  const races = useRaces(profile);
  const raceId = selection.raceId ?? "all";
  const curves = useDegradationCurves(profile, raceId);
  return (
    <Card
      title="Degradation curves"
      subtitle="Median pace vs field (driver-adjusted, fuel-corrected) by tyre age; band = IQR, dashed = selected model with typical context. Ages with < 5 laps hidden."
      actions={
        <Field label="Races">
          <select
            className={inputClass}
            value={raceId}
            onChange={(e) => {
              update({ raceId: e.target.value === "all" ? undefined : e.target.value });
            }}
          >
            <option value="all">All training races</option>
            {(races.data ?? []).map((r) => (
              <option key={r.race_id} value={r.race_id}>
                {r.season} ({r.split})
              </option>
            ))}
          </select>
        </Field>
      }
    >
      {curves.data && curves.data.length > 0 ? (
        <DegradationCurveChart points={curves.data} />
      ) : (
        <StatusMessage loading={curves.isPending} error={curves.error} empty={curves.data ? "No curve data" : undefined} />
      )}
    </Card>
  );
}
