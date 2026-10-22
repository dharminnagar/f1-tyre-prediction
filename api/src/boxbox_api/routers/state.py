"""Tyre state at a decision lap."""

from typing import Annotated

from fastapi import APIRouter, Query

from boxbox_api.dependencies import ModelQuery, ProfileDep
from boxbox_api.schemas import TyreState
from boxbox_api.services import tyre_state

router = APIRouter(tags=["state"])


@router.get("/state")
def state(
    store: ProfileDep,
    race_id: str,
    driver_id: str,
    lap: Annotated[int, Query(ge=1, le=200)],
    model: ModelQuery = None,
) -> TyreState:
    """Degradation, next-lap/next-5 forecast, remaining competitive laps and why."""
    return tyre_state(store, race_id, driver_id, lap, model)
