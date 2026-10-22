"""Pit-window and counterfactual simulations."""

from fastapi import APIRouter

from boxbox_api.dependencies import StoreDep
from boxbox_api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    PitWindowRequest,
    PitWindowResponse,
)
from boxbox_api.services import counterfactual_response, pit_window

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/pit-window")
def pit_window_route(request: PitWindowRequest, store: StoreDep) -> PitWindowResponse:
    """Stay-out vs pit cost per candidate lap and the window beyond model uncertainty."""
    return pit_window(store.get(request.profile), request)


@router.post("/counterfactual")
def counterfactual_route(request: CounterfactualRequest, store: StoreDep) -> CounterfactualResponse:
    """Lap-by-lap stay-out vs pit-at-lap trajectories and the cumulative delta."""
    return counterfactual_response(store.get(request.profile), request)
