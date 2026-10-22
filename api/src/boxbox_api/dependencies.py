"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, Query, Request

from boxbox.config import Profile
from boxbox.schema import ModelName
from boxbox_api.store import ProfileStore, Store


def get_store(request: Request) -> Store:
    """The store loaded at startup."""
    store: object = request.app.state.store
    if not isinstance(store, Store):
        raise RuntimeError("artifact store not initialised")
    return store


def get_profile_store(
    store: Annotated[Store, Depends(get_store)],
    profile: Annotated[Profile | None, Query(description="subset | full (default profile)")] = None,
) -> ProfileStore:
    """The requested profile's artifacts."""
    return store.get(profile)


StoreDep = Annotated[Store, Depends(get_store)]
ProfileDep = Annotated[ProfileStore, Depends(get_profile_store)]
ModelQuery = Annotated[ModelName | None, Query(description="defaults to the selected model")]
