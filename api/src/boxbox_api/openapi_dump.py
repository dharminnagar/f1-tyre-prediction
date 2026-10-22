"""Write the OpenAPI document without loading artifacts (``python -m boxbox_api.openapi_dump``)."""

import json
import sys
from pathlib import Path

from boxbox_api.main import create_app


def main() -> None:
    """Dump ``create_app().openapi()`` to the path given as the first argument."""
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("api/openapi.json")
    target.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")


if __name__ == "__main__":
    main()
