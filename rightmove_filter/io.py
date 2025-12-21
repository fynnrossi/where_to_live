from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import PropertyListing


def load_listings_from_json(path: Path) -> List[PropertyListing]:
    """Load listings from a JSON file containing a list of objects."""

    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("Listing payload must be a list of objects")
    return [PropertyListing.from_dict(item) for item in data]


def load_listings(payload: Iterable[dict]) -> List[PropertyListing]:
    return [PropertyListing.from_dict(item) for item in payload]


def write_listings_to_json(path: Path, listings: Iterable[PropertyListing]) -> None:
    serialised = [listing.to_dict() for listing in listings]
    path.write_text(json.dumps(serialised, indent=2))
