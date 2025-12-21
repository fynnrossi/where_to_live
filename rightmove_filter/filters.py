from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from .models import PropertyListing


@dataclass
class FilterCriteria:
    max_price: Optional[int] = None
    min_bedrooms: int = 0
    max_travel_time_minutes: Optional[int] = None
    required_features: Sequence[str] = ()
    excluded_postcodes: Sequence[str] = ()


def filter_listings(listings: Iterable[PropertyListing], criteria: FilterCriteria) -> List[PropertyListing]:
    results: List[PropertyListing] = []
    for listing in listings:
        if criteria.max_price is not None and listing.price > criteria.max_price:
            continue
        if listing.bedrooms < criteria.min_bedrooms:
            continue
        if criteria.max_travel_time_minutes is not None:
            if listing.travel_time_minutes is None or listing.travel_time_minutes > criteria.max_travel_time_minutes:
                continue
        if criteria.required_features and not listing.matches_features(criteria.required_features):
            continue
        if criteria.excluded_postcodes:
            normalised_postcodes = {pc.strip().upper() for pc in criteria.excluded_postcodes if pc}
            if listing.postcode.strip().upper() in normalised_postcodes:
                continue
        results.append(listing)
    return sorted(results, key=lambda item: (item.price, -item.bedrooms, item.listing_id))
