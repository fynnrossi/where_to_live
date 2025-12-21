from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rightmove_filter import FilterCriteria, filter_listings, load_listings_from_json
from rightmove_filter.models import PropertyListing


def load_sample_listings():
    return load_listings_from_json(Path(__file__).parents[1] / "data" / "sample_listings.json")


def test_filters_respect_price_and_bedrooms():
    listings = load_sample_listings()
    criteria = FilterCriteria(max_price=2200, min_bedrooms=2)
    results = filter_listings(listings, criteria)

    assert all(listing.price <= 2200 for listing in results)
    assert all(listing.bedrooms >= 2 for listing in results)
    assert [listing.listing_id for listing in results] == ["D4", "A1"]


def test_travel_time_and_features():
    listings = load_sample_listings()
    criteria = FilterCriteria(max_travel_time_minutes=40, required_features=["parking"])
    results = filter_listings(listings, criteria)

    assert [listing.listing_id for listing in results] == ["A1", "C3"]
    assert all(listing.travel_time_minutes is not None and listing.travel_time_minutes <= 40 for listing in results)


def test_excluded_postcodes_are_removed():
    listings = load_sample_listings()
    criteria = FilterCriteria(excluded_postcodes=["N1 0AA", "SE15 4AB"])
    results = filter_listings(listings, criteria)

    postcodes = {listing.postcode for listing in results}
    assert postcodes.isdisjoint({"N1 0AA", "SE15 4AB"})


def test_property_listing_normalises_features():
    listing = PropertyListing.from_dict({
        "listing_id": "X",
        "price": 1000,
        "bedrooms": 1,
        "bathrooms": 1,
        "property_type": "flat",
        "postcode": "AB1 2CD",
        "listing_url": "https://example.test",
        "features": ["Garden", " garden ", "PARKING"],
    })

    assert listing.matches_features(["garden"])
    assert listing.matches_features(["parking"])
    assert listing.matches_features(["garden", "parking"])


def test_filter_sort_order():
    listings = [
        PropertyListing("1", 2000, 1, 1, "flat", "N1", "", travel_time_minutes=20),
        PropertyListing("2", 1800, 2, 1, "flat", "N1", "", travel_time_minutes=30),
        PropertyListing("3", 1800, 1, 1, "flat", "N1", "", travel_time_minutes=25),
    ]
    criteria = FilterCriteria()
    results = filter_listings(listings, criteria)

    assert [listing.listing_id for listing in results] == ["2", "3", "1"]
