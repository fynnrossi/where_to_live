#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rightmove_filter import FilterCriteria, filter_listings, load_listings_from_json


def parse_args():
    parser = argparse.ArgumentParser(description="Filter property listings from a JSON feed")
    parser.add_argument("--source", type=Path, default=Path(__file__).parents[1] / "data" / "sample_listings.json",
                        help="Path to a JSON file containing listings")
    parser.add_argument("--max-price", type=int, dest="max_price", help="Maximum monthly price (GBP)")
    parser.add_argument("--min-bedrooms", type=int, default=0, help="Minimum bedroom count")
    parser.add_argument("--max-travel-time", type=int, dest="max_travel_time", help="Max door-to-door minutes")
    parser.add_argument("--require", nargs="*", default=[], help="Required features e.g. garden parking")
    parser.add_argument("--exclude-postcode", nargs="*", default=[], dest="exclude_postcode",
                        help="Postcodes to exclude entirely")
    return parser.parse_args()


def main():
    args = parse_args()
    listings = load_listings_from_json(args.source)
    criteria = FilterCriteria(
        max_price=args.max_price,
        min_bedrooms=args.min_bedrooms,
        max_travel_time_minutes=args.max_travel_time,
        required_features=args.require,
        excluded_postcodes=args.exclude_postcode,
    )
    filtered = filter_listings(listings, criteria)
    for listing in filtered:
        print(listing.summary())


if __name__ == "__main__":
    main()
