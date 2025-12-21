from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Set


@dataclass
class PropertyListing:
    """Represents a single listing from RightMove or another property feed."""

    listing_id: str
    price: int
    bedrooms: int
    bathrooms: int
    property_type: str
    postcode: str
    listing_url: str
    description: str = ""
    travel_time_minutes: Optional[int] = None
    features: Set[str] = field(default_factory=set)

    @classmethod
    def from_dict(cls, raw: dict) -> "PropertyListing":
        """Create a :class:`PropertyListing` from a dictionary payload."""

        return cls(
            listing_id=str(raw.get("listing_id") or raw.get("id") or ""),
            price=int(raw.get("price", 0)),
            bedrooms=int(raw.get("bedrooms", 0)),
            bathrooms=int(raw.get("bathrooms", 0)),
            property_type=str(raw.get("property_type", "")).strip() or "unknown",
            postcode=str(raw.get("postcode", "")).strip(),
            listing_url=str(raw.get("listing_url", "")).strip(),
            description=str(raw.get("description", "")).strip(),
            travel_time_minutes=cls._parse_optional_int(raw.get("travel_time_minutes")),
            features=cls._normalise_features(raw.get("features", [])),
        )

    @staticmethod
    def _parse_optional_int(value) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalise_features(value: Iterable[str]) -> Set[str]:
        if value is None:
            return set()
        return {str(item).strip().lower() for item in value if str(item).strip()}

    def matches_features(self, required: Iterable[str]) -> bool:
        required_set = {feature.strip().lower() for feature in required if feature}
        return required_set.issubset(self.features)

    def to_dict(self) -> dict:
        return {
            "listing_id": self.listing_id,
            "price": self.price,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "property_type": self.property_type,
            "postcode": self.postcode,
            "listing_url": self.listing_url,
            "description": self.description,
            "travel_time_minutes": self.travel_time_minutes,
            "features": sorted(self.features),
        }

    def summary(self) -> str:
        travel = f"; travel: {self.travel_time_minutes} mins" if self.travel_time_minutes is not None else ""
        return (
            f"£{self.price:,} | {self.bedrooms} bed / {self.bathrooms} bath {self.property_type}"
            f" in {self.postcode}{travel}"
        )
