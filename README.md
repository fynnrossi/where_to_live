# RightMove Filter

A lightweight Python toolkit for filtering and ranking property listings from Rightmove (or similar exports).
It ships with a small sample dataset and a CLI so you can try the filters immediately.

## Features
- Parse listing payloads into a structured dataclass
- Filter by price, bedrooms, travel time, must-have features, and excluded postcodes
- Sort results by affordability while keeping better sized homes ahead of smaller ones at the same price
- Command-line helper to experiment with filter combinations

## Getting started
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the sample filter against the bundled data:
   ```bash
   python scripts/filter_listings.py --min-bedrooms 2 --max-price 2200 --max-travel-time 45 --require parking
   ```

## Package layout
- `rightmove_filter/` – core package
  - `models.py` – dataclass representing a listing with helpers for serialisation
  - `filters.py` – filtering logic and criteria definition
  - `io.py` – loading and writing listings to JSON
- `data/sample_listings.json` – small fixture of listings for experimentation
- `scripts/filter_listings.py` – CLI entrypoint demonstrating the filters
- `tests/` – unit tests covering the filter behaviour and legacy travel time functions

## Running tests
```bash
pytest
```

## License
MIT
