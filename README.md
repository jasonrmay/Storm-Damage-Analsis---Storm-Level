# Storm Damage Analysis — Storm Level

A multi-language (Python + R) analytics project that builds a unified storm-event dataset from public federal sources, then applies machine-learning models to **predict property damage** and **classify storm types** across the continental United States.

## Project Overview

The project spans three phases, each in its own directory:

1. **`data_wrangling/`** — A Python pipeline that fetches, cleans, and merges seven datasets into a single analysis-ready table keyed on FIPS county codes and temporal dimensions (month, year).
2. **`storm_dmg_regress/`** — Python modeling scripts that predict storm-level property damage using XGBoost regression, XGBoost classification (binary: damage vs. no damage), and permutation-importance analysis.
3. **`storm_type_classifier/`** — An R Markdown workflow (tidymodels + XGBoost) that classifies storm events into six categories (Flash Flood, Flood, Hail, Heavy Rain, Thunderstorm Wind, Tornado) and produces geospatial risk-profile maps comparing historical vs. predicted dominant storm types on a 0.5° grid.

## Data Sources

| Source | What it provides |
|---|---|
| **NOAA Storm Events Database** (NCEI) | Storm-level details: event type, damage, coordinates, timing |
| **U.S. Census Bureau ACS 5-Year API** | County-level population, median household income, housing age |
| **NCEI Climate at a Glance** | Monthly county temperature and temperature anomaly |
| **NOAA Oceanic Niño Index (ONI / RONI)** | Monthly regional climate oscillation averages |
| **NOAA GeoPlatform — Coastal Counties** | Shoreline and watershed coastal classification |

A detailed variable dictionary is in [`VarDescriptions.pdf`](VarDescriptions.pdf).

## Repository Structure

```
├── data_wrangling/
│   ├── dataGetter.py          # Fetches raw data from APIs and NOAA FTP
│   ├── dataProcessor.py       # Cleans each dataset (FIPS construction, damage parsing, etc.)
│   ├── merger.py              # Left-joins all tables on FIPS + month
│   ├── data_demo.ipynb        # End-to-end demo (loops 2009–2024)
│   └── static_data/           # Bundled CSVs (coastal counties, RONI)
│
├── storm_dmg_regress/
│   ├── data_prep.py           # Feature engineering: log-transform, cyclical month, one-hot encoding
│   ├── xgboost_regress.py     # XGBoost regression (log₁₀ damage) with RandomizedSearchCV
│   ├── xgboost_classifier.py  # XGBoost binary classifier (damage vs. no damage)
│   ├── perm_importance.py     # Permutation importance with collapsed categorical groups
│   └── demo.ipynb             # Modeling walkthrough notebook
│
├── storm_type_classifier/
│   └── FinalStormClassifier.Rmd   # tidymodels XGBoost multiclass classifier with SMOTE,
│                                  # grid tuning, and geospatial risk-profile maps
│
├── VarDescriptions.pdf        # Full variable descriptions
└── README.md
```

## Engineered Features

The pipeline and modeling scripts compute several derived variables beyond the raw data:

- **`STORM_AREA_SQMILES`** — Approximate rectangular area of the storm footprint (spherical geometry from begin/end coordinates).
- **`DURATION_MINUTES`** — Storm duration from military-time begin/end fields, adjusted for midnight crossover.
- **`DAMAGE_PROPERTY`** — Parsed from NOAA's `K`/`M`/`B` suffix notation into numeric dollars.
- **`log10_damage`** — Log₁₀-transformed damage for regression targets.
- **`month_sin` / `month_cos`** — Cyclical month encoding so the model captures seasonal proximity (e.g., December and January are close).
- **`DAMAGE_DENSITY`** / **`DURATION_DENSITY`** — Damage and duration per unit storm area.
- **`LAT_X_TEMP`** / **`LAT_X_ANOMALY`** — Latitude × temperature interaction terms capturing regional climate patterns.
- **`DAMAGE_ZERO`** / **`AREA_ZERO`** / **`DURATION_ZERO`** — Binary indicators for zero-valued measurements.

## Getting Started

### Prerequisites

**Python** (data wrangling + damage modeling):

```
pip install pandas numpy requests us scikit-learn xgboost scipy
```

**R** (storm type classifier):

```r
install.packages(c("tidymodels", "xgboost", "themis", "doParallel",
                    "caret", "vip", "patchwork", "scales"))
```

### Census API Key

The data pipeline calls the U.S. Census Bureau's ACS 5-Year API. A key is free and recommended to avoid rate limits.

1. Request a key at <https://api.census.gov/data/key_signup.html>.
2. Set the environment variable:

   ```bash
   # macOS / Linux
   export CENSUS_KEY="your_key_here"

   # Windows PowerShell
   $env:CENSUS_KEY = "your_key_here"

   # Conda environment
   conda env config vars set CENSUS_KEY=your_key_here
   ```

3. Verify in Python:

   ```python
   import os
   print(os.getenv("CENSUS_KEY"))
   ```

### Running the Data Pipeline

```python
from data_wrangling.dataGetter import DataGetter
from data_wrangling.dataProcessor import DataProcessor
from data_wrangling.merger import Merge
import os

CENSUS_KEY = os.getenv("CENSUS_KEY")

for year in range(2009, 2025):
    dg = DataGetter(CENSUS_KEY=CENSUS_KEY, year=year, download=False,
                    roni_raw_data_path="data_wrangling/static_data/RONI_data/rawData.csv",
                    coastal_type_raw_data_path="data_wrangling/static_data/Coastal_data_2010/coastal_counties_2010.csv")
    raw = dg.fetch_all()

    dp = DataProcessor(year=year, data_dict=raw)
    cleaned = dp.process_all()

    merger = Merge(cleaned, year=year, download=True)
    merged = merger.merge()
```

The merged output lands in `./merged_data/{year}/merged_data.csv`.

### Running the Models

Refer to `storm_dmg_regress/demo.ipynb` for the Python damage-prediction workflow and `storm_type_classifier/FinalStormClassifier.Rmd` for the R storm-type classifier (knit to HTML or run interactively in RStudio).

## Year Coverage

The pipeline supports **2009–2024** (constrained by ACS 5-Year availability and NOAA Storm Events file naming). Static datasets (coastal counties, RONI) are bundled and do not require additional downloads.
