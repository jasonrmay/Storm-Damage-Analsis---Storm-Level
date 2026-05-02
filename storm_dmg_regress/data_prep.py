import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "BEGIN_LAT",
    "BEGIN_LON",
    "STORM_AREA_SQMILES",
    "DURATION_MINUTES",
    "TEMPERATURE_F",
    "ANOMALY_F",
    "RONI_AVG",
    "POPULATION",
    "MEDIAN_INCOME",
    "MEDIAN_YEAR_BUILT",
]

CATEGORICAL_COLS = [
    "EVENT_TYPE",
    "MODAL_YEAR_BUILT_BIN",
    "COASTAL_TYPE_SHORELINE",
    "COASTAL_TYPE_WATERSHED",
]

CYCLICAL_COLS = ["month_sin", "month_cos"]

DROP_COLS = ["STATE", "MONTH_NAME", "CZ_NAME", "FIPS", "MONTH", "DAMAGE_PROPERTY"]

TARGET_RAW = "DAMAGE_PROPERTY"
TARGET = "log10_damage"

def filter_target(df: pd.DataFrame, drop_zeros: bool = True) -> pd.DataFrame:
    """Drop rows with zero damage, replace na with median damage for that county and storm type
    and add the log10 target."""

    df.groupby(["CZ_NAME", "EVENT_TYPE"])[TARGET_RAW].transform(lambda x: x.fillna(x.median()))

    if drop_zeros:
        df = df.loc[df[TARGET_RAW] > 0].copy()
        df[TARGET] = np.log10(df[TARGET_RAW])
        return df

    return df


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    ''' Encode month as cyclical features. '''
    radians = 2 * np.pi * (df["MONTH"].astype(float) - 1) / 12.0
    df["month_sin"] = np.sin(radians)
    df["month_cos"] = np.cos(radians)
    return df


def prep_data(df: pd.DataFrame, drop_zeros: bool = True) -> pd.DataFrame:
    
    out = df.copy()

    out = filter_target(out, drop_zeros)

    out = add_cyclical_month(out)

    if drop_zeros:
        keep = NUMERIC_COLS + CATEGORICAL_COLS + CYCLICAL_COLS + [TARGET]
        out = out[keep]
    else:
        keep = NUMERIC_COLS + CATEGORICAL_COLS + CYCLICAL_COLS + [TARGET_RAW]
        out = out[keep]

    # Coerce categoricals to string so get_dummies behaves predictably.
    for c in CATEGORICAL_COLS:
        out[c] = out[c].astype("string").dropna()
        
    # One-hot encode the categoricals. Keep all levels for interpretability.
    out = pd.get_dummies(out, columns=CATEGORICAL_COLS, drop_first=False, dtype=np.uint8)

    return out