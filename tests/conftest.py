import numpy as np
import pandas as pd
import pytest


def make_binary_frame(rows: int = 180, seed: int = 7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age = rng.randint(20, 66, rows)
    income = rng.normal(5200, 1600, rows).round(0)
    debt_ratio = rng.uniform(0.0, 1.0, rows).round(3)
    employment_years = rng.randint(0, 31, rows)
    has_default = rng.rand(rows) < 0.22
    region = rng.choice(["north", "south", "east", "west"], rows)
    score = (
        0.00007 * income
        + 0.09 * employment_years
        - 3.2 * debt_ratio
        - 1.4 * has_default
        + rng.normal(0.0, 0.45, rows)
        - 0.9
    )
    approved = score > 0
    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "employment_years": employment_years,
            "has_default": has_default,
            "region": region,
            "approved": approved,
        }
    )


def make_multiclass_frame(rows: int = 180, seed: int = 11) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age = rng.randint(20, 66, rows)
    income = rng.normal(5200, 1600, rows).round(0)
    debt_ratio = rng.uniform(0.0, 1.0, rows).round(3)
    region = rng.choice(["north", "south", "east", "west"], rows)
    score = (
        0.00007 * income
        - 3.2 * debt_ratio
        + rng.normal(0.0, 0.6, rows)
    )
    risk = np.where(score > 0.4, "low", np.where(score > -0.4, "medium", "high"))
    return pd.DataFrame(
        {
            "age": age,
            "income": income,
            "debt_ratio": debt_ratio,
            "region": region,
            "risk": risk,
        }
    )


@pytest.fixture()
def binary_frame() -> pd.DataFrame:
    return make_binary_frame()


@pytest.fixture()
def multiclass_frame() -> pd.DataFrame:
    return make_multiclass_frame()
