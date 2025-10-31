import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Union
import random


def WP_no_ties(Ta, Tb, Da, Db):
    # Ta, Tb = Time of death or censoring of group A or B respectively
    # Da, Db = Indicator of death (1) or censoring (0) for group A or B respectively

    if len(Ta) != len(Da) or len(Tb) != len(Db):
        raise ValueError("Input arrays must have the same length within each group.")
    if len(Ta) == 0 or len(Tb) == 0 or len(Da) == 0 or len(Db) == 0:
        raise ValueError("Input arrays must not be empty.")

    Wa = 0
    games = len(Ta) * len(Tb)
    for a in range(len(Ta)):
        for b in range(len(Tb)):
            if Ta[a] < 0 or Tb[b] < 0:
                raise ValueError("Time values must be non-negative.")
            if Da[a] not in [0, 1] or Db[b] not in [0, 1]:
                raise ValueError("Death indicators must be 0 or 1.")

            if Da[a] == 0 and Db[b] == 0:
                games -= 1
            elif Da[a] == 1 and Db[b] == 0:
                if Ta[a] > Tb[b]:
                    games -= 1
            elif Da[a] == 0 and Db[b] == 1:
                if Tb[b] < Ta[a]:
                    Wa += 1.0
                else:
                    games -= 1
            else:
                if Ta[a] > Tb[b]:
                    Wa += 1
                elif Ta[a] == Tb[b]:
                    games -= 1

    return Wa / games if games > 0 else None


def find_x_for_max_y_leq(
    df: pd.DataFrame, value: float
) -> Optional[Union[float, pd.Series]]:
    """Find x in `df` whose y is the largest but <= value.
    If there are multiple such x, return the smallest one. If there is no big enough y return False as death indicator (this means censoring).

    - `df` is expected to have two columns: x and y (names 'X'/'Y' or 'x'/'y' or first two cols).
    - `value` is a float (expected between 0 and 1 by caller).
    """
    # Accept any numeric `value`; don't restrict to [0,1]
    try:
        value = float(value)
    except Exception:
        raise ValueError("value must be numeric")

    # Infer column names
    if "Y" in df.columns and "X" in df.columns:
        y_col, x_col = "Y", "X"
    elif "y" in df.columns and "x" in df.columns:
        y_col, x_col = "y", "x"
    else:
        x_col, y_col = df.columns[0], df.columns[1]

    # Ensure numeric
    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    x_vals = df[x_col]

    # We might have value larger than any y in table
    if y_vals.iloc[-1] < value:
        return x_vals.iloc[-1], False

    # Find the largest y <= value (the second output is death indicator and is True if found)
    i = len(y_vals) - 1
    while i > 0 and y_vals.iloc[i - 1] >= value:
        i -= 1
    i -= 1

    # Doublecheck before returning
    if y_vals.iloc[i] <= value:
        return x_vals.iloc[i], True
    else:
        return None, None


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    placebo = pd.read_csv(
        "original_points_placebo.csv"
    )  # We load the points with extra boundary points already added
    treatment = pd.read_csv(
        "original_points_treatment.csv"
    )  # We load the points with extra boundary points already added

    # Parameters
    sample_number_placebo = 208
    sample_number_treatment = 212
    number_of_simulations = 10000

    WR_values = []
    for i in range(number_of_simulations):
        placebo_samples_x = []
        placebo_deaths = []
        placebo_samples_y = []
        for i in range(sample_number_placebo):
            y_sample = np.random.uniform(0.0, 1.0)
            x_sample, death = find_x_for_max_y_leq(placebo, y_sample)
            placebo_samples_x.append((x_sample))
            placebo_deaths.append(death)
            placebo_samples_y.append((y_sample))

        treatment_samples_x = []
        treatment_samples_y = []
        treatment_deaths = []
        for i in range(sample_number_treatment):
            y_sample = np.random.uniform(0.0, 1.0)
            x_sample, death = find_x_for_max_y_leq(treatment, y_sample)
            treatment_samples_x.append((x_sample))
            treatment_samples_y.append((y_sample))
            treatment_deaths.append(death)

        Ta = treatment_samples_x
        Tb = placebo_samples_x
        Da = treatment_deaths
        Db = placebo_deaths
        wp = WP_no_ties(Ta, Tb, Da, Db)
        if wp > 0.999:  # If WP is too high, we skip this simulation
            continue
        WR_values.append(wp / (1 - wp))

    plt.hist(WR_values, bins=20, color="skyblue", edgecolor="black")
    print(f"Mean WR between treatment and placebo: {np.mean(WR_values)}")
    print(f"Median WR between treatment and placebo: {np.median(WR_values)}")
    plt.title("Histogram of WR Values")
    plt.xlabel("WR Value")
    plt.ylabel("Frequency")
    plt.show()
