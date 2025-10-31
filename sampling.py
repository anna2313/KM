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

    - `df` is expected to have two columns: x and y (names 'X'/'Y' or 'x'/'y' or first two cols).
    - `value` is a float (expected between 0 and 1 by caller).

    Returns the x corresponding to the maximum y satisfying y <= value.
    If multiple rows share that y, returns a pandas Series of x values (preserving order).
    If no y <= value exists, returns None and prints a message.
    """
    # Accept any numeric `value`; don't restrict to [0,1] so callers can query
    # values outside that range (e.g., larger than any table y).
    try:
        value = float(value)
    except Exception:
        raise ValueError("value must be numeric")

    # infer column names
    if "Y" in df.columns and "X" in df.columns:
        y_col, x_col = "Y", "X"
    elif "y" in df.columns and "x" in df.columns:
        y_col, x_col = "y", "x"
    else:
        x_col, y_col = df.columns[0], df.columns[1]

    # ensure numeric
    y_vals = pd.to_numeric(df[y_col], errors="coerce")
    x_vals = df[x_col]

    if y_vals.iloc[-1] < value:
        return x_vals.iloc[-1], False

    i = len(y_vals) - 1
    while i > 0 and y_vals.iloc[i - 1] >= value:
        i -= 1
    i -= 1

    if y_vals.iloc[i] <= value:
        return x_vals.iloc[i], True
    else:
        return None, None


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    sampled_step_dotted = pd.read_csv("sampled_step_dotted.csv")
    sampled_step_full = pd.read_csv("sampled_step_full.csv")
    df_dotted = pd.read_csv("original_points_dotted.csv")
    df_full = pd.read_csv("original_points_full.csv")

    # Plot: sampled step and original points
    plt.figure(figsize=(8, 4))
    plt.step(
        sampled_step_dotted["x"],
        sampled_step_dotted["y"],
        where="post",
        label="Sampled Piecewise-Constant (Dotted)",
    )
    plt.scatter(
        df_dotted.iloc[:, 0],
        df_dotted.iloc[:, 1],
        color="red",
        label="Original Points (Dotted)",
    )

    plt.step(
        sampled_step_full["x"],
        sampled_step_full["y"],
        where="post",
        label="Sampled Piecewise-Constant (Full)",
    )
    plt.scatter(
        df_full.iloc[:, 0],
        df_full.iloc[:, 1],
        color="blue",
        label="Original Points (Full)",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Piecewise-Constant Sampling")
    plt.legend()
    plt.show()

    WR_values = []
    for i in range(10000):
        sample_number = 200
        dotted_samples_x = []
        dotted_deaths = []
        dotted_samples_y = []
        for i in range(208):
            y_sample = np.random.uniform(0.0, 1.0)
            x_sample, death = find_x_for_max_y_leq(df_dotted, y_sample)
            dotted_samples_x.append((x_sample))
            dotted_deaths.append(death)
            dotted_samples_y.append((y_sample))

        full_samples_x = []
        full_samples_y = []
        full_deaths = []
        for i in range(212):
            y_sample = np.random.uniform(0.0, 1.0)
            x_sample, death = find_x_for_max_y_leq(df_full, y_sample)
            full_samples_x.append((x_sample))
            full_samples_y.append((y_sample))
            full_deaths.append(death)

        Ta = full_samples_x
        Tb = dotted_samples_x
        Da = full_deaths
        Db = dotted_deaths
        wp = WP_no_ties(Ta, Tb, Da, Db)
        """print(f"Calculated WP between full and dotted samples: {wp}")"""
        if wp > 0.999:
            continue
        """print(f"Calculated WR between full and dotted samples: {wp/(1-wp)}")"""
        WR_values.append(wp / (1 - wp))

    plt.hist(WR_values, bins=20, color="skyblue", edgecolor="black")
    print(f"Mean WR between full and dotted samples: {np.mean(WR_values)}")
    print(f"Median WR between full and dotted samples: {np.median(WR_values)}")
    plt.title("Histogram of WR Values")
    plt.xlabel("WR Value")
    plt.ylabel("Frequency")
    plt.show()
