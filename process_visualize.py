import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sample_piecewise_constant(
    df: pd.DataFrame, upper_bound: float, epsilon: float = 0.01
) -> pd.DataFrame:
    """Return a sampled piecewise-constant function DataFrame with columns 'x' and 'y'.

    Behavior & assumptions:
    - The function domain is [first_x, last_x] where first_x and last_x are taken from the
      provided DataFrame (sorted by x).
    - Between table x-values the function is constant and equal to the y-value at the left
      endpoint (a left-continuous step). This is implemented using searchsorted
      (values on [x_i, x_{i+1}) get y_i).
    - If the input DataFrame is empty, we return an error.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    # Handle empty input
    if df is None or df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    if upper_bound is None:
        raise ValueError("upper_bound must not be None.")
    if upper_bound <= 0:
        raise ValueError("upper_bound must be > 0")

    # Normalize column names: accept X/Y or x/y or infer first two columns
    if "X" in df.columns and "Y" in df.columns:
        x_col, y_col = "X", "Y"
    elif "x" in df.columns and "y" in df.columns:
        x_col, y_col = "x", "y"
    else:
        x_col, y_col = df.columns[0], df.columns[1]

    # Add boundary points at x=0 and x=upper_bound
    new_row = pd.DataFrame({x_col: [0], y_col: [0]})
    df = pd.concat([new_row, df], ignore_index=True)
    new_row = pd.DataFrame({x_col: [upper_bound], y_col: [df[y_col].iloc[-1]]})
    df = pd.concat([df, new_row], ignore_index=True)

    # Ensure monotonicity
    list_idx = list(df.index)
    idx_old = list_idx[0]
    for i in range(1, len(list_idx)):
        idx = list_idx[i]
        if np.abs(df.at[idx, x_col] - df.at[idx_old, x_col]) < np.abs(
            df.at[idx, y_col] - df.at[idx_old, y_col]
        ):
            df.at[idx, x_col] = df.at[idx_old, x_col]
        else:
            df.at[idx, y_col] = df.at[idx_old, y_col]
        idx_old = idx
    x_vals = np.asarray(df[x_col], dtype=float)
    y_vals = np.asarray(df[y_col], dtype=float)

    x_min, x_max = x_vals[0], x_vals[-1]

    # Build sampling points from x_min to x_max inclusive
    n_steps = max(1, int(np.ceil((x_max - x_min) / epsilon)))
    sample_x = np.linspace(x_min, x_max, n_steps + 1)

    # For each sample point, find index of left interval: x_i <= sample_x < x_{i+1}
    sample_y = np.zeros_like(sample_x)
    idx_x = 0
    idx_y = 0
    while idx_y < len(sample_y):
        if idx_x == len(y_vals) - 1:
            sample_y[idx_y] = y_vals[idx_x]
            idx_y += 1
        elif sample_x[idx_y] < x_vals[idx_x + 1]:
            sample_y[idx_y] = y_vals[idx_x]
            idx_y += 1
        else:
            idx_x += 1

    return pd.DataFrame({"x": sample_x, "y": sample_y}), df


if __name__ == "__main__":
    # Read input CSV files
    # CSV files have two columns without headers (X, Y pairs) separated by commas

    file_path_placebo = "placebo.csv"
    df_placebo = pd.read_csv(file_path_placebo, names=["X", "Y"])

    file_path_treatment = "treatment.csv"
    df_treatment = pd.read_csv(file_path_treatment, names=["X", "Y"])

    # Sample piecewise-constant functions
    epsilon = 0.001  # Step size for sampling
    upper_bound = 5.0  # Upper bound for x-values (time horizon)

    sampled_placebo, placebo = sample_piecewise_constant(
        df_placebo, upper_bound=upper_bound, epsilon=epsilon
    )
    print("\nSampled (first rows):")
    print(sampled_placebo)

    sampled_treatment, treatment = sample_piecewise_constant(
        df_treatment, upper_bound=upper_bound, epsilon=epsilon
    )
    print("\nSampled (first rows):")
    print(sampled_treatment)

    sampled_placebo.to_csv("sampled_placebo.csv", index=False)
    sampled_treatment.to_csv("sampled_treatment.csv", index=False)
    placebo.to_csv("original_points_placebo.csv", index=False)
    treatment.to_csv("original_points_treatment.csv", index=False)

    # Plot: sampled step and original points
    plt.figure(figsize=(8, 4))
    plt.step(
        sampled_placebo["x"],
        sampled_placebo["y"],
        where="post",
        label="Sampled Piecewise-Constant (Placebo)",
    )
    plt.scatter(
        placebo.iloc[:, 0],
        placebo.iloc[:, 1],
        color="red",
        label="Original Points (Placebo)",
    )

    plt.step(
        sampled_treatment["x"],
        sampled_treatment["y"],
        where="post",
        label="Sampled Piecewise-Constant (Treatment)",
    )
    plt.scatter(
        treatment.iloc[:, 0],
        treatment.iloc[:, 1],
        color="blue",
        label="Original Points (Treatment)",
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Piecewise-Constant Sampling")
    plt.legend()
    plt.show()
