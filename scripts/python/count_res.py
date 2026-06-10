#!/usr/bin/env python3
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model")
    args = parser.parse_args()

    csv_path = os.environ["SCRATCH"] + "/grounding-vlms/results/combinedResults.csv"

    df = pd.read_csv(csv_path)
    print("Columns in dataset:", df.columns.tolist())
    print("Available models in dataset:", df["model"].unique())
    return
    df = df[df["model"] == args.model_name]
    # Ensure integer columns are numeric
    df["truth"] = pd.to_numeric(df["truth"], errors="coerce")
    df["model_result"] = pd.to_numeric(df["model_result"], errors="coerce")

    # Drop rows where either value is missing / non-numeric
    df = df.dropna(subset=["truth", "model_result"])

    df["error"] = df["model_result"] - df["truth"]
    df["abs_error"] = df["error"].abs()
    df["correct"] = df["model_result"] == df["truth"]

    print("\n=== Summary Statistics ===")
    print(f"N: {len(df)}")
    print(f"Accuracy: {df['correct'].mean():.4f}")
    print(f"Mean truth: {df['truth'].mean():.4f}")
    print(f"Mean model_result: {df['model_result'].mean():.4f}")

    print("\n=== Error Metrics ===")
    print(f"Mean error / bias: {df['error'].mean():.4f}")
    print(f"Mean absolute error: {df['abs_error'].mean():.4f}")
    print(f"Median absolute error: {df['abs_error'].median():.4f}")
    print(f"Max absolute error: {df['abs_error'].max()}")

    print("\n=== Over/Underestimation ===")
    print(f"Over rate: {(df['error'] > 0).mean():.4f}")
    print(f"Under rate: {(df['error'] < 0).mean():.4f}")

    print("\n=== Truth Distribution ===")
    print(df["truth"].describe())

    print("\n=== Model Result Distribution ===")
    print(df["model_result"].describe())

    print("\n=== Error Distribution ===")
    print(df["error"].describe())

    print("\n=== Confusion-like Counts ===")
    print(
        df.groupby(["truth", "model_result"])
        .size()
        .reset_index(name="count")
        .sort_values(["truth", "model_result"])
        .to_string(index=False)
    )

    print("\n=== Per-Truth Accuracy ===")
    print(
        df.groupby("truth")
        .agg(
            n=("truth", "size"),
            accuracy=("correct", "mean"),
            mean_abs_error=("abs_error", "mean"),
            mean_model_result=("model_result", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
