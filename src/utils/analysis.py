"""
Analysis utilities for game theory experiment results.
"""
import json
from typing import Dict, Any
import pandas as pd
from pathlib import Path


def load_results_as_dataframe(filepath: str) -> pd.DataFrame:
    """Load experiment results as a pandas DataFrame."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Flatten the results for analysis
    records = []
    for item in data:
        if not item["success"]:
            continue

        config = item["config"]
        results = item["results"]

        base_record = {
            "experiment_type": config["experiment_type"],
            "model_name": config["model_name"],
            "use_reasoning": config["use_reasoning"],
            "timestamp": item["timestamp"],
            "duration_seconds": item["duration_seconds"],
        }

        if config["experiment_type"] == "axelrod":
            record = base_record.copy()
            record.update(
                {
                    "total_score": results.get("total_score", 0),
                    "total_rounds": results.get("total_rounds", 0),
                    "num_opponents": results.get("num_opponents", 0),
                    "average_score_per_round": results.get(
                        "average_score_per_round", 0
                    ),
                    "performance_ratio": results.get("performance_ratio", 0),
                    "rounds_per_match": config.get("rounds_per_match", 50),
                }
            )
            records.append(record)

        elif config["experiment_type"] == "dictator":
            record = base_record.copy()
            record.update(
                {
                    "average_altruism_ratio": results.get("average_altruism_ratio", 0),
                    "num_scenarios": results.get("num_scenarios", 0),
                }
            )
            records.append(record)

    return pd.DataFrame(records)


def generate_report(results_path: str, output_dir: str = "analysis"):
    """Generate a summary analysis report (no plots)."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load data
    df = load_results_as_dataframe(results_path)

    if df.empty:
        print("No valid data to analyze")
        return

    # Generate summary statistics
    summary = generate_summary_stats(df)

    # Save summary as JSON
    summary_path = output_path / "summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Summary statistics saved to {summary_path}")

    return summary


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics from experiment results."""
    summary = {
        "total_experiments": len(df),
        "unique_models": df["model_name"].nunique(),
        "experiment_types": df["experiment_type"].value_counts().to_dict(),
    }

    # Axelrod statistics
    axelrod_df = df[df["experiment_type"] == "axelrod"]
    if not axelrod_df.empty:
        summary["axelrod"] = {
            "best_model": axelrod_df.loc[
                axelrod_df["performance_ratio"].idxmax(), "model_name"
            ],
            "best_performance_ratio": float(axelrod_df["performance_ratio"].max()),
            "average_performance_ratio": float(axelrod_df["performance_ratio"].mean()),
            "performance_by_model": axelrod_df.groupby("model_name")[
                "performance_ratio"
            ]
            .mean()
            .to_dict(),
        }

    # Dictator statistics
    dictator_df = df[df["experiment_type"] == "dictator"]
    if not dictator_df.empty:
        summary["dictator"] = {
            "most_altruistic_model": dictator_df.loc[
                dictator_df["average_altruism_ratio"].idxmax(), "model_name"
            ],
            "highest_altruism_ratio": float(
                dictator_df["average_altruism_ratio"].max()
            ),
            "average_altruism_ratio": float(
                dictator_df["average_altruism_ratio"].mean()
            ),
            "altruism_by_model": dictator_df.groupby("model_name")[
                "average_altruism_ratio"
            ]
            .mean()
            .to_dict(),
        }

    # Reasoning impact
    if "use_reasoning" in df.columns:
        reasoning_impact = {}

        for exp_type in df["experiment_type"].unique():
            type_df = df[df["experiment_type"] == exp_type]
            if exp_type == "axelrod" and "performance_ratio" in type_df.columns:
                reasoning_impact[exp_type] = {
                    "with_reasoning": float(
                        type_df[type_df["use_reasoning"] == True][
                            "performance_ratio"
                        ].mean()
                    ),
                    "without_reasoning": float(
                        type_df[type_df["use_reasoning"] == False][
                            "performance_ratio"
                        ].mean()
                    ),
                }
            elif exp_type == "dictator" and "average_altruism_ratio" in type_df.columns:
                reasoning_impact[exp_type] = {
                    "with_reasoning": float(
                        type_df[type_df["use_reasoning"] == True][
                            "average_altruism_ratio"
                        ].mean()
                    ),
                    "without_reasoning": float(
                        type_df[type_df["use_reasoning"] == False][
                            "average_altruism_ratio"
                        ].mean()
                    ),
                }

        summary["reasoning_impact"] = reasoning_impact

    return summary


def print_leaderboard(results_path: str):
    """Print a leaderboard of model performance."""
    df = load_results_as_dataframe(results_path)

    if df.empty:
        print("No data available")
        return

    print("\nMODEL LEADERBOARD")
    print("=" * 50)

    # Axelrod leaderboard
    axelrod_df = df[df["experiment_type"] == "axelrod"]
    if not axelrod_df.empty:
        print("\nAXELROD TOURNAMENT (Cooperation Strategy)")
        axelrod_ranking = (
            axelrod_df.groupby("model_name")["performance_ratio"]
            .mean()
            .sort_values(ascending=False)
        )

        for i, (model, score) in enumerate(axelrod_ranking.items(), 1):
            print(f"{i:2d}. {model:<20} Performance: {score:.3f}")

    # Dictator leaderboard
    dictator_df = df[df["experiment_type"] == "dictator"]
    if not dictator_df.empty:
        print("\nDICTATOR GAME (Altruism)")
        dictator_ranking = (
            dictator_df.groupby("model_name")["average_altruism_ratio"]
            .mean()
            .sort_values(ascending=False)
        )

        for i, (model, score) in enumerate(dictator_ranking.items(), 1):
            print(f"{i:2d}. {model:<20} Altruism: {score:.1%}")

    print("\n" + "=" * 50)
