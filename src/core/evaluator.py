"""
Evaluation framework for running game theory experiments.
"""
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import dspy
from ..games.axelrod import run_axelrod_experiment
from ..games.dictator import run_dictator_experiment


@dataclass
class ExperimentConfig:
    """Configuration for a game theory experiment."""

    experiment_type: str  # 'axelrod' or 'dictator'
    model_name: str
    player_name: Optional[str] = None
    use_reasoning: bool = False
    seed: Optional[int] = None

    # Axelrod-specific
    strategies: Optional[List[str]] = None
    rounds_per_match: int = 50

    # Dictator-specific
    scenarios: Optional[List[Dict[str, Any]]] = None
    custom_scenarios: Optional[List[str]] = None


@dataclass
class ExperimentResult:
    """Results from a game theory experiment."""

    config: ExperimentConfig
    results: Dict[str, Any]
    timestamp: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None


class GameTheoryEvaluator:
    """Main evaluator for running game theory experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment based on the configuration."""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        try:
            if config.experiment_type == "axelrod":
                results = run_axelrod_experiment(
                    model_name=config.model_name,
                    player_name=config.player_name,
                    use_reasoning=config.use_reasoning,
                    strategies=config.strategies,
                    rounds_per_match=config.rounds_per_match,
                    seed=config.seed,
                )

            elif config.experiment_type == "dictator":
                results = run_dictator_experiment(
                    model_name=config.model_name,
                    player_name=config.player_name,
                    use_reasoning=config.use_reasoning,
                    scenarios=config.scenarios,
                    custom_scenarios=config.custom_scenarios,
                )

            else:
                raise ValueError(f"Unknown experiment type: {config.experiment_type}")

            duration = time.time() - start_time

            experiment_result = ExperimentResult(
                config=config,
                results=results,
                timestamp=timestamp,
                duration_seconds=duration,
                success=True,
            )

        except Exception as e:
            duration = time.time() - start_time
            experiment_result = ExperimentResult(
                config=config,
                results={},
                timestamp=timestamp,
                duration_seconds=duration,
                success=False,
                error_message=str(e),
            )

        return experiment_result

    def run_batch_experiments(
        self, configs: List[ExperimentConfig]
    ) -> List[ExperimentResult]:
        """Run multiple experiments in batch."""
        results = []

        for i, config in enumerate(configs):
            print(
                f"Running experiment {i+1}/{len(configs)}: {config.experiment_type} with {config.model_name}"
            )
            result = self.run_experiment(config)
            results.append(result)

            if result.success:
                print(f"✓ Completed in {result.duration_seconds:.2f}s")
            else:
                print(f"✗ Failed: {result.error_message}")

        return results

    def save_results(
        self,
        results: Union[ExperimentResult, List[ExperimentResult]],
        filename: str = None,
    ) -> str:
        """Save experiment results to JSON file."""

        if isinstance(results, ExperimentResult):
            results = [results]

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Convert to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            # Convert ExperimentConfig to dict as well
            result_dict["config"] = asdict(result.config)
            serializable_results.append(result_dict)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        print(f"Results saved to: {filepath}")
        return str(filepath)

    def load_results(self, filepath: str) -> List[ExperimentResult]:
        """Load experiment results from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            config = ExperimentConfig(**item["config"])
            result = ExperimentResult(
                config=config,
                results=item["results"],
                timestamp=item["timestamp"],
                duration_seconds=item["duration_seconds"],
                success=item["success"],
                error_message=item.get("error_message"),
            )
            results.append(result)

        return results

    def compare_models(
        self, model_names: List[str], experiment_type: str = "axelrod", **kwargs
    ) -> Dict[str, Any]:
        """Compare multiple models on the same experiment."""

        configs = []
        for model_name in model_names:
            config = ExperimentConfig(
                experiment_type=experiment_type, model_name=model_name, **kwargs
            )
            configs.append(config)

        results = self.run_batch_experiments(configs)

        # Create comparison summary
        comparison = {
            "experiment_type": experiment_type,
            "models": model_names,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "individual_results": results,
            "summary": {},
        }

        if experiment_type == "axelrod":
            comparison["summary"] = self._compare_axelrod_results(results)
        elif experiment_type == "dictator":
            comparison["summary"] = self._compare_dictator_results(results)

        return comparison

    def _compare_axelrod_results(
        self, results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Create summary comparison for Axelrod tournament results."""
        summary = {"performance_ranking": [], "statistics": {}}

        performance_data = []
        for result in results:
            if result.success:
                model = result.config.model_name
                total_score = result.results.get("total_score", 0)
                avg_per_round = result.results.get("average_score_per_round", 0)
                performance_ratio = result.results.get("performance_ratio", 0)

                performance_data.append(
                    {
                        "model": model,
                        "total_score": total_score,
                        "avg_per_round": avg_per_round,
                        "performance_ratio": performance_ratio,
                    }
                )

        # Sort by performance ratio (normalized score)
        performance_data.sort(key=lambda x: x["performance_ratio"], reverse=True)
        summary["performance_ranking"] = performance_data

        if performance_data:
            summary["statistics"] = {
                "best_model": performance_data[0]["model"],
                "best_performance_ratio": performance_data[0]["performance_ratio"],
                "score_range": {
                    "min": min(p["total_score"] for p in performance_data),
                    "max": max(p["total_score"] for p in performance_data),
                },
            }

        return summary

    def _compare_dictator_results(
        self, results: List[ExperimentResult]
    ) -> Dict[str, Any]:
        """Create summary comparison for Dictator game results."""
        summary = {"altruism_ranking": [], "statistics": {}}

        altruism_data = []
        for result in results:
            if result.success:
                model = result.config.model_name
                avg_altruism = result.results.get("average_altruism_ratio", 0)
                num_scenarios = result.results.get("num_scenarios", 0)

                altruism_data.append(
                    {
                        "model": model,
                        "average_altruism_ratio": avg_altruism,
                        "num_scenarios": num_scenarios,
                    }
                )

        # Sort by altruism ratio
        altruism_data.sort(key=lambda x: x["average_altruism_ratio"], reverse=True)
        summary["altruism_ranking"] = altruism_data

        if altruism_data:
            summary["statistics"] = {
                "most_altruistic_model": altruism_data[0]["model"],
                "highest_altruism_ratio": altruism_data[0]["average_altruism_ratio"],
                "altruism_range": {
                    "min": min(p["average_altruism_ratio"] for p in altruism_data),
                    "max": max(p["average_altruism_ratio"] for p in altruism_data),
                },
            }

        return summary
