#!/usr/bin/env python3
"""
Game Theory Bench - Command line interface for running game theory experiments.
"""
import argparse
import sys
import os
from typing import List

# Load environment variables from .env file
import load_env

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src as gt


def run_experiment(model_name: str, experiment_type: str, config: str = None):
    """Run an experiment with specified model and type."""
    print(f"Running {experiment_type} experiment with {model_name}")

    if not gt.validate_environment(model_name):
        print(f"Error: Environment not configured for {model_name}")
        return False

    evaluator = gt.GameTheoryEvaluator()

    # Use provided config or default
    if config and config in gt.DEFAULT_CONFIGS:
        config_dict = gt.get_default_config(config)
        config_dict["model_name"] = model_name
        exp_config = gt.ExperimentConfig(**config_dict)
    else:
        # Use standard configuration
        if experiment_type == "axelrod":
            exp_config = gt.ExperimentConfig(
                experiment_type="axelrod",
                model_name=model_name,
                strategies=gt.get_strategy_set("basic"),
                rounds_per_match=50,
            )
        elif experiment_type == "dictator":
            exp_config = gt.ExperimentConfig(
                experiment_type="dictator",
                model_name=model_name,
                custom_scenarios=gt.get_scenario_set("basic"),
            )
        else:
            print(f"Error: Unknown experiment type: {experiment_type}")
            return False

    result = evaluator.run_experiment(exp_config)

    if result.success:
        results_file = evaluator.save_results(result)
        print(f"Experiment completed in {result.duration_seconds:.2f} seconds")
        print(f"Results saved to: {results_file}")

        # Display key results
        if experiment_type == "axelrod":
            print(f"Total Score: {result.results['total_score']}")
            print(f"Performance Ratio: {result.results['performance_ratio']:.3f}")
        elif experiment_type == "dictator":
            print(f"Average Altruism: {result.results['average_altruism_ratio']:.1%}")
            print(f"Scenarios Tested: {result.results['num_scenarios']}")

        return True
    else:
        print(f"Experiment failed: {result.error_message}")
        return False


def compare_models(models: List[str], experiment_type: str):
    """Compare multiple models on the same experiment."""
    print(f"Comparing {len(models)} models on {experiment_type} experiment")

    # Validate environments
    valid_models = []
    for model in models:
        if gt.validate_environment(model):
            valid_models.append(model)
        else:
            print(f"Warning: Skipping {model} - environment not configured")

    if not valid_models:
        print("Error: No valid models to compare")
        return False

    evaluator = gt.GameTheoryEvaluator()
    comparison = evaluator.compare_models(valid_models, experiment_type)

    # Save results
    results_file = evaluator.save_results(
        [result for result in comparison["individual_results"] if result.success]
    )

    # Print results
    gt.print_leaderboard(results_file)
    print(f"Full results saved to: {results_file}")
    return True


def list_available():
    """List all available configurations."""
    print("AVAILABLE MODELS:")
    for model in gt.list_available_models():
        configured = (
            "configured" if gt.validate_environment(model) else "not configured"
        )
        print(f"  {model} ({configured})")

    print("\nAVAILABLE EXPERIMENT CONFIGURATIONS:")
    for config_name in gt.DEFAULT_CONFIGS.keys():
        print(f"  {config_name}")

    print("\nAVAILABLE STRATEGIES:")
    print(f"  {', '.join(gt.list_available_strategies())}")

    print("\nAVAILABLE SCENARIOS:")
    print(f"  {', '.join(gt.list_available_scenarios())}")


def main():
    parser = argparse.ArgumentParser(
        description="Game Theory Bench - LLM Game Theory Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python main.py run gpt-3.5-turbo axelrod
            python main.py run gpt-4 dictator
            python main.py run gpt-4 axelrod --config axelrod_full
            python main.py compare gpt-3.5-turbo gpt-4 dictator
            python main.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("model", help="Model name")
    run_parser.add_argument(
        "experiment", choices=["axelrod", "dictator"], help="Experiment type"
    )
    run_parser.add_argument("--config", help="Configuration name (optional)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument("models", nargs="+", help="Model names to compare")
    compare_parser.add_argument(
        "experiment", choices=["axelrod", "dictator"], help="Experiment type"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List available options")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("results_file", help="Path to results JSON file")
    analyze_parser.add_argument(
        "--output-dir", default="analysis", help="Output directory for analysis"
    )

    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args.model, args.experiment, args.config)

    elif args.command == "compare":
        compare_models(args.models, args.experiment)

    elif args.command == "list":
        list_available()

    elif args.command == "analyze":
        summary = gt.generate_report(args.results_file, args.output_dir)
        print(f"Analysis complete. Check {args.output_dir}/ for summary.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
