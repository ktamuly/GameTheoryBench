"""
Game Theory Bench – a testbed for game-theoretic experiments with language models.

This package provides formal implementations of classic experiments
and supports integration with multiple model providers.
"""

from .core.evaluator import GameTheoryEvaluator, ExperimentConfig, ExperimentResult

from .games.axelrod import run_axelrod_experiment, AxelrodTournament, AxelrodPlayer

from .games.dictator import (
    run_dictator_experiment,
    DictatorExperiment,
    DictatorPlayer,
)

from .utils.config import (
    MODELS,
    STRATEGY_SETS,
    SCENARIO_SETS,
    DEFAULT_CONFIGS,
    get_model_config,
    get_strategy_set,
    get_scenario_set,
    get_default_config,
    validate_environment,
    list_available_models,
    list_available_strategies,
    list_available_scenarios,
)

from .utils.analysis import (
    load_results_as_dataframe,
    generate_report,
    print_leaderboard,
)

__version__ = "1.0.0"
__author__ = "Game Theory Bench"


def compare_models(models: list, experiment_type: str = "axelrod", **kwargs):
    """Compare multiple models on an experiment."""
    evaluator = GameTheoryEvaluator()
    return evaluator.compare_models(models, experiment_type, **kwargs)


# Package info
__all__ = [
    # Core classes
    "GameTheoryEvaluator",
    "ExperimentConfig",
    "ExperimentResult",
    # Experiment functions
    "run_axelrod_experiment",
    "run_dictator_experiment",
    "compare_models",
    # Configuration
    "MODELS",
    "STRATEGY_SETS",
    "SCENARIO_SETS",
    "DEFAULT_CONFIGS",
    "get_model_config",
    "get_strategy_set",
    "get_scenario_set",
    "get_default_config",
    "validate_environment",
    "list_available_models",
    "list_available_strategies",
    "list_available_scenarios",
    # Analysis
    "load_results_as_dataframe",
    "generate_report",
    "print_leaderboard",
]
