"""
Configuration utilities for game theory experiments.
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for language models."""

    name: str
    max_tokens: int = 100
    temperature: float = 0.7
    provider: str = "openai"  # openai, anthropic, etc.


# Predefined model configurations
MODELS = {
    # OpenAI models
    "gpt-4": ModelConfig("gpt-4", max_tokens=100, temperature=0.7),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", max_tokens=100, temperature=0.7),
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", max_tokens=100, temperature=0.7),
    # Anthropic models
    "claude-3-opus": ModelConfig(
        "claude-3-opus-20240229", max_tokens=100, temperature=0.7, provider="anthropic"
    ),
    "claude-3-sonnet": ModelConfig(
        "claude-3-sonnet-20240229",
        max_tokens=100,
        temperature=0.7,
        provider="anthropic",
    ),
    "claude-3-haiku": ModelConfig(
        "claude-3-haiku-20240307", max_tokens=100, temperature=0.7, provider="anthropic"
    ),
}


# Predefined strategy sets for Axelrod tournaments
STRATEGY_SETS = {
    "basic": ["always_cooperate", "always_defect", "tit_for_tat"],
    "classic": [
        "always_cooperate",
        "always_defect",
        "tit_for_tat",
        "grudger",
        "random",
    ],
    "extended": [
        "always_cooperate",
        "always_defect",
        "tit_for_tat",
        "grudger",
        "random",
        "tit_for_two_tats",
        "generous_tit_for_tat",
        "pavlov",
        "suspicious_tit_for_tat",
    ],
    "competitive": ["always_defect", "grudger", "suspicious_tit_for_tat", "pavlov"],
    "cooperative": [
        "always_cooperate",
        "tit_for_tat",
        "generous_tit_for_tat",
        "tit_for_two_tats",
    ],
}


# Predefined dictator game scenario sets
SCENARIO_SETS = {
    "basic": [
        "charity_children",
        "another_person",
        "unemployed_person",
        "elderly_person",
    ],
    "charity_focus": ["charity_children", "food_bank", "medical_research"],
    "individual_focus": [
        "another_person",
        "student_textbooks",
        "unemployed_person",
        "elderly_person",
        "community_member",
    ],
    "comprehensive": [
        "another_person",
        "charity_children",
        "student_textbooks",
        "unemployed_person",
        "community_member",
        "food_bank",
        "elderly_person",
        "disaster_family",
        "medical_research",
        "child_education",
    ],
}


# Default experiment configurations
DEFAULT_CONFIGS = {
    "axelrod_quick": {
        "experiment_type": "axelrod",
        "strategies": STRATEGY_SETS["basic"],
        "rounds_per_match": 10,
        "use_reasoning": False,
    },
    "axelrod_standard": {
        "experiment_type": "axelrod",
        "strategies": STRATEGY_SETS["classic"],
        "rounds_per_match": 50,
        "use_reasoning": False,
    },
    "axelrod_full": {
        "experiment_type": "axelrod",
        "strategies": STRATEGY_SETS["extended"],
        "rounds_per_match": 100,
        "use_reasoning": True,
    },
    "dictator_quick": {
        "experiment_type": "dictator",
        "custom_scenarios": SCENARIO_SETS["basic"],
        "use_reasoning": False,
    },
    "dictator_standard": {
        "experiment_type": "dictator",
        "custom_scenarios": SCENARIO_SETS["comprehensive"],
        "use_reasoning": False,
    },
    "dictator_reasoning": {
        "experiment_type": "dictator",
        "custom_scenarios": SCENARIO_SETS["comprehensive"],
        "use_reasoning": True,
    },
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name in MODELS:
        return MODELS[model_name]

    # Default configuration for unknown models
    return ModelConfig(model_name)


def get_strategy_set(set_name: str) -> List[str]:
    """Get a predefined set of strategies."""
    if set_name in STRATEGY_SETS:
        return STRATEGY_SETS[set_name].copy()

    raise ValueError(
        f"Unknown strategy set: {set_name}. Available: {list(STRATEGY_SETS.keys())}"
    )


def get_scenario_set(set_name: str) -> List[str]:
    """Get a predefined set of dictator game scenarios."""
    if set_name in SCENARIO_SETS:
        return SCENARIO_SETS[set_name].copy()

    raise ValueError(
        f"Unknown scenario set: {set_name}. Available: {list(SCENARIO_SETS.keys())}"
    )


def get_default_config(config_name: str) -> Dict[str, Any]:
    """Get a predefined experiment configuration."""
    if config_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[config_name].copy()

    raise ValueError(
        f"Unknown config: {config_name}. Available: {list(DEFAULT_CONFIGS.keys())}"
    )


def list_available_models() -> List[str]:
    """List all available predefined models."""
    return list(MODELS.keys())


def list_available_strategies() -> List[str]:
    """List all available strategies."""
    all_strategies = set()
    for strategies in STRATEGY_SETS.values():
        all_strategies.update(strategies)
    return sorted(list(all_strategies))


def list_available_scenarios() -> List[str]:
    """List all available dictator game scenarios."""
    all_scenarios = set()
    for scenarios in SCENARIO_SETS.values():
        all_scenarios.update(scenarios)
    return sorted(list(all_scenarios))


# Environment variable names for API keys
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def validate_environment(model_name: str) -> bool:
    """Check if required environment variables are set."""
    import os

    model_config = get_model_config(model_name)
    provider = model_config.provider

    if provider in API_KEY_VARS:
        env_var = API_KEY_VARS[provider]
        return env_var in os.environ and os.environ[env_var].strip() != ""

    return True  # Unknown provider, assume it's okay
