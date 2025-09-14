# Game Theory Bench

A testbed for evaluating game-theoretic behavior of language models. The repository currently includes canonical experiments such as the Axelrod tournament (iterated Prisoner’s Dilemma) and the Dictator game, with a modular design to add many more experiments over time. It supports integration with multiple model providers.

## Features

- Canonical experiments (e.g., Axelrod tournament, Dictator game)
- Multi-model support via adapters
- Optional reasoning mode (not persisted to results)
- Analysis utilities: statistics and leaderboards
- Modular design to add new experiments

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd GameTheoryBench

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose your provider)
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### Usage

```bash
# Run experiments
python main.py run gpt-3.5-turbo axelrod
python main.py run gpt-4 dictator
python main.py run gpt-4 axelrod --config axelrod_full

# Compare multiple models
python main.py compare gpt-3.5-turbo gpt-4 dictator
```

### Programmatic Usage

```python
import src as gt

# Run single experiment
evaluator = gt.GameTheoryEvaluator()
config = gt.ExperimentConfig(
    experiment_type="axelrod",
    model_name="gpt-4",
    strategies=gt.get_strategy_set("extended"),
    rounds_per_match=100
)
result = evaluator.run_experiment(config)

# Compare models
comparison = gt.compare_models(
    ["gpt-3.5-turbo", "gpt-4"], 
    experiment_type="dictator"
)
```

## Games

### Axelrod Tournament (Prisoner's Dilemma)

Iterated Prisoner’s Dilemma in which a model-controlled player competes against deterministic strategies.

**Strategies Available:**
- `always_cooperate`, `always_defect`  
- `tit_for_tat`, `generous_tit_for_tat`, `suspicious_tit_for_tat`
- `grudger`, `pavlov`, `random`
- `tit_for_two_tats`

**Payoff Matrix (per round):**
- Both cooperate: 3 points each
- Both defect: 1 point each
- Defect vs Cooperate: 5 vs 0 points

### Dictator Game

Economic experiment measuring allocation decisions. A model-controlled player determines how much of an endowment to give to a recipient.

**Scenarios Available:**
- Charity donations (children, general causes)
- Anonymous strangers
- Known participants  
- Disadvantaged recipients

## Configuration

The system includes predefined configurations for quick experiments:

```python
# Quick configs
gt.get_default_config("axelrod_quick")    # 3 strategies, 10 rounds
gt.get_default_config("axelrod_standard") # 5 strategies, 50 rounds  
gt.get_default_config("dictator_quick")   # 2 scenarios

# Strategy sets
gt.get_strategy_set("basic")      # 3 basic strategies
gt.get_strategy_set("extended")   # All 9 strategies
gt.get_strategy_set("competitive") # Aggressive strategies only

# Scenario sets  
gt.get_scenario_set("basic")        # 2 basic scenarios
gt.get_scenario_set("comprehensive") # All 5 scenarios
```

## Analysis

Built-in analysis tools for experiment results:

```python
# Generate summary-only report
gt.generate_report("results/experiment_results_20240101.json")

# Leaderboard from results
gt.print_leaderboard("results.json")
```

## Information and Visibility

- Axelrod tournament: the model sees the current round index, total rounds, and a bounded summary of recent observed history. Opponent identifiers are neutral and do not encode strategy identity. Current-round opponent actions are never revealed prior to the model’s action.
- Dictator game: the model sees the endowment, recipient description, anonymity flag, and optional scenario context. The model returns a numeric allocation. Results store allocations and derived metrics only; internal reasoning traces are not persisted.

## Architecture

```
src/
├── core/         # Base classes and evaluator
├── games/        # Game implementations (Axelrod, Dictator)
├── signatures/   # Signatures and prompt formatters
└── utils/        # Configuration and analysis utilities
```

**Key Components:**
- `GameTheoryEvaluator`: Experiment orchestration
- `ExperimentConfig`: Experiment configuration
- `AxelrodTournament`: Iterated Prisoner’s Dilemma
- `DictatorExperiment`: Dictator game

## Extending

### Adding New Games

1. Create game class inheriting from `Game`
2. Implement signatures in `signatures/`
3. Add experiment runner function
4. Register in `utils/config.py`

### Adding New Strategies

```python
# Add to AxelrodStrategies class
@staticmethod  
def my_strategy(game_state: GameState, game_config: Dict[str, Any]) -> str:
    # Your strategy logic
    return "COOPERATE" or "DEFECT"

# Register in STRATEGY_SETS
STRATEGY_SETS["my_set"] = ["my_strategy", ...]
```

## CLI Reference

```bash
# Available commands
python main.py run <model> <experiment>      # Run experiment
python main.py compare <models> <experiment> # Model comparison  
python main.py list                          # Show available options
python main.py analyze <results_file>        # Generate summary (no plots)
```

## Supported Models

Works with configured providers. Examples include:
- OpenAI models (e.g., `gpt-4`, `gpt-3.5-turbo`).
- Anthropic models (e.g., `claude-3-sonnet`).

## Results Format

Experiment results are saved as JSON with comprehensive metadata. Example keys for Axelrod:

```json
{
  "config": { "experiment_type": "axelrod", "model_name": "gpt-4" },
  "results": {
    "total_score": 245,
    "total_rounds": 300,
    "average_score_per_round": 0.82,
    "performance_ratio": 0.85,
    "match_results": [ { "opponent": "tit_for_tat", "llm_score": 90, ... } ],
    "behavioral_metrics": {
      "llm_cooperation_rate": 0.62,
      "opponent_cooperation_rate": 0.58,
      "outcome_counts": { "CC": 120, "CD": 70, "DC": 60, "DD": 50 },
      "niceness_rate": 0.78,
      "retaliation_rate": 0.66,
      "forgiveness_rate": 0.21,
      "switch_rate": 0.34,
      "social_welfare": { "total": 720, "per_round_avg": 2.40 },
      "per_opponent": { "tit_for_tat": { "llm_cooperation_rate": 0.7, ... } }
    }
  },
  "timestamp": "2024-01-01_12:00:00",
  "duration_seconds": 45.2,
  "success": true
}
```

For Dictator experiments, the `results` include:

```json
{
  "average_altruism_ratio": 0.54,
  "median_altruism_ratio": 0.50,
  "num_scenarios": 12,
  "altruism_by_recipient_type": { "another person": 0.50, "charity...": 0.59 },
  "altruism_by_endowment": { "50": 0.50, "100": 0.53, "200": 0.58 },
  "altruism_by_anonymity": { "True": 0.54, "False": 0.52 },
  "altruism_distribution": { "p25": 0.50, "p50": 0.50, "p75": 0.60 },
  "endowment_altruism_correlation": 0.22,
  "scenario_results": [ { "scenario_name": "charity_children_50", "amount_given": 30.0, ... } ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Security and Privacy

- Do not commit API keys. Use environment variables or an ignored `.env` file locally.
- The system does not write internal reasoning traces to results; outputs capture decisions and derived metrics only.
