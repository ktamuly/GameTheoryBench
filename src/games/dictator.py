"""
Dictator Game implementation.
Evaluates allocation behavior under varied recipient contexts and anonymity.
"""
import os
from typing import Dict, List, Any, Optional
import dspy
from ..core.game import Game, GameResult, TournamentResult, GameState, Player
from ..core.player import ModelPlayer
from ..signatures.dictator_sigs import (
    DictatorGameModule,
    format_dictator_scenario,
    format_recipient_info,
    DICTATOR_SCENARIOS,
)
from ..utils.config import get_model_config


class DictatorPlayer(ModelPlayer):
    """Model-controlled player for Dictator Game allocations."""

    def __init__(
        self, name: str, model_name: str = "gpt-3.5-turbo", use_reasoning: bool = False
    ):
        super().__init__(name, model_name)
        self.use_reasoning = use_reasoning
        self.dictator_module = DictatorGameModule(use_reasoning=use_reasoning)

    def make_move(self, game_state: GameState, game_config: Dict[str, Any]) -> float:
        """Return an allocation decision in the Dictator Game."""

        # Extract scenario details from config
        endowment = game_config.get("endowment", 100)
        recipient_type = game_config.get("recipient_type", "another person")
        recipient_characteristics = game_config.get("recipient_characteristics", {})
        scenario_context = game_config.get("context")
        anonymity = game_config.get("anonymity", True)

        # Prepare formatted inputs
        scenario = format_dictator_scenario(
            endowment=endowment,
            recipient_type=recipient_type,
            anonymity=anonymity,
            context=scenario_context,
        )

        recipient_info = format_recipient_info(
            recipient_type, recipient_characteristics
        )
        endowment_str = f"${endowment}"

        # Obtain allocation decision
        result = self.dictator_module(
            scenario=scenario, endowment=endowment_str, recipient_info=recipient_info
        )

        return result.amount


class DictatorGame(Game):
    """Single-player Dictator Game."""

    def __init__(
        self,
        endowment: float = 100,
        recipient_type: str = "another person",
        recipient_characteristics: Dict[str, Any] = None,
        anonymity: bool = True,
        context: str = None,
        config: Dict[str, Any] = None,
    ):

        super().__init__(config)
        self.endowment = endowment
        self.recipient_type = recipient_type
        self.recipient_characteristics = recipient_characteristics or {}
        self.anonymity = anonymity
        self.context = context

        # Update config with game parameters
        self.config.update(
            {
                "endowment": endowment,
                "recipient_type": recipient_type,
                "recipient_characteristics": self.recipient_characteristics,
                "anonymity": anonymity,
                "context": context,
            }
        )

        self.player = None
        self.allocation_made = False

    def add_player(self, player: Player):
        """Add the decision-making player."""
        if self.player is not None:
            raise ValueError("Dictator Game only supports 1 player")
        self.player = player

    def play_round(self) -> GameResult:
        """Execute the single allocation decision round."""
        if self.player is None:
            raise ValueError("No player added to the game")

        if self.allocation_made:
            raise ValueError("Game already completed")

        # Get allocation decision
        amount_given = self.player.make_move(self.state, self.config)

        # Validate allocation
        amount_given = max(0, min(amount_given, self.endowment))
        amount_kept = self.endowment - amount_given

        # Calculate altruism ratio
        altruism_ratio = amount_given / self.endowment if self.endowment > 0 else 0

        # Create result
        scores = {self.player.name: amount_kept}  # Player keeps the rest
        moves = {self.player.name: amount_given}  # Move is the amount given

        self.allocation_made = True

        return GameResult(
            player_scores=scores,
            moves=moves,
            metadata={
                "endowment": self.endowment,
                "amount_given": amount_given,
                "amount_kept": amount_kept,
                "altruism_ratio": altruism_ratio,
                "recipient_type": self.recipient_type,
                "anonymity": self.anonymity,
            },
            is_terminal=True,
        )

    def is_terminal(self) -> bool:
        """Dictator Game is always terminal after one decision."""
        return self.allocation_made


class DictatorExperiment:
    """Run multiple Dictator Game scenarios."""

    def __init__(
        self, llm_player: DictatorPlayer, scenarios: List[Dict[str, Any]] = None
    ):

        self.llm_player = llm_player

        # Use default scenarios if none provided
        if scenarios is None:
            scenarios = self._create_default_scenarios()

        self.scenarios = scenarios

    def _create_default_scenarios(self) -> List[Dict[str, Any]]:
        """Create default set of dictator game scenarios."""
        scenarios = []

        # Standard endowments to test
        endowments = [50, 100, 200]

        for endowment in endowments:
            # Charity scenario
            scenarios.append(
                {
                    "name": f"charity_children_{endowment}",
                    "endowment": endowment,
                    "recipient_type": "a charity that helps feed hungry children",
                    "anonymity": True,
                    "context": "This is a real charity that will receive your donation.",
                }
            )

            # Anonymous person scenario
            scenarios.append(
                {
                    "name": f"anonymous_person_{endowment}",
                    "endowment": endowment,
                    "recipient_type": "another person",
                    "anonymity": True,
                    "context": "This person was randomly selected and will never know your identity.",
                }
            )

        return scenarios

    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete dictator game experiment."""
        scenario_results = []
        total_altruism = 0

        for scenario in self.scenarios:
            # Create game for this scenario
            game = DictatorGame(
                endowment=scenario.get("endowment", 100),
                recipient_type=scenario.get("recipient_type", "another person"),
                recipient_characteristics=scenario.get("recipient_characteristics", {}),
                anonymity=scenario.get("anonymity", True),
                context=scenario.get("context"),
            )

            game.add_player(self.llm_player)

            # Play the game
            result = game.play_game()
            game_result = result.game_results[0]  # Single round game

            # Extract key metrics
            altruism_ratio = game_result.metadata["altruism_ratio"]
            amount_given = game_result.metadata["amount_given"]
            endowment = game_result.metadata["endowment"]

            scenario_summary = {
                "scenario_name": scenario.get("name", "unnamed"),
                "endowment": endowment,
                "amount_given": amount_given,
                "amount_kept": endowment - amount_given,
                "altruism_ratio": altruism_ratio,
                "recipient_type": scenario.get("recipient_type"),
                "anonymity": scenario.get("anonymity"),
                "context": scenario.get("context"),
            }

            scenario_results.append(scenario_summary)
            total_altruism += altruism_ratio

            # Reset player for next scenario
            self.llm_player.reset()

        # Calculate experiment statistics
        num_scenarios = len(self.scenarios)
        average_altruism = total_altruism / num_scenarios if num_scenarios > 0 else 0

        # Calculate altruism by category
        altruism_by_recipient = {}
        for result in scenario_results:
            recipient_type = result["recipient_type"]
            if recipient_type not in altruism_by_recipient:
                altruism_by_recipient[recipient_type] = []
            altruism_by_recipient[recipient_type].append(result["altruism_ratio"])

        # Average by recipient type
        avg_altruism_by_recipient = {
            recipient_type: sum(ratios) / len(ratios)
            for recipient_type, ratios in altruism_by_recipient.items()
        }

        # Altruism by endowment
        altruism_by_endowment: Dict[str, List[float]] = {}
        for r in scenario_results:
            key = str(r["endowment"])  # stringify for JSON stability
            altruism_by_endowment.setdefault(key, []).append(r["altruism_ratio"])
        avg_altruism_by_endowment = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in altruism_by_endowment.items()
        }

        # Altruism by anonymity
        altruism_by_anonymity: Dict[str, List[float]] = {"True": [], "False": []}
        for r in scenario_results:
            altruism_by_anonymity["True" if r["anonymity"] else "False"].append(
                r["altruism_ratio"]
            )
        avg_altruism_by_anonymity = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in altruism_by_anonymity.items()
        }

        # Distributional statistics
        ratios = [r["altruism_ratio"] for r in scenario_results]
        if ratios:
            sorted_ratios = sorted(ratios)
            n = len(sorted_ratios)
            def pct(p):
                idx = max(0, min(n - 1, int(round(p * (n - 1)))))
                return sorted_ratios[idx]
            median = pct(0.5)
            p25 = pct(0.25)
            p75 = pct(0.75)
        else:
            median = p25 = p75 = 0.0

        # Correlation between endowment and altruism ratio (Pearson)
        xs = [float(r["endowment"]) for r in scenario_results]
        ys = ratios
        if len(xs) >= 2 and len(ys) >= 2:
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den_x = sum((x - mean_x) ** 2 for x in xs)
            den_y = sum((y - mean_y) ** 2 for y in ys)
            denom = (den_x * den_y) ** 0.5
            endowment_altruism_corr = num / denom if denom > 0 else 0.0
        else:
            endowment_altruism_corr = 0.0

        return {
            "player": self.llm_player.name,
            "model": self.llm_player.model_name,
            "average_altruism_ratio": average_altruism,
            "median_altruism_ratio": median,
            "num_scenarios": num_scenarios,
            "altruism_by_recipient_type": avg_altruism_by_recipient,
            "altruism_by_endowment": avg_altruism_by_endowment,
            "altruism_by_anonymity": avg_altruism_by_anonymity,
            "altruism_distribution": {"p25": p25, "p50": median, "p75": p75},
            "endowment_altruism_correlation": endowment_altruism_corr,
            "scenario_results": scenario_results,
            "experiment_config": {
                "scenarios": self.scenarios,
                "use_reasoning": self.llm_player.use_reasoning,
            },
        }


def run_dictator_experiment(
    model_name: str = "gpt-3.5-turbo",
    player_name: str = None,
    use_reasoning: bool = False,
    scenarios: List[Dict[str, Any]] = None,
    custom_scenarios: List[str] = None,
) -> Dict[str, Any]:
    """Run a complete Dictator Game experiment.

    Args:
        model_name: The model to use
        player_name: Name for the player (defaults to model name)
        use_reasoning: Whether to enable reasoning mode
        scenarios: Custom scenarios to run
        custom_scenarios: List of preset scenario names to use

    Returns:
        Experiment results dictionary
    """
    if player_name is None:
        player_name = f"Player_{model_name.replace('-', '_')}"

    # Configure model client using provider settings
    cfg = get_model_config(model_name)
    provider = cfg.provider.lower() if cfg and cfg.provider else "openai"

    if provider == "anthropic":
        lm = dspy.Anthropic(
            model=cfg.name,
            max_tokens=cfg.max_tokens,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=cfg.temperature,
        )
    else:
        lm = dspy.OpenAI(
            model=cfg.name,
            max_tokens=cfg.max_tokens,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=cfg.temperature,
        )
    dspy.configure(lm=lm)

    # Clear any existing cache
    try:
        dspy.settings.rm.clear_cache()
    except:
        pass

    # Create model-controlled player
    llm_player = DictatorPlayer(
        name=player_name, model_name=model_name, use_reasoning=use_reasoning
    )

    # Handle preset scenarios
    if custom_scenarios:
        scenarios = []
        endowments = [50, 100, 200]  # Multiple endowments for variety

        for scenario_name in custom_scenarios:
            if scenario_name in DICTATOR_SCENARIOS:
                scenario_config = DICTATOR_SCENARIOS[scenario_name].copy()
                for endowment in endowments:
                    scenario_config["name"] = f"{scenario_name}_{endowment}"
                    scenario_config["endowment"] = endowment
                    scenarios.append(scenario_config.copy())

    # Run experiment
    experiment = DictatorExperiment(llm_player=llm_player, scenarios=scenarios)

    return experiment.run_experiment()
