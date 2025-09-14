"""
Player implementations for game theory experiments.
"""
import random
from typing import Any, Dict, List, Tuple
import dspy
from .game import Player, GameState


class StrategyPlayer(Player):
    """A player that uses hardcoded strategies."""

    def __init__(self, name: str, strategy_func: callable):
        super().__init__(name)
        self.strategy_func = strategy_func

    def make_move(self, game_state: GameState, game_config: Dict[str, Any]) -> Any:
        return self.strategy_func(game_state, game_config)


class ModelPlayer(Player):
    """A programmatic player backed by a language model."""

    def __init__(
        self, name: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7
    ):
        super().__init__(name)
        self.model_name = model_name
        self.temperature = temperature
        self._setup_dspy()

    def _setup_dspy(self):
        """Initialize the model client for decision making."""
        # This will be configured per game type in the specific implementations
        pass

    def make_move(self, game_state: GameState, game_config: Dict[str, Any]) -> Any:
        """Override in specific game implementations."""
        raise NotImplementedError("ModelPlayer must be subclassed for specific games")


# Classic Axelrod Tournament Strategies
class AxelrodStrategies:
    """Collection of classic strategies from Axelrod's tournament."""

    @staticmethod
    def always_cooperate(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Always cooperate."""
        return "COOPERATE"

    @staticmethod
    def always_defect(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Always defect."""
        return "DEFECT"

    @staticmethod
    def tit_for_tat(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Cooperate first, then copy opponent's last move."""
        if not game_state.history:
            return "COOPERATE"
        # Assume two-player game where opponent is the other player
        last_round = game_state.history[-1]
        opponent_moves = [
            move
            for name, move in last_round["moves"].items()
            if name != game_config.get("player_name")
        ]
        if opponent_moves:
            return opponent_moves[0]
        return "COOPERATE"

    @staticmethod
    def tit_for_two_tats(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Defect only after two consecutive opponent defections."""
        if len(game_state.history) < 2:
            return "COOPERATE"

        # Get last two opponent moves
        recent_history = game_state.history[-2:]
        opponent_moves = []

        for round_data in recent_history:
            round_opponent_moves = [
                move
                for name, move in round_data["moves"].items()
                if name != game_config.get("player_name")
            ]
            if round_opponent_moves:
                opponent_moves.append(round_opponent_moves[0])

        # Defect only if opponent defected in last two rounds
        if len(opponent_moves) >= 2 and all(
            move == "DEFECT" for move in opponent_moves
        ):
            return "DEFECT"
        return "COOPERATE"

    @staticmethod
    def generous_tit_for_tat(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Tit-for-tat with 10% forgiveness."""
        if not game_state.history:
            return "COOPERATE"

        last_round = game_state.history[-1]
        opponent_moves = [
            move
            for name, move in last_round["moves"].items()
            if name != game_config.get("player_name")
        ]

        if opponent_moves and opponent_moves[0] == "DEFECT":
            # 10% chance to forgive
            if random.random() < 0.1:
                return "COOPERATE"
            return "DEFECT"
        return "COOPERATE"

    @staticmethod
    def grudger(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Cooperate until opponent defects once, then always defect."""
        for round_data in game_state.history:
            opponent_moves = [
                move
                for name, move in round_data["moves"].items()
                if name != game_config.get("player_name")
            ]
            if opponent_moves and opponent_moves[0] == "DEFECT":
                return "DEFECT"
        return "COOPERATE"

    @staticmethod
    def pavlov(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Win-stay, lose-shift based on payoff matrix."""
        if not game_state.history:
            return "COOPERATE"

        last_round = game_state.history[-1]
        my_name = game_config.get("player_name")
        my_last_move = last_round["moves"].get(my_name)
        my_last_score = last_round["scores"].get(my_name, 0)

        # Simple heuristic: if got 3+ points (cooperation reward or temptation),
        # repeat move; otherwise switch
        if my_last_score >= 3:  # Won (R=3 or T=5)
            return my_last_move
        else:  # Lost (S=0 or P=1)
            return "DEFECT" if my_last_move == "COOPERATE" else "COOPERATE"

    @staticmethod
    def random_player(game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Randomly cooperate or defect."""
        return random.choice(["COOPERATE", "DEFECT"])

    @staticmethod
    def suspicious_tit_for_tat(
        game_state: GameState, game_config: Dict[str, Any]
    ) -> str:
        """Like tit-for-tat but starts with defection."""
        if not game_state.history:
            return "DEFECT"

        last_round = game_state.history[-1]
        opponent_moves = [
            move
            for name, move in last_round["moves"].items()
            if name != game_config.get("player_name")
        ]
        if opponent_moves:
            return opponent_moves[0]
        return "COOPERATE"


# Strategy factory
def create_strategy_player(
    strategy_name: str, player_name: str = None
) -> StrategyPlayer:
    """Create a strategy player by name."""
    if player_name is None:
        player_name = f"Strategy_{strategy_name}"

    strategy_map = {
        "always_cooperate": AxelrodStrategies.always_cooperate,
        "always_defect": AxelrodStrategies.always_defect,
        "tit_for_tat": AxelrodStrategies.tit_for_tat,
        "tit_for_two_tats": AxelrodStrategies.tit_for_two_tats,
        "generous_tit_for_tat": AxelrodStrategies.generous_tit_for_tat,
        "grudger": AxelrodStrategies.grudger,
        "pavlov": AxelrodStrategies.pavlov,
        "random": AxelrodStrategies.random_player,
        "suspicious_tit_for_tat": AxelrodStrategies.suspicious_tit_for_tat,
        "friedman": AxelrodStrategies.grudger,  # Alias for grudger
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return StrategyPlayer(player_name, strategy_map[strategy_name])
