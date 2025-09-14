"""
Base game theory classes and interfaces.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class GameResult:
    """Result of a single game interaction."""

    player_scores: Dict[str, float]
    moves: Dict[str, Any]
    metadata: Dict[str, Any]
    is_terminal: bool = False


@dataclass
class TournamentResult:
    """Result of a complete tournament."""

    total_scores: Dict[str, float]
    game_results: List[GameResult]
    metadata: Dict[str, Any]
    statistics: Dict[str, Any]


class GameState:
    """Represents the current state of a game."""

    def __init__(self):
        self.round_number = 0
        self.history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    def add_round(self, moves: Dict[str, Any], scores: Dict[str, float]):
        """Add a round to the game history."""
        self.history.append(
            {"round": self.round_number, "moves": moves.copy(), "scores": scores.copy()}
        )
        self.round_number += 1


class Player(ABC):
    """Abstract base class for all players."""

    def __init__(self, name: str):
        self.name = name
        self.total_score = 0.0

    @abstractmethod
    def make_move(self, game_state: GameState, game_config: Dict[str, Any]) -> Any:
        """Make a move given the current game state."""
        pass

    def reset(self):
        """Reset player state for a new game."""
        self.total_score = 0.0


class Game(ABC):
    """Abstract base class for game theory games."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state = GameState()
        self.players: List[Player] = []

    @abstractmethod
    def add_player(self, player: Player):
        """Add a player to the game."""
        pass

    @abstractmethod
    def play_round(self) -> GameResult:
        """Play a single round of the game."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        pass

    def play_game(self) -> TournamentResult:
        """Play a complete game until terminal state."""
        game_results = []

        while not self.is_terminal():
            result = self.play_round()
            game_results.append(result)

            # Update player scores
            for player_name, score in result.player_scores.items():
                for player in self.players:
                    if player.name == player_name:
                        player.total_score += score

        # Calculate final scores
        total_scores = {player.name: player.total_score for player in self.players}

        return TournamentResult(
            total_scores=total_scores,
            game_results=game_results,
            metadata=self.config,
            statistics=self._calculate_statistics(game_results),
        )

    def _calculate_statistics(self, game_results: List[GameResult]) -> Dict[str, Any]:
        """Calculate game statistics."""
        return {
            "total_rounds": len(game_results),
            "average_scores": {
                player.name: player.total_score / len(game_results)
                if game_results
                else 0
                for player in self.players
            },
        }

    def reset(self):
        """Reset the game state."""
        self.state = GameState()
        for player in self.players:
            player.reset()
