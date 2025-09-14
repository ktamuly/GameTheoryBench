"""
Axelrod tournament implementation.
Iterated Prisoner’s Dilemma with baseline strategies and a model-controlled player.
"""
import os
import random
from typing import Dict, List, Any, Optional
import dspy
from ..core.game import Game, GameResult, TournamentResult, GameState, Player
from ..core.player import StrategyPlayer, create_strategy_player, ModelPlayer
from ..signatures.axelrod_sigs import (
    PrisonersDilemmaModule,
    format_axelrod_context,
    format_axelrod_history,
    format_strategy_context,
)
from ..utils.config import get_model_config


# Payoff matrix constants
PAYOFFS = {
    ("COOPERATE", "COOPERATE"): (3, 3),  # Reward
    ("COOPERATE", "DEFECT"): (0, 5),  # Sucker's payoff, Temptation
    ("DEFECT", "COOPERATE"): (5, 0),  # Temptation, Sucker's payoff
    ("DEFECT", "DEFECT"): (1, 1),  # Punishment
}


class AxelrodPlayer(ModelPlayer):
    """Model-controlled player for the tournament."""

    def __init__(
        self, name: str, model_name: str = "gpt-3.5-turbo", use_reasoning: bool = False
    ):
        super().__init__(name, model_name)
        self.use_reasoning = use_reasoning
        self.pd_module = PrisonersDilemmaModule(use_reasoning=use_reasoning)

    def make_move(self, game_state: GameState, game_config: Dict[str, Any]) -> str:
        """Select an action in the current round."""

        # Prepare decision context
        round_num = game_state.round_number + 1
        total_rounds = game_config.get("rounds_per_match", 50)
        context = format_axelrod_context(round_num, total_rounds)
        history = format_axelrod_history(game_state.history, self.name)
        strategy_context = format_strategy_context()

        # Invoke the decision module
        result = self.pd_module(
            context=context, history=history, strategy_context=strategy_context
        )

        return result.move


class PrisonersDilemmaGame(Game):
    """Two-player repeated decision game."""

    def __init__(self, rounds_per_match: int = 50, config: Dict[str, Any] = None):
        super().__init__(config)
        self.rounds_per_match = rounds_per_match
        self.config["rounds_per_match"] = rounds_per_match
        self.players = []

    def add_player(self, player: Player):
        """Add a player."""
        if len(self.players) >= 2:
            raise ValueError("Prisoner's Dilemma only supports 2 players")
        self.players.append(player)

        # Set player-specific config
        if isinstance(player, (ModelPlayer, StrategyPlayer)):
            self.config[f"player_name"] = player.name

    def play_round(self) -> GameResult:
        """Play a single round."""
        if len(self.players) != 2:
            raise ValueError("Need exactly 2 players for Prisoner's Dilemma")

        # Get moves from both players
        moves = {}
        for player in self.players:
            player_config = self.config.copy()
            player_config["player_name"] = player.name
            if len(self.players) == 2:
                opponent = [p for p in self.players if p != player][0]
                player_config["opponent_name"] = opponent.name

            move = player.make_move(self.state, player_config)
            moves[player.name] = move

        # Calculate scores based on payoff matrix
        player_names = list(moves.keys())
        move1, move2 = moves[player_names[0]], moves[player_names[1]]
        score1, score2 = PAYOFFS.get((move1, move2), (0, 0))

        scores = {player_names[0]: score1, player_names[1]: score2}

        # Update game state
        self.state.add_round(moves, scores)

        return GameResult(
            player_scores=scores,
            moves=moves,
            metadata={"round": self.state.round_number - 1},
            is_terminal=self.is_terminal(),
        )

    def is_terminal(self) -> bool:
        """Return True when the maximum number of rounds is reached."""
        return self.state.round_number >= self.rounds_per_match


class AxelrodTournament:
    """Tournament between a model-controlled player and multiple baseline strategies."""

    def __init__(
        self,
        llm_player: AxelrodPlayer,
        strategy_names: List[str] = None,
        rounds_per_match: int = 50,
        tournament_seed: int = None,
    ):

        self.llm_player = llm_player
        self.rounds_per_match = rounds_per_match

        # Default strategies if none provided
        if strategy_names is None:
            strategy_names = [
                "always_cooperate",
                "always_defect",
                "tit_for_tat",
                "grudger",
                "random",
                "tit_for_two_tats",
                "generous_tit_for_tat",
                "pavlov",
                "suspicious_tit_for_tat",
            ]

        self.strategy_names = strategy_names
        self.tournament_seed = tournament_seed

        # Set up random seed for reproducibility
        if tournament_seed is not None:
            random.seed(tournament_seed)

    def run_tournament(self) -> Dict[str, Any]:
        """Run the complete tournament."""
        tournament_results: List[Dict[str, Any]] = []
        total_llm_score = 0
        total_strategy_score = 0
        total_rounds = 0

        # Behavioral metrics accumulators
        outcome_counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
        llm_coop_total = 0
        opp_coop_total = 0
        retaliation_num = 0
        retaliation_den = 0
        forgiveness_num = 0
        forgiveness_den = 0
        switch_count = 0
        switch_den = 0
        niceness_count = 0
        social_welfare_total = 0
        per_opponent_metrics: Dict[str, Dict[str, Any]] = {}

        for i, strategy_name in enumerate(self.strategy_names):
            # Create strategy player with a neutral name to avoid revealing
            # the underlying strategy through the identifier.
            strategy_player = create_strategy_player(
                strategy_name, player_name=f"Opponent_{i + 1}"
            )

            # Create and run the game
            game = PrisonersDilemmaGame(self.rounds_per_match)
            game.add_player(self.llm_player)
            game.add_player(strategy_player)

            # Play the match
            match_result = game.play_game()

            # Record results
            llm_score = match_result.total_scores[self.llm_player.name]
            strategy_score = match_result.total_scores[strategy_player.name]

            match_summary = {
                "opponent": strategy_name,
                "opponent_number": i + 1,
                "llm_score": llm_score,
                "strategy_score": strategy_score,
                "rounds_played": len(match_result.game_results),
            }

            # Compute per-round behavioral data for this match
            rounds = match_result.game_results
            if rounds:
                opp_name = strategy_player.name
                llm_coop_match = 0
                mutual_coop_match = 0

                for r_index, r in enumerate(rounds):
                    moves = r.moves
                    scores = r.player_scores

                    llm_move = moves.get(self.llm_player.name, "")
                    opp_move = moves.get(opp_name, "")

                    # Cooperation counts
                    if llm_move == "COOPERATE":
                        llm_coop_total += 1
                        llm_coop_match += 1
                    if opp_move == "COOPERATE":
                        opp_coop_total += 1

                    # Outcome typing from LLM perspective
                    if llm_move == "COOPERATE" and opp_move == "COOPERATE":
                        outcome_counts["CC"] += 1
                        mutual_coop_match += 1
                    elif llm_move == "COOPERATE" and opp_move == "DEFECT":
                        outcome_counts["CD"] += 1
                    elif llm_move == "DEFECT" and opp_move == "COOPERATE":
                        outcome_counts["DC"] += 1
                    elif llm_move == "DEFECT" and opp_move == "DEFECT":
                        outcome_counts["DD"] += 1

                    # Niceness: cooperate on first move vs each opponent
                    if r_index == 0 and llm_move == "COOPERATE":
                        niceness_count += 1

                    # Retaliation and forgiveness require previous round
                    if r_index > 0:
                        prev_moves = rounds[r_index - 1].moves
                        prev_llm = prev_moves.get(self.llm_player.name, "")
                        prev_opp = prev_moves.get(opp_name, "")

                        # Retaliation: P(LLM defects at t | Opp defected at t-1)
                        if prev_opp == "DEFECT":
                            retaliation_den += 1
                            if llm_move == "DEFECT":
                                retaliation_num += 1

                        # Forgiveness: P(LLM cooperates at t | LLM defected at t-1 and Opp cooperated at t-1)
                        if prev_llm == "DEFECT" and prev_opp == "COOPERATE":
                            forgiveness_den += 1
                            if llm_move == "COOPERATE":
                                forgiveness_num += 1

                        # Stability via switch rate of LLM
                        if prev_llm in ("COOPERATE", "DEFECT") and llm_move in ("COOPERATE", "DEFECT"):
                            switch_den += 1
                            if llm_move != prev_llm:
                                switch_count += 1

                    # Social welfare
                    social_welfare_total += scores.get(self.llm_player.name, 0) + scores.get(opp_name, 0)

                per_opponent_metrics[strategy_name] = {
                    "llm_cooperation_rate": llm_coop_match / len(rounds),
                    "mutual_cooperation_rate": mutual_coop_match / len(rounds),
                }

            tournament_results.append(match_summary)
            total_llm_score += llm_score
            total_strategy_score += strategy_score
            total_rounds += len(match_result.game_results)

            # Reset players for next match
            self.llm_player.reset()
            strategy_player.reset()

        # Calculate tournament statistics
        num_opponents = len(self.strategy_names)
        avg_score_per_opponent = (
            total_llm_score / num_opponents if num_opponents > 0 else 0
        )
        avg_score_per_round = total_llm_score / total_rounds if total_rounds > 0 else 0

        # Theoretical maximum (all temptation payoffs)
        theoretical_max = total_rounds * 5
        performance_ratio = (
            total_llm_score / theoretical_max if theoretical_max > 0 else 0
        )

        # Aggregate behavioral metrics
        llm_coop_rate = llm_coop_total / total_rounds if total_rounds > 0 else 0.0
        opp_coop_rate = opp_coop_total / total_rounds if total_rounds > 0 else 0.0
        niceness_rate = niceness_count / num_opponents if num_opponents > 0 else 0.0
        retaliation_rate = (
            retaliation_num / retaliation_den if retaliation_den > 0 else 0.0
        )
        forgiveness_rate = (
            forgiveness_num / forgiveness_den if forgiveness_den > 0 else 0.0
        )
        switch_rate = switch_count / switch_den if switch_den > 0 else 0.0
        welfare_per_round = social_welfare_total / total_rounds if total_rounds > 0 else 0.0

        return {
            "player": self.llm_player.name,
            "model": self.llm_player.model_name,
            "total_score": total_llm_score,
            "total_opponent_score": total_strategy_score,
            "total_rounds": total_rounds,
            "num_opponents": num_opponents,
            "average_score_per_opponent": avg_score_per_opponent,
            "average_score_per_round": avg_score_per_round,
            "performance_ratio": performance_ratio,
            "match_results": tournament_results,
            "behavioral_metrics": {
                "llm_cooperation_rate": llm_coop_rate,
                "opponent_cooperation_rate": opp_coop_rate,
                "outcome_counts": outcome_counts,
                "niceness_rate": niceness_rate,
                "retaliation_rate": retaliation_rate,
                "forgiveness_rate": forgiveness_rate,
                "switch_rate": switch_rate,
                "social_welfare": {
                    "total": social_welfare_total,
                    "per_round_avg": welfare_per_round,
                },
                "per_opponent": per_opponent_metrics,
            },
            "tournament_config": {
                "rounds_per_match": self.rounds_per_match,
                "strategies": self.strategy_names,
                "seed": self.tournament_seed,
            },
        }


def run_axelrod_experiment(
    model_name: str = "gpt-3.5-turbo",
    player_name: str = None,
    use_reasoning: bool = False,
    strategies: List[str] = None,
    rounds_per_match: int = 50,
    seed: int = None,
) -> Dict[str, Any]:
    """Run a complete Axelrod tournament experiment.

    Args:
        model_name: The model to use
        player_name: Name for the player (defaults to model name)
        use_reasoning: Whether to enable reasoning mode
        strategies: List of strategy names to compete against
        rounds_per_match: Number of rounds per match
        seed: Random seed for reproducibility

    Returns:
        Tournament results dictionary
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
    llm_player = AxelrodPlayer(
        name=player_name, model_name=model_name, use_reasoning=use_reasoning
    )

    # Run tournament
    tournament = AxelrodTournament(
        llm_player=llm_player,
        strategy_names=strategies,
        rounds_per_match=rounds_per_match,
        tournament_seed=seed,
    )

    return tournament.run_tournament()
