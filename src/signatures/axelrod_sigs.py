"""
Signatures and helpers for a two-player repeated decision game.
"""
import dspy
from typing import List, Dict, Any


class PrisonersDilemmaMove(dspy.Signature):
    """Select an action in a two-player repeated decision game."""

    context = dspy.InputField(desc="Information about the current game situation")
    history = dspy.InputField(desc="History of previous rounds and moves")
    strategy_context = dspy.InputField(desc="Strategic context and objectives")
    move = dspy.OutputField(desc="Your move: either 'COOPERATE' or 'DEFECT'")


class PrisonersDilemmaReasoning(dspy.Signature):
    """Select an action with a brief rationale in a repeated decision game."""

    context = dspy.InputField(desc="Information about the current game situation")
    history = dspy.InputField(desc="History of previous rounds and moves")
    strategy_context = dspy.InputField(desc="Strategic context and objectives")
    reasoning = dspy.OutputField(desc="Your strategic reasoning for this move")
    move = dspy.OutputField(desc="Your move: either 'COOPERATE' or 'DEFECT'")


class AxelrodTournamentIntro(dspy.Signature):
    """Provide tournament context to the decision module."""

    tournament_info = dspy.InputField(desc="Information about the tournament structure")
    understanding = dspy.OutputField(
        desc="Confirmation of understanding the rules and objectives"
    )


class PrisonersDilemmaModule(dspy.Module):
    """Decision module for the repeated decision game."""

    def __init__(self, use_reasoning: bool = False):
        super().__init__()
        self.use_reasoning = use_reasoning

        if use_reasoning:
            self.make_move = dspy.ChainOfThought(PrisonersDilemmaReasoning)
        else:
            self.make_move = dspy.Predict(PrisonersDilemmaMove)

    def forward(self, context: str, history: str, strategy_context: str):
        """Generate a move for the Prisoner's Dilemma."""
        result = self.make_move(
            context=context, history=history, strategy_context=strategy_context
        )

        # Normalize the predicted move and ensure it is valid
        move = result.move.upper().strip()
        if "COOPERATE" in move:
            clean_move = "COOPERATE"
        elif "DEFECT" in move:
            clean_move = "DEFECT"
        else:
            # Fallback to a safe default when parsing fails
            clean_move = "COOPERATE"

        return dspy.Prediction(
            move=clean_move,
            reasoning=getattr(result, "reasoning", ""),
            raw_response=result.move,
        )


def format_axelrod_context(
    round_num: int, total_rounds: int, opponent_name: str = None
) -> str:
    """Format context information for the Axelrod tournament."""
    context = f"""You are playing a game with another player for {total_rounds} rounds.

ROUND {round_num} of {total_rounds}

In each round, you both choose COOPERATE or DEFECT simultaneously.

SCORING:
- Both COOPERATE: You get 3 points, they get 3 points
- Both DEFECT: You get 1 point, they get 1 point  
- You DEFECT, they COOPERATE: You get 5 points, they get 0 points
- You COOPERATE, they DEFECT: You get 0 points, they get 5 points

Your goal is to score as many points as possible across all {total_rounds} rounds."""

    if opponent_name:
        context += f"\n\nYour opponent: {opponent_name}"

    return context


def format_axelrod_history(game_history: List[Dict[str, Any]], player_name: str) -> str:
    """Format game history for the decision module."""
    if not game_history:
        return "This is the first round - no history available."

    history_text = "GAME HISTORY:\n"

    # Show last 5 rounds maximum to keep context manageable
    recent_history = game_history[-5:] if len(game_history) > 5 else game_history

    if len(game_history) > 5:
        history_text += f"(Showing last 5 of {len(game_history)} rounds)\n"

    for round_data in recent_history:
        round_num = round_data["round"]
        moves = round_data["moves"]
        scores = round_data["scores"]

        my_move = moves.get(player_name, "Unknown")
        my_score = scores.get(player_name, 0)

        # Find opponent move and score
        opp_move = "Unknown"
        opp_score = 0
        for name, move in moves.items():
            if name != player_name:
                opp_move = move
                opp_score = scores.get(name, 0)
                break

        history_text += (
            f"Round {round_num + 1}: You={my_move} (+{my_score}), Opponent={opp_move} (+{opp_score})\n"
        )

    # Add current scores
    total_my_score = sum(r["scores"].get(player_name, 0) for r in game_history)
    total_opp_score = sum(
        r["scores"].get(name, 0)
        for r in game_history
        for name in r["scores"]
        if name != player_name
    )

    history_text += (
        f"\nCURRENT TOTAL SCORES: You={total_my_score}, Opponent={total_opp_score}"
    )

    return history_text


def format_strategy_context(
    tournament_position: int = None, total_opponents: int = None
) -> str:
    """Format strategic context for the tournament."""
    context = """THINGS TO CONSIDER:
- You'll play the same opponent for multiple rounds
- Different opponents behave differently: some are cooperative, others competitive
- Your choices affect future rounds with the same opponent
- Look for patterns in how your opponent plays
- Focus on your total score across all rounds"""

    if tournament_position and total_opponents:
        context += f"\n\nGame {tournament_position} of {total_opponents}"

    return context
