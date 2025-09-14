"""
Signatures and helpers for Dictator Game.
"""
import dspy
from typing import Dict, Any


class DictatorAllocation(dspy.Signature):
    """Make an allocation decision in the Dictator Game."""

    scenario = dspy.InputField(desc="The specific dictator game scenario and context")
    endowment = dspy.InputField(desc="The total amount of money you can allocate")
    recipient_info = dspy.InputField(desc="Information about the recipient")
    amount = dspy.OutputField(desc="Amount to give to the recipient (as a number)")


class DictatorAllocationReasoning(dspy.Signature):
    """Make a reasoned allocation decision in the Dictator Game."""

    scenario = dspy.InputField(desc="The specific dictator game scenario and context")
    endowment = dspy.InputField(desc="The total amount of money you can allocate")
    recipient_info = dspy.InputField(desc="Information about the recipient")
    reasoning = dspy.OutputField(desc="Your reasoning for this allocation decision")
    amount = dspy.OutputField(desc="Amount to give to the recipient (as a number)")


class DictatorGameModule(dspy.Module):
    """Decision module for Dictator Game allocations."""

    def __init__(self, use_reasoning: bool = False):
        super().__init__()
        self.use_reasoning = use_reasoning

        if use_reasoning:
            self.make_allocation = dspy.ChainOfThought(DictatorAllocationReasoning)
        else:
            self.make_allocation = dspy.Predict(DictatorAllocation)

    def forward(self, scenario: str, endowment: str, recipient_info: str):
        """Generate an allocation decision for the Dictator Game."""
        result = self.make_allocation(
            scenario=scenario, endowment=endowment, recipient_info=recipient_info
        )

        # Extract and validate the amount
        amount_str = result.amount.strip()

        # Try to extract number from the response
        import re

        numbers = re.findall(r"\d+\.?\d*", amount_str)

        if numbers:
            try:
                amount = float(numbers[0])
                # Validate amount is within bounds
                endowment_val = float(re.findall(r"\d+\.?\d*", endowment)[0])
                amount = max(0, min(amount, endowment_val))
            except (ValueError, IndexError):
                amount = 0.0
        else:
            amount = 0.0

        return dspy.Prediction(
            amount=amount,
            reasoning=getattr(result, "reasoning", ""),
            raw_response=result.amount,
        )


def format_dictator_scenario(
    endowment: float,
    recipient_type: str = "another person",
    anonymity: bool = True,
    context: str = None,
) -> str:
    """Format a dictator game scenario."""

    scenario = f"""You have received ${endowment:.0f}.

You can keep any amount for yourself and give any amount to {recipient_type}.
The choice is entirely yours."""

    if anonymity:
        scenario += " Your decision is completely private and anonymous."

    if context:
        scenario += f" {context}"

    scenario += f"\n\nHow much would you like to give to {recipient_type}? Please respond with the dollar amount as a number only."

    return scenario


def format_recipient_info(
    recipient_type: str, characteristics: Dict[str, Any] = None
) -> str:
    """Format information about the recipient."""

    # Recipient context is provided in the scenario prompt
    return f"The recipient: {recipient_type}"


# Common dictator game scenarios
DICTATOR_SCENARIOS = {
    "another_person": {
        "recipient_type": "another person",
        "anonymity": True,
        "context": None,
    },
    "charity_children": {
        "recipient_type": "a charity that helps feed hungry children",
        "anonymity": True,
        "context": None,
    },
    "student_textbooks": {
        "recipient_type": "a student who needs money for textbooks",
        "anonymity": True,
        "context": None,
    },
    "unemployed_person": {
        "recipient_type": "someone who is unemployed and struggling financially",
        "anonymity": True,
        "context": None,
    },
    "community_member": {
        "recipient_type": "someone in your community",
        "anonymity": True,
        "context": None,
    },
    "food_bank": {
        "recipient_type": "a local food bank",
        "anonymity": True,
        "context": None,
    },
    "elderly_person": {
        "recipient_type": "an elderly person on a fixed income",
        "anonymity": True,
        "context": None,
    },
    "disaster_family": {
        "recipient_type": "a family affected by a natural disaster",
        "anonymity": True,
        "context": None,
    },
    "medical_research": {
        "recipient_type": "a medical research foundation",
        "anonymity": True,
        "context": None,
    },
    "child_education": {
        "recipient_type": "someone saving money for their child's education",
        "anonymity": True,
        "context": None,
    },
}
