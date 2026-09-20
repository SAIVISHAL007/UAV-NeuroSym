from uav_neurosym.schema import FlightScenario, ActionProposal, GuardrailValidationResult
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator


class BaselineAgent:
    """
    Standard Baseline LLM Flight Agent (One-Shot Unconstrained).
    Directly proposes a flight plan without symbolic verification or reflection loops,
    emulating the unconstrained baseline models evaluated in UAVBench.
    """

    def __init__(self, llm_client: LLMClient):
        self.client = llm_client
        self.validator = SymbolicGuardrailValidator()

    def run(self, scenario: FlightScenario) -> tuple[ActionProposal, GuardrailValidationResult]:
        # Generate one-shot proposal
        proposal = self.client.generate_flight_plan(scenario, reflection_feedback=None, is_baseline=True)
        # Validate result (without reflection or retry)
        validation_result = self.validator.validate(scenario, proposal)
        return proposal, validation_result
