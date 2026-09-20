from typing import List, Dict, Any
from uav_neurosym.schema import FlightScenario, ActionProposal, GuardrailValidationResult
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.guardrails.validator import SymbolicGuardrailValidator


class NeuroSymbolicAgent:
    """
    Neuro-Symbolic Agent with Automated Guardrail Reflection Loop.
    Merges Neural Reasoner (LLM) with Deterministic Symbolic Guardrails.
    Intercepts physical/regulatory violations, constructs numeric diagnostic feedback,
    and forces iterative self-correction up to max_retries iterations.
    """

    def __init__(self, llm_client: LLMClient, max_retries: int = 3):
        self.client = llm_client
        self.validator = SymbolicGuardrailValidator()
        self.max_retries = max_retries

    def run(self, scenario: FlightScenario) -> tuple[ActionProposal, GuardrailValidationResult, List[Dict[str, Any]]]:
        reflection_history: List[Dict[str, Any]] = []
        feedback: str = None
        proposal: ActionProposal = None
        validation_result: GuardrailValidationResult = None

        for attempt in range(1, self.max_retries + 1):
            # 1. Neural Generation Phase
            proposal = self.client.generate_flight_plan(scenario, reflection_feedback=feedback, is_baseline=False)

            # 2. Symbolic Verification Phase
            validation_result = self.validator.validate(scenario, proposal)

            # Record iteration log
            reflection_history.append({
                "attempt": attempt,
                "proposal": proposal.model_dump(),
                "validation": validation_result.model_dump(),
                "feedback_sent": feedback
            })

            # 3. Decision & Reflection Loop
            if validation_result.is_valid:
                # Certified safe! Return early
                return proposal, validation_result, reflection_history

            # Guardrail breached! Update feedback for next reflection iteration
            feedback = validation_result.diagnostic_message

        # Returned last attempt if max retries exceeded
        return proposal, validation_result, reflection_history
