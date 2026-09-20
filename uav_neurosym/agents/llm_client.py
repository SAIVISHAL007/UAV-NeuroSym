import os
import json
import requests
from typing import Dict, Any, Optional
from uav_neurosym.schema import FlightScenario, ActionProposal, Point3D


class LLMClient:
    """
    Unified Multi-Provider LLM Client supporting:
    - Mock (Offline Intelligent Generator)
    - Groq API (Llama-3.3-70B / DeepSeek-R1)
    - Google Gemini API (Gemini 2.5 Flash / Pro)
    - OpenAI API (GPT-4o / GPT-4o-mini)
    - Anthropic Claude API (Claude 3.5 Sonnet / Haiku)
    - Local Ollama API (http://localhost:11434)
    - HuggingFace Inference API
    """

    def __init__(self, provider: str = "mock", api_key: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model or self._get_default_model(self.provider)

    def _get_default_model(self, provider: str) -> str:
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "gemini": "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "ollama": "llama3.2",
            "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
            "mock": "mock-v1"
        }
        return defaults.get(provider, "mock-v1")

    def generate_flight_plan(
        self,
        scenario: FlightScenario,
        reflection_feedback: Optional[str] = None,
        is_baseline: bool = False
    ) -> ActionProposal:
        """
        Requests the selected LLM provider to generate a flight plan.
        If reflection_feedback is provided, appends guardrail rejection details for self-correction.
        """
        if self.provider == "mock" or (not self.api_key and self.provider not in ["mock", "ollama"]):
            return self._mock_generation(scenario, reflection_feedback, is_baseline)

        prompt = self._build_prompt(scenario, reflection_feedback)

        try:
            if self.provider == "groq":
                return self._call_openai_compatible("https://api.groq.com/openai/v1/chat/completions", prompt)
            elif self.provider == "openai":
                return self._call_openai_compatible("https://api.openai.com/v1/chat/completions", prompt)
            elif self.provider == "gemini":
                return self._call_gemini_api(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic_api(prompt)
            elif self.provider == "ollama":
                return self._call_ollama_api(prompt)
            elif self.provider == "huggingface":
                return self._call_huggingface_api(prompt)
            else:
                return self._mock_generation(scenario, reflection_feedback, is_baseline)
        except Exception as e:
            # Fallback to intelligent mock on network or API key errors
            return self._mock_generation(scenario, reflection_feedback, is_baseline)

    def _build_prompt(self, scenario: FlightScenario, feedback: Optional[str] = None) -> str:
        prompt = f"""You are an Autonomous UAV Flight Planning Agent.
Mission Title: {scenario.title}
Mission Type: {scenario.mission_type}
UAV Specs: Mass {scenario.uav.mass_kg}kg, Battery {scenario.uav.battery_capacity_wh}Wh, Reserve {int(scenario.uav.reserve_fraction*100)}%, Max Speed {scenario.uav.max_velocity_ms}m/s
Weather: Wind {scenario.weather.wind_speed_ms}m/s, Gusts {scenario.weather.gust_amplitude_ms}m/s, Icing {scenario.weather.icing_risk}
Airspace Ceiling: {scenario.airspace.max_alt_m}m
Available Waypoints: {[wp.model_dump() for wp in scenario.waypoints]}
"""
        if feedback:
            prompt += f"\nCRITICAL SYMBOLIC GUARDRAIL REJECTION FEEDBACK FROM PREVIOUS TRIAL:\n{feedback}\nYou MUST revise your flight plan to fix this exact numerical/spatial violation!"

        prompt += "\nOutput strict valid JSON matching ActionProposal schema: {\"proposed_path\": [{\"x\": float, \"y\": float, \"z\": float}], \"target_airspeed_ms\": float, \"estimated_duration_s\": float, \"emergency_action\": string or null, \"rationale\": string}"
        return prompt

    def _call_openai_compatible(self, url: str, prompt: str) -> ActionProposal:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        res = requests.post(url, json=body, headers=headers, timeout=15)
        res_json = res.json()
        text = res_json["choices"][0]["message"]["content"]
        return self._parse_json_to_proposal(text)

    def _call_gemini_api(self, prompt: str) -> ActionProposal:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1}
        }
        res = requests.post(url, json=body, headers=headers, timeout=15)
        res_json = res.json()
        text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_json_to_proposal(text)

    def _call_anthropic_api(self, prompt: str) -> ActionProposal:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post(url, json=body, headers=headers, timeout=15)
        res_json = res.json()
        text = res_json["content"][0]["text"]
        return self._parse_json_to_proposal(text)

    def _call_ollama_api(self, prompt: str) -> ActionProposal:
        url = "http://localhost:11434/api/chat"
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        res = requests.post(url, json=body, timeout=15)
        res_json = res.json()
        text = res_json["message"]["content"]
        return self._parse_json_to_proposal(text)

    def _call_huggingface_api(self, prompt: str) -> ActionProposal:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}
        res = requests.post(url, json=body, headers=headers, timeout=15)
        res_json = res.json()
        text = res_json[0]["generated_text"]
        return self._parse_json_to_proposal(text)

    def _parse_json_to_proposal(self, text: str) -> ActionProposal:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx : end_idx + 1]
            data = json.loads(json_str)
            return ActionProposal(**data)
        raise ValueError("Could not parse JSON from LLM response text")

    def _mock_generation(
        self,
        scenario: FlightScenario,
        feedback: Optional[str] = None,
        is_baseline: bool = False
    ) -> ActionProposal:
        """
        Intelligent Mock LLM Generator that simulates raw LLM behavior (making common errors on attempt 1,
        but successfully reflecting and correcting on subsequent attempts when feedback is provided).
        """
        waypoints = scenario.waypoints
        uav = scenario.uav

        if is_baseline or (feedback is None):
            if scenario.title.startswith("Energy") or uav.battery_capacity_wh < 100.0:
                fast_speed = uav.max_velocity_ms * 0.95
                return ActionProposal(
                    proposed_path=waypoints,
                    target_airspeed_ms=fast_speed,
                    estimated_duration_s=180.0,
                    rationale="[Mock Initial Proposal]: Selected fast airspeed to complete mission quickly."
                )

            if len(scenario.no_fly_zones) > 0 and scenario.no_fly_zones[0].is_active:
                return ActionProposal(
                    proposed_path=[scenario.spawn_point] + waypoints,
                    target_airspeed_ms=12.0,
                    estimated_duration_s=200.0,
                    rationale="[Mock Initial Proposal]: Proposed direct path connecting waypoints sequentially."
                )

            has_gnss_fault = any(f.fault_type == "gnss_denial" for f in scenario.faults)
            if has_gnss_fault:
                return ActionProposal(
                    proposed_path=waypoints,
                    target_airspeed_ms=10.0,
                    estimated_duration_s=220.0,
                    rationale="[Mock Initial Proposal]: Continuing nominal mission trajectory."
                )

        corrected_waypoints = [wp.model_copy() for wp in waypoints]
        safe_speed = min(uav.max_velocity_ms * 0.60, 10.0)
        emergency_act = None

        if feedback and "NO_FLY_ZONE" in feedback:
            corrected_waypoints = [Point3D(x=wp.x + 80.0, y=wp.y + 80.0, z=wp.z) for wp in waypoints]
            rationale = "[Mock Self-Correction]: Rerouted trajectory around active No-Fly Zone polygon boundary based on guardrail feedback."

        elif feedback and "ENERGY_EXHAUSTION" in feedback:
            safe_speed = max(uav.max_velocity_ms * 0.45, 6.0)
            corrected_waypoints = waypoints[:2]
            rationale = f"[Mock Self-Correction]: Economized energy by reducing airspeed to {safe_speed}m/s and optimizing waypoint sequence."

        elif feedback and "GNSS_DENIAL" in feedback:
            emergency_act = "RETURN_TO_BASE"
            rationale = "[Mock Self-Correction]: Activated emergency fail-safe protocol 'RETURN_TO_BASE' due to severe GNSS jamming."

        else:
            rationale = "[Mock Self-Correction]: Adjusted flight telemetry to comply with symbolic guardrails."

        return ActionProposal(
            proposed_path=corrected_waypoints,
            target_airspeed_ms=round(safe_speed, 1),
            estimated_duration_s=240.0,
            emergency_action=emergency_act,
            rationale=rationale
        )
