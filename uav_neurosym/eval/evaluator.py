import math
from typing import List, Dict, Any
from uav_neurosym.schema import FlightScenario
from uav_neurosym.agents.baseline_agent import BaselineAgent
from uav_neurosym.agents.neuro_symbolic_agent import NeuroSymbolicAgent


class BenchmarkEvaluator:
    """
    Evaluates and compares Baseline LLM vs. Neuro-Symbolic Agent across UAVBench metrics:
    Accuracy, Zero-Violation Rate, Standard Deviation, and Balanced Style Score (BSS).
    """

    def evaluate_suite(
        self,
        scenarios: List[FlightScenario],
        baseline_agent: BaselineAgent,
        neuro_agent: NeuroSymbolicAgent
    ) -> Dict[str, Any]:
        
        baseline_results = []
        neuro_results = []

        for sc in scenarios:
            # 1. Run Baseline Agent
            b_prop, b_val = baseline_agent.run(sc)
            baseline_results.append({
                "scenario_id": sc.scenario_id,
                "title": sc.title,
                "safety_tag": sc.safety_tag,
                "is_valid": b_val.is_valid,
                "violation": b_val.violation_category,
                "numerical_error": b_val.numerical_error,
                "rationale": b_prop.rationale
            })

            # 2. Run Neuro-Symbolic Agent
            n_prop, n_val, n_history = neuro_agent.run(sc)
            neuro_results.append({
                "scenario_id": sc.scenario_id,
                "title": sc.title,
                "safety_tag": sc.safety_tag,
                "is_valid": n_val.is_valid,
                "violation": n_val.violation_category,
                "numerical_error": n_val.numerical_error,
                "attempts_required": len(n_history),
                "rationale": n_prop.rationale,
                "history": n_history
            })

        # Calculate comparative metrics
        b_metrics = self._compute_metrics(baseline_results)
        n_metrics = self._compute_metrics(neuro_results)

        return {
            "baseline": {
                "metrics": b_metrics,
                "detailed_runs": baseline_results
            },
            "neuro_symbolic": {
                "metrics": n_metrics,
                "detailed_runs": neuro_results
            },
            "improvement": {
                "accuracy_gain_pct": round(n_metrics["accuracy_pct"] - b_metrics["accuracy_pct"], 2),
                "bss_gain": round(n_metrics["bss"] - b_metrics["bss"], 4)
            }
        }

    def _compute_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(runs)
        if total == 0:
            return {}

        valid_count = sum(1 for r in runs if r["is_valid"])
        accuracy_pct = (valid_count / total) * 100.0

        # Group by safety tag / category for per-style accuracy
        categories: Dict[str, List[bool]] = {}
        for r in runs:
            cat = r["safety_tag"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["is_valid"])

        per_style_accuracies = [
            sum(1 for v in vals if v) / len(vals)
            for vals in categories.values()
        ]

        mean_acc = sum(per_style_accuracies) / max(len(per_style_accuracies), 1)

        # Standard Deviation across styles (Eq. 28)
        if len(per_style_accuracies) > 1:
            variance = sum((a - mean_acc) ** 2 for a in per_style_accuracies) / len(per_style_accuracies)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        # Balanced Style Score (BSS) (Eq. 29)
        # BSS = (Geometric Mean of per-style acc) * (1 - std_dev / mean_acc)
        eps = 1e-6
        geom_mean = 1.0
        for acc in per_style_accuracies:
            geom_mean *= (acc + eps)
        geom_mean = math.pow(geom_mean, 1.0 / max(len(per_style_accuracies), 1))

        penalty = 1.0 - (std_dev / (mean_acc + eps))
        bss = max(geom_mean * max(penalty, 0.0), 0.0)

        return {
            "total_scenarios": total,
            "passed_scenarios": valid_count,
            "accuracy_pct": round(accuracy_pct, 2),
            "mean_accuracy": round(mean_acc * 100.0, 2),
            "std_dev_pct": round(std_dev * 100.0, 2),
            "bss": round(bss, 4)
        }
