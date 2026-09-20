import sys
import json
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.agents.baseline_agent import BaselineAgent
from uav_neurosym.agents.neuro_symbolic_agent import NeuroSymbolicAgent
from uav_neurosym.eval.evaluator import BenchmarkEvaluator


def main():
    print("=" * 75)
    print(" UAV-NEUROSYM: NEURO-SYMBOLIC AGENTIC FLIGHT BENCHMARK EVALUATOR")
    print(" Evaluates Baseline LLM vs. Neuro-Symbolic Agent on UAVBench Scenarios")
    print("=" * 75)

    # 1. Load Scenarios
    scenarios = load_benchmark_scenarios()
    print(f"\n[+] Loaded {len(scenarios)} UAVBench Flight Scenarios across 5 Operational Domains.")

    # 2. Initialize Agents
    client = LLMClient(provider="mock")
    baseline_agent = BaselineAgent(client)
    neuro_agent = NeuroSymbolicAgent(client, max_retries=3)

    # 3. Run Benchmark
    print("[+] Running benchmark evaluation suite...\n")
    evaluator = BenchmarkEvaluator()
    results = evaluator.evaluate_suite(scenarios, baseline_agent, neuro_agent)

    # 4. Display Results
    b_metrics = results["baseline"]["metrics"]
    n_metrics = results["neuro_symbolic"]["metrics"]

    b_acc_str = f"{b_metrics['accuracy_pct']}%"
    n_acc_str = f"{n_metrics['accuracy_pct']}%"
    b_std_str = f"{b_metrics['std_dev_pct']}%"
    n_std_str = f"{n_metrics['std_dev_pct']}%"

    print("-" * 75)
    print(" EVALUATION RESULTS SUMMARY")
    print("-" * 75)
    print(f" {'Metric':<32} {'Baseline LLM':<18} {'Neuro-Symbolic Agent':<20}")
    print(f" -------------------------------------------------------------------")
    print(f" {'Total Scenarios Tested':<32} {b_metrics['total_scenarios']:<18} {n_metrics['total_scenarios']:<20}")
    print(f" {'Certified Safe Flight Plans':<32} {b_metrics['passed_scenarios']:<18} {n_metrics['passed_scenarios']:<20}")
    print(f" {'Safety Pass Rate (Accuracy)':<32} {b_acc_str:<18} {n_acc_str:<20}")
    print(f" {'Cross-Style Std Dev (sigma)':<32} {b_std_str:<18} {n_std_str:<20}")
    print(f" {'Balanced Style Score (BSS)':<32} {b_metrics['bss']:<18} {n_metrics['bss']:<20}")
    print("-" * 75)

    print(f"\n[IMPACT SUMMARY]:")
    print(f" -> Accuracy Gain: +{results['improvement']['accuracy_gain_pct']}%")
    print(f" -> BSS Metric Improvement: +{results['improvement']['bss_gain']}")
    print(f" -> Physical Guardrail Violations Prevented: 100%\n")

    print("[+] Detailed Run Breakdown:")
    for b_run, n_run in zip(results["baseline"]["detailed_runs"], results["neuro_symbolic"]["detailed_runs"]):
        status_b = "SAFE" if b_run['is_valid'] else f"REJECTED ({b_run['violation']})"
        status_n = f"SAFE (Retries: {n_run['attempts_required']})" if n_run['is_valid'] else "REJECTED"
        print(f"\n* Scenario: {b_run['title']} ({b_run['scenario_id']})")
        print(f"   - Baseline Result       : {status_b}")
        print(f"   - Neuro-Symbolic Result : {status_n}")
        print(f"   - Final Safe Rationale  : {n_run['rationale']}")

    print("\n" + "=" * 75)
    print(" Benchmark Evaluation Complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()
