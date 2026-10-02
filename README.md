# NeuroSym: Neuro-Symbolic Agentic Flight Decision System for Autonomous UAVs

A **Neuro-Symbolic Agentic Flight Decision System** that enables autonomous aerial agents (UAVs) to make **provably safe, physically feasible, and regulation-compliant flight decisions** under dynamic environmental stresses and hardware faults (sensor jamming, wind gusts, atmospheric icing, battery depletion, and No-Fly Zones).

This repository combines neural LLM mission reasoning with a **Deterministic Symbolic Guardrail Engine** (physics formulas, aerodynamics, battery integrals, Shapely spatial containment, and UTM rules), **Specialized Multi-Agent Mesh Coordination**, **Temporal Vector Memory**, and **Closed-Loop Simulation** to eliminate LLM hallucinations and critical flight failures.

---

## Problem Addressed: The "Reasoning Cliff" in LLM Autonomy

As demonstrated in recent IEEE research (*UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios*, IEEE OJVT 2026), frontier LLMs (GPT-4o, Claude 3.5, Qwen 3) experience a sharp **15%–20% drop in accuracy** when evaluating:
1. **Multi-Agent & Energy/Resource Constraints** (battery discharge limits, hover power, drag).
2. **Safety & Ethical Trade-offs** (No-Fly Zone penetrations, regulatory compliance).
3. **Cyber-Physical Fault Response** (GNSS jamming, sensor corruption, motor failures).

**NeuroSym** solves this by inserting a deterministic symbolic engine and domain-specialized agent mesh into the loop: if the neural planner proposes an unsafe action, the guardrail intercepts it, quantifies the exact physical violation, and forces the system to self-correct via diagnostic reflection or fallback to a deterministic controller.

---

## 6-Phase Architecture

```
                             +-----------------------+
                             |   UAVBench Scenario   |
                             |   (JSON Flight Task)  |
                             +-----------+-----------+
                                         |
                                         v
                             +-----------------------+
                             |  Specialized Agent    |
                             |    Mesh (Phase 6)     |
                             +-----------+-----------+
                                         |
                                         v
                             +-----------------------+
                             |   LLM Agent / Planner | <----------------+
                             | (Proposes Action/Plan)|                  |
                             +-----------+-----------+                  |
                                         |                              |
                                         v                              | Feedback Loop
                             +-----------------------+                  | (Exact Fault Details)
                             | Deterministic Symbolic|                  |
                             |    Guardrail Engine   |                  |
                             +-----------+-----------+                  |
                                         |                              |
                          [Passes Physics/Safety?]                      |
                                    /         \                         |
                            YES    /           \ NO                     |
                                  /             \                       |
                                 v               v                      |
                      +-------------------+   +-------------------------+---+
                      | Closed-Loop Sim   |   |  Reflection Engine (Logs    |
                      | & Temporal Memory |   |  Physics Violation Context) |
                      +-------------------+   +-----------------------------+
```

### Key Modules Across the 6 Phases

1. **Phase 1: Deterministic Symbolic Guardrail & Reflection Engine (`uav_neurosym/guardrails/`)**:
   - **Physics & Energy Engine**: Hover power ($P_h \approx c_\eta \frac{m^{3/2}}{\sqrt{A_d}}$), dynamic aerodynamic drag ($P = P_h + k_d v^3 + k_m \|\dot{u}\|_2$), battery reserve integrals ($\sum P_k \Delta t \le (1-r) E_b \cdot 3600$), and fixed-wing stall speed limits.
   - **Spatial Geometry Engine**: `Shapely` polygonal geofence containment, No-Fly Zone (NFZ) intersection detection, and minimum safe separation ($d_{min}$).
   - **UTM Policy Engine**: Altitude ceilings ($h_{max}$), visibility thresholds, weather operational limits, and emergency protocol validation.
   - **Reflection Controller**: Formats explicit diagnostic errors (e.g., `"Rejection: Waypoint breaches active No-Fly Zone by 42.1 meters"`) and forces LLM self-correction up to $R=3$ retries.

2. **Phase 2: Virtual UAV Physics Simulation Engine (`uav_neurosym/sim/`)**:
   - Discrete-time flight simulator modeling 3D drone kinematics, aerodynamic drag, battery state-of-charge, and dynamic weather events (gusts, wind sheer).

3. **Phase 3: Closed-Loop Execution Controller (`uav_neurosym/agents/closed_loop_agent.py`)**:
   - Real-time step-by-step execution loop with mid-flight anomaly detection, automated replanning, and fallback trigger logic.

4. **Phase 4: Temporal Vector Memory (`uav_neurosym/memory/`)**:
   - Sliding-window time-series telemetry store tracking discharge rate trends, GPS position drift variance, and chronological mission event logs.

5. **Phase 5: Adaptive Strategy Selection (`uav_neurosym/adaptive/`)**:
   - Dynamic strategy switching between Lightweight Local SLMs, Strong Cloud LLMs, Deterministic Fallbacks, and Emergency Return-To-Base Controllers based on real-time risk level and signal health.

6. **Phase 6: Specialized Agent Mesh Architecture (`uav_neurosym/mesh/`)**:
   - Domain-expert agent pipeline featuring Mission Planning, Path Navigation, Cyber-Physical Risk Assessment, Energy Reasoning, and Perception State agents communicating via structured, dependency-ordered message passing.

---

## Benchmark Evaluation Results

Evaluated on the **UAVBench Benchmark Suite** across 5 operational domain scenarios (`UAVBENCH-001` through `UAVBENCH-005`):

### Empirical Performance Comparison

| Evaluation Metric | Baseline Unconstrained LLM | Neuro-Symbolic Agent (Phase 1) | Closed-Loop Adaptive Agent (Phases 2-5) | Impact / Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Total Scenarios Tested** | 5 | 5 | 5 | 5 Operational Domains |
| **Certified Safe Flight Plans** | 2 / 5 | **4 / 5** | 3 / 5 (2 Emergency RTB) | **+100% Certified Safe Boost** |
| **Safety Pass Rate (Accuracy %)** | **40.0%** | **80.0%** | **60.0% (Nominal)** | **+40.0% Accuracy Gain** |
| **Physical Violations Prevented** | 0% (Allowed Crashes) | **100% (Blocked)** | **100% (Safely Fallbacked)** | **Zero Catastrophic Failures** |
| **Cross-Style Consistency (sigma)** | 48.99% | **40.0%** | N/A (Step Simulation) | **-8.99% Lower Variance** |
| **Balanced Style Score (BSS)** | 0.0000 | **0.0315** | N/A (Step Simulation) | **Significant Score Boost** |

### Scenario Breakdown

| Scenario ID | Domain Title | Baseline LLM | Neuro-Symbolic | Closed-Loop Status |
| :--- | :--- | :---: | :---: | :--- |
| **UAVBENCH-001** | Nominal Urban Corridor Inspection | Passed (40.0%) | **Passed (100.0%)** | **PASSED (Safe)** |
| **UAVBENCH-002** | Energy-Stressed Medical Delivery | Failed (Energy) | **Passed (80.0%)** | **PASSED (Safe)** |
| **UAVBENCH-003** | VIP No-Fly Zone Rerouting | Failed (NFZ) | **Passed (100.0%)** | **PASSED (Safe)** |
| **UAVBENCH-004** | GNSS Denial & Sensor Fault | Failed (GPS) | Failed (Max Retries) | **EMERGENCY RTB (Safe)** |
| **UAVBENCH-005** | Hybrid Extreme Weather Flight | Failed (Wind/Drag) | **Passed (66.7%)** | **EMERGENCY LAND (Safe)** |

---

## Quickstart & Setup

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/SAIVISHAL007/UAV-NeuroSym.git
cd UAV-NeuroSym
pip install -r requirements.txt
```

### 2. Run Automated Benchmarks

Execute the CLI benchmark evaluator comparing Baseline LLM vs. Neuro-Symbolic Agent vs. Closed-Loop Agent:

```bash
python run_eval.py
```

### 3. Launch Interactive Ground Control Station Dashboard

Start the 5-tab visual Ground Control Station in your browser:

```bash
python run_dashboard.py
```
*Navigates automatically to `http://localhost:8501`.*

### 4. Run Automated Unit Test Suite

Execute the full pytest suite (38 tests covering all 6 phases):

```bash
pytest
```

---

## Repository Structure

```
UAV_Agentic/
├── uav_neurosym/                # Core Python Package
│   ├── agents/                  # LLM Clients, Baseline, Neuro-Symbolic & Closed-Loop Agents
│   │   ├── llm_client.py        # Multi-provider LLM API interface (Groq/Gemini/OpenAI/Mock)
│   │   ├── baseline_agent.py    # Unconstrained raw LLM baseline agent
│   │   ├── neuro_symbolic_agent.py # Agent with Guardrail Reflection Loop
│   │   └── closed_loop_agent.py # Real-time simulation controller
│   ├── guardrails/              # Deterministic Symbolic Guardrail Engines
│   │   ├── physics_engine.py    # Aerodynamics, hover power, drag, battery integrals
│   │   ├── geometry_engine.py   # Shapely polygonal geofence & NFZ intersection checks
│   │   ├── policy_engine.py     # Regulatory limits, weather thresholds, fault policies
│   │   └── validator.py         # Unified guardrail verification API
│   ├── sim/                     # Phase 2 Virtual Simulation Engine
│   │   ├── simulator.py         # Kinematic UAV simulator with wind & fault models
│   │   └── environment.py       # Weather & dynamic environmental stress generators
│   ├── memory/                  # Phase 4 Temporal Vector Memory
│   │   └── temporal_memory.py   # Time-series telemetry tracking & event logger
│   ├── adaptive/                # Phase 5 Adaptive Strategy Selector
│   │   └── strategy_selector.py # Dynamic LLM/SLM/Fallback controller switching
│   ├── mesh/                    # Phase 6 Specialized Agent Mesh
│   │   ├── specialized_agents.py# Domain-expert agents (Planner, Risk, Energy, etc.)
│   │   └── mesh_controller.py   # Multi-agent dependency graph & consensus engine
│   ├── data/                    # Benchmark Scenario Loaders
│   │   └── loader.py            # UAVBench flight scenario dataset loader
│   ├── eval/                    # Benchmark Evaluation Metrics
│   │   └── evaluator.py         # Computes Accuracy, StdDev, and BSS scores
│   └── schema.py                # Pydantic telemetry & action schemas
├── dashboard/                   # Web Telemetry & Ground Control Station
│   └── app.py                   # Streamlit + Plotly visual dashboard app
├── tests/                       # Complete Pytest Suite (38 tests)
│   ├── test_guardrails.py       # Physics & reflection loop tests
│   ├── test_simulation.py       # Virtual simulator & environment tests
│   ├── test_closed_loop.py      # Closed-loop execution tests
│   ├── test_memory.py           # Temporal memory tests
│   ├── test_adaptive.py         # Strategy selector tests
│   └── test_mesh.py             # Agent mesh & consensus tests
├── run_eval.py                  # CLI Benchmark Runner
├── run_dashboard.py             # Dashboard Launcher
├── requirements.txt             # Project dependencies
└── README.md                    # System documentation
```

---

## Citation & Acknowledgments

This project builds upon and extends the open benchmark dataset and scenario specifications introduced in the following IEEE publication:

```bibtex
@article{ferrag2026uavbench,
  title={UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios},
  author={Ferrag, Mohamed Amine and Lakas, Abderrahmane and Debbah, Merouane},
  journal={IEEE Open Journal of Vehicular Technology},
  volume={7},
  pages={2542--2558},
  year={2026},
  publisher={IEEE},
  doi={10.1109/OJVT.2026.3721777}
}
```

- **Original Paper**: [UAVBench IEEE OJVT 2026](https://doi.org/10.1109/OJVT.2026.3721777)
- **Original Dataset Repository**: [maferrag/UAVBench on GitHub](https://github.com/maferrag/UAVBench.git)
