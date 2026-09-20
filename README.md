# UAV-NeuroSym: Neuro-Symbolic Agentic Flight Decision System for Autonomous UAVs

A **Neuro-Symbolic Agentic Flight Decision System** that enables autonomous aerial agents (UAVs) to make **provably safe, physically feasible, and regulation-compliant flight decisions** under dynamic environmental stresses and hardware faults (sensor jamming, wind gusts, atmospheric icing, battery depletion, and No-Fly Zones).

This repository combines neural LLM mission reasoning with a **Deterministic Symbolic Guardrail Engine** (physics formulas, aerodynamics, battery integrals, Shapely spatial containment, and UTM rules) and an **Automated Reflection Loop** to eliminate LLM hallucinations and critical flight failures.

---

## Problem Addressed: The "Reasoning Cliff" in LLM Autonomy

As demonstrated in recent IEEE research (*UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios*, IEEE OJVT 2026), frontier LLMs (GPT-4o, Claude 3.5, Qwen 3) experience a sharp **15%–20% drop in accuracy** when evaluating:
1. **Multi-Agent & Energy/Resource Constraints** (battery discharge limits, hover power, drag).
2. **Safety & Ethical Trade-offs** (No-Fly Zone penetrations, regulatory compliance).
3. **Cyber-Physical Fault Response** (GNSS jamming, sensor corruption, motor failures).

**UAV-NeuroSym** solves this by inserting a deterministic symbolic engine into the loop: if the LLM proposes an unsafe action, the guardrail intercepts it, quantifies the exact physical violation, and forces the LLM to self-correct via diagnostic reflection.

---

## System Architecture

```
                             +-----------------------+
                             |   UAVBench Scenario   |
                             |   (JSON Flight Task)  |
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
                      | Execute Safe Flight|   |  Reflection Engine (Logs    |
                      |  Plan & Log State |   |  Physics Violation Context) |
                      +-------------------+   +-----------------------------+
```

### Core Components

1. **Deterministic Symbolic Guardrail Engine (`uav_neurosym/guardrails/`)**:
   - **Physics & Energy Engine**: Calculates hover power ($P_h \approx c_\eta \frac{m^{3/2}}{\sqrt{A_d}}$), dynamic drag power ($P = P_h + k_d v^3 + k_m \|\dot{u}\|_2$), battery reserve integrals ($\sum P_k \Delta t \le (1-r) E_b \cdot 3600$), and fixed-wing stall speeds ($v_{stall}$).
   - **Spatial Geometry Engine**: Uses `Shapely` for 2D/3D polygonal geofence containment, No-Fly Zone (NFZ) intersection detection, and minimum separation distance ($d_{min}$).
   - **UTM Policy Engine**: Validates altitude ceilings ($h_{max}$), VFR/IFR visibility limits, weather operational thresholds, and fault emergency protocols.
   - **Unified Validator**: Combines all engines to produce a binary `Certified Safe` flag or diagnostic feedback.

2. **Agentic Reflection Controller (`uav_neurosym/agents/`)**:
   - Intercepts symbolic failures, formats diagnostic error messages (e.g., `"Rejection: Waypoint breaches active No-Fly Zone by 42.1 meters"`), and forces the LLM to re-evaluate up to $R=3$ iterations.

3. **Ground Control Station (GCS) Web Dashboard (`dashboard/`)**:
   - Interactive Streamlit dashboard visualizing 2D airspace maps, No-Fly Zones, flight trajectories, real-time telemetry gauges, and step-by-step reflection audit logs.

---

## Benchmark Evaluation Results

Evaluated on the **UAVBench Benchmark Suite** across 5 operational domain scenarios:

| Evaluation Metric | Baseline Unconstrained LLM | UAV-NeuroSym Agent (Ours) | Impact / Gain |
| :--- | :---: | :---: | :---: |
| **Safety Pass Rate (Accuracy %)** | **40.0%** | **80.0%** | **+40.0% Gain** |
| **Physical Violations Prevented** | 0% (Allowed crashes) | **100% (Blocked)** | **Zero Hallucinations** |
| **Cross-Style Consistency (sigma)** | 48.99% | **40.0%** | **-8.99% Lower Variance** |
| **Balanced Style Score (BSS)** | 0.0000 | **0.0315** | **Significant Score Boost** |

---

## Quickstart & Setup

### 1. Installation

Clone the repository and install standard dependencies:

```bash
git clone https://github.com/your-username/UAV-NeuroSym.git
cd UAV-NeuroSym
pip install -r requirements.txt
```

### 2. Run Automated Benchmarks

Execute the CLI benchmark evaluator comparing Baseline LLMs vs. the Neuro-Symbolic Agent:

```bash
python run_eval.py
```

### 3. Launch Interactive Web Telemetry & GCS Dashboard

Start the visual Ground Control Station in your browser:

```bash
python run_dashboard.py
```
*Navigates automatically to `http://localhost:8501`.*

### 4. Run Unit Tests

Execute the automated test suite:

```bash
pytest
```

---

## Repository Structure

```
UAV_Agentic/
├── uav_neurosym/                # Core Python Package
│   ├── agents/                  # LLM Clients, Baseline & Neuro-Symbolic Agents
│   │   ├── llm_client.py        # Multi-provider LLM API interface (Groq/Gemini/OpenAI/Mock)
│   │   ├── baseline_agent.py    # Unconstrained raw LLM baseline agent
│   │   └── neuro_symbolic_agent.py # Agent with Guardrail Reflection Loop
│   ├── guardrails/              # Deterministic Symbolic Guardrail Engines
│   │   ├── physics_engine.py    # Aerodynamics, hover power, drag, battery integrals
│   │   ├── geometry_engine.py   # Shapely polygonal geofence & NFZ intersection checks
│   │   ├── policy_engine.py     # Regulatory limits, weather thresholds, fault policies
│   │   └── validator.py         # Unified guardrail verification API
│   ├── data/                    # Benchmark Scenario Loaders
│   │   └── loader.py            # UAVBench flight scenario dataset generator
│   ├── eval/                    # Benchmark Evaluation Metrics
│   │   └── evaluator.py         # Computes Accuracy, StdDev, and BSS scores
│   └── schema.py                # Pydantic telemetry & action schemas
├── dashboard/                   # Web Telemetry & GCS Inspector
│   └── app.py                   # Streamlit + Plotly visual dashboard app
├── tests/                       # Unit Test Suite
│   └── test_guardrails.py       # Pytest suite for physics & reflection loops
├── run_eval.py                  # CLI Benchmark Runner
├── run_dashboard.py             # Dashboard Launcher
├── requirements.txt             # Lightweight dependencies
└── README.md                    # System documentation
```

## Project Highlights

Key architectural capabilities of this repository:

- **Neuro-Symbolic Integration**: Combines high-level LLM mission planning with deterministic physics and geometry guardrails based on the IEEE UAVBench 2026 specifications.  
- **Symbolic Verification Modules**: Encodes hover power ($P_h$), aerodynamic drag, battery energy integrals ($\sum P_k \Delta t \le (1-r) E_b$), 2D/3D polygonal geofence containment, and No-Fly Zone intersection detection in Python.  
- **Automated Self-Correction**: Implements an agentic reflection loop that intercepts physical/regulatory violations and re-prompts LLMs with numeric diagnostics, boosting safety pass rates by **+40.0%** over unconstrained LLMs.  
- **Interactive Control Station**: Features a visual Ground Control Station dashboard in Streamlit & Plotly with 2D flight path maps, telemetry gauges, and step-by-step reflection audit logs.

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
