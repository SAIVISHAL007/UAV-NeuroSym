import streamlit as st
import plotly.graph_objects as go
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.agents.baseline_agent import BaselineAgent
from uav_neurosym.agents.neuro_symbolic_agent import NeuroSymbolicAgent


st.set_page_config(
    page_title="UAV-NeuroSym Ground Control Station",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3440;
    }
    .status-pass {
        color: #00e676;
        font-weight: bold;
    }
    .status-fail {
        color: #ff5252;
        font-weight: bold;
    }
    .reflection-box {
        background-color: #181c24;
        border-left: 4px solid #ff9100;
        padding: 12px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("UAV-NeuroSym: Neuro-Symbolic Agentic Flight Control Station")
    st.caption("Real-Time Closed-Loop Physics Guardrails & Reflection Inspector for Autonomous UAV Systems (UAVBench)")

    scenarios = load_benchmark_scenarios()
    scenario_dict = {f"{s.scenario_id}: {s.title}": s for s in scenarios}

    # Sidebar Controls
    st.sidebar.header("Control Panel")
    selected_name = st.sidebar.selectbox("Select Flight Scenario", list(scenario_dict.keys()))
    scenario = scenario_dict[selected_name]

    llm_provider = st.sidebar.selectbox(
        "LLM Provider Engine",
        ["mock", "groq", "gemini", "openai", "anthropic", "ollama", "huggingface"],
        index=0
    )
    api_key_input = st.sidebar.text_input("API Key (Optional for Mock)", type="password")
    max_retries = st.sidebar.slider("Max Reflection Retries (R)", 1, 5, 3)

    run_button = st.sidebar.button("Execute Autonomous Flight Plan", use_container_width=True)

    # Display Scenario Metadata Header
    st.subheader(f"Scenario Detail: {scenario.title}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mission Type", scenario.mission_type.value.upper())
    col2.metric("UAV Platform", f"{scenario.uav.uav_id} ({scenario.uav.uav_type.value})")
    col3.metric("Battery Capacity", f"{scenario.uav.battery_capacity_wh} Wh")
    col4.metric("Wind / Gusts", f"{scenario.weather.wind_speed_ms} +/- {scenario.weather.gust_amplitude_ms} m/s")
    col5.metric("Risk Level", scenario.risk_level.name)

    if run_button or "current_scenario" not in st.session_state:
        st.session_state["current_scenario"] = scenario
        client = LLMClient(provider=llm_provider, api_key=api_key_input)
        
        # 1. Run Baseline Agent
        baseline_agent = BaselineAgent(client)
        b_proposal, b_val = baseline_agent.run(scenario)

        # 2. Run Neuro-Symbolic Agent
        neuro_agent = NeuroSymbolicAgent(client, max_retries=max_retries)
        n_proposal, n_val, n_history = neuro_agent.run(scenario)

        st.session_state["b_proposal"] = b_proposal
        st.session_state["b_val"] = b_val
        st.session_state["n_proposal"] = n_proposal
        st.session_state["n_val"] = n_val
        st.session_state["n_history"] = n_history

    b_proposal = st.session_state.get("b_proposal")
    b_val = st.session_state.get("b_val")
    n_proposal = st.session_state.get("n_proposal")
    n_val = st.session_state.get("n_val")
    n_history = st.session_state.get("n_history", [])

    # Results Overview Banner
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.write("### Unconstrained Baseline LLM")
        if b_val and b_val.is_valid:
            st.success("CERTIFIED SAFE")
        else:
            st.error(f"REJECTED: {b_val.violation_category if b_val else 'Failed'}")
        st.info(f"**Rationale**: {b_proposal.rationale if b_proposal else ''}")

    with res_col2:
        st.write("### Neuro-Symbolic Agent (Ours)")
        if n_val and n_val.is_valid:
            st.success(f"CERTIFIED SAFE (Attempts: {len(n_history)})")
        else:
            st.error("REJECTED (Max Retries Exceeded)")
        st.info(f"**Rationale**: {n_proposal.rationale if n_proposal else ''}")

    with res_col3:
        st.write("### Symbolic Telemetry Metrics")
        m = n_val.metrics if n_val else {}
        st.metric("Power Draw", f"{m.get('power_watts', 0)} W")
        st.metric("Energy Consumed", f"{m.get('energy_consumed_wh', 0)} Wh", delta=f"Budget: {m.get('usable_battery_wh', 0)} Wh")

    # 2D Interactive Airspace Flight Map
    st.divider()
    st.subheader("2D Interactive Airspace Flight Map & Geofence Inspection")

    fig = go.Figure()

    # Draw Airspace Boundary
    airspace_verts = scenario.airspace.vertices + [scenario.airspace.vertices[0]]
    ax, ay = zip(*airspace_verts)
    fig.add_trace(go.Scatter(
        x=ax, y=ay, mode="lines",
        name="Authorized Geofence",
        line=dict(color="#00e676", width=2, dash="dash")
    ))

    # Draw No-Fly Zones
    for nfz in scenario.no_fly_zones:
        nfz_verts = nfz.polygon + [nfz.polygon[0]]
        nx, ny = zip(*nfz_verts)
        fig.add_trace(go.Scatter(
            x=nx, y=ny, mode="lines", fill="toself",
            name=f"NFZ: {nfz.name}",
            fillcolor="rgba(255, 82, 82, 0.3)",
            line=dict(color="#ff5252", width=2)
        ))

    # Draw Waypoints
    wpx = [wp.x for wp in scenario.waypoints]
    wpy = [wp.y for wp in scenario.waypoints]
    fig.add_trace(go.Scatter(
        x=wpx, y=wpy, mode="markers+text",
        name="Waypoints",
        text=[f"WP{i+1}" for i in range(len(scenario.waypoints))],
        textposition="top center",
        marker=dict(size=12, color="#ffd600", symbol="diamond")
    ))

    # Draw Spawn Point
    fig.add_trace(go.Scatter(
        x=[scenario.spawn_point.x], y=[scenario.spawn_point.y],
        mode="markers+text", name="Spawn Point",
        text=["SPAWN"], textposition="bottom center",
        marker=dict(size=14, color="#00b0ff", symbol="square")
    ))

    # Draw Baseline Path (Red)
    if b_proposal and b_proposal.proposed_path:
        bx = [p.x for p in b_proposal.proposed_path]
        by = [p.y for p in b_proposal.proposed_path]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines+markers",
            name="Baseline Path (Unconstrained)",
            line=dict(color="#ff5252", width=3, dash="dot")
        ))

    # Draw Neuro-Symbolic Safe Corrected Path (Green)
    if n_proposal and n_proposal.proposed_path:
        nx = [p.x for p in n_proposal.proposed_path]
        ny = [p.y for p in n_proposal.proposed_path]
        fig.add_trace(go.Scatter(
            x=nx, y=ny, mode="lines+markers",
            name="Neuro-Symbolic Safe Path",
            line=dict(color="#00e676", width=4)
        ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Ground Coordinate X (meters)",
        yaxis_title="Ground Coordinate Y (meters)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_state=True)

    # Reflection Loop Terminal Inspector
    st.divider()
    st.subheader("Agentic Self-Correction & Reflection Inspector")
    st.write("Step-by-step audit log showing how symbolic guardrail failures trigger diagnostic feedback to force self-correction:")

    for item in n_history:
        att = item["attempt"]
        val = item["validation"]
        prop = item["proposal"]
        
        with st.expander(f"Iterative Reflection Step #{att} — {'CERTIFIED SAFE' if val['is_valid'] else 'GUARDRAIL REJECTED'}", expanded=True):
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.write("**LLM Proposed Telemetry & Action:**")
                st.json({
                    "target_airspeed_ms": prop["target_airspeed_ms"],
                    "estimated_duration_s": prop["estimated_duration_s"],
                    "emergency_action": prop.get("emergency_action"),
                    "rationale": prop["rationale"]
                })
            with col_b:
                st.write("**Symbolic Guardrail Diagnostic Feedback:**")
                if val["is_valid"]:
                    st.success(val["diagnostic_message"])
                else:
                    st.error(val["diagnostic_message"])


if __name__ == "__main__":
    main()
