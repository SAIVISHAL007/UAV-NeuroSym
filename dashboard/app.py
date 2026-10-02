import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import time
from uav_neurosym.data.loader import load_benchmark_scenarios
from uav_neurosym.agents.llm_client import LLMClient
from uav_neurosym.agents.baseline_agent import BaselineAgent
from uav_neurosym.agents.neuro_symbolic_agent import NeuroSymbolicAgent
from uav_neurosym.agents.closed_loop_agent import ClosedLoopAgent
from uav_neurosym.mesh.mesh_controller import MultiAgentMeshController
from uav_neurosym.mesh.specialized_agents import AgentRole


st.set_page_config(
    page_title="NeuroSym — Ground Control Station",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium Dark Glassmorphism CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .main { background-color: #0a0e17; }
    section[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1a2035; }

    h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; }

    .stMetric {
        background: linear-gradient(135deg, #131825 0%, #1a2035 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #252d44;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00e676 0%, #00bcd4 100%);
        color: #0a0e17;
        font-weight: 700;
        font-size: 0.7rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        letter-spacing: 0.5px;
    }
    .phase-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c4dff 0%, #536dfe 100%);
        color: #fff;
        font-weight: 600;
        font-size: 0.65rem;
        padding: 2px 8px;
        border-radius: 12px;
        margin-right: 4px;
    }
    .mesh-agent-card {
        background: linear-gradient(135deg, #131825 0%, #182030 100%);
        border: 1px solid #252d44;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 6px;
    }
    .flag-chip {
        display: inline-block;
        background: rgba(255, 145, 0, 0.15);
        color: #ff9100;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 8px;
        margin: 2px;
        border: 1px solid rgba(255, 145, 0, 0.3);
    }
    .flag-critical {
        background: rgba(255, 82, 82, 0.15);
        color: #ff5252;
        border-color: rgba(255, 82, 82, 0.3);
    }
    .strategy-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px;
    }
    .strategy-slm { background: #1a237e; color: #82b1ff; }
    .strategy-llm { background: #1b5e20; color: #69f0ae; }
    .strategy-det { background: #4a148c; color: #ea80fc; }
    .strategy-emg { background: #b71c1c; color: #ff8a80; }
    .timeline-event {
        background: #131825;
        border-left: 3px solid #536dfe;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # ─── Hero Header ───
    st.markdown("""
    <div style='text-align: center; padding: 10px 0 5px 0;'>
        <h1 style='background: linear-gradient(135deg, #00e676, #00bcd4, #7c4dff);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.2rem; margin-bottom: 4px;'>
            NeuroSym — Ground Control Station
        </h1>
        <p style='color: #8899aa; font-size: 0.9rem; margin-bottom: 8px;'>
            Neuro-Symbolic Agentic Flight Decision System for Autonomous UAV Operations
        </p>
        <span class='hero-badge'>PHASE 1 Guardrails</span>
        <span class='hero-badge'>PHASE 2 Simulation</span>
        <span class='hero-badge'>PHASE 3 Closed-Loop</span>
        <span class='hero-badge'>PHASE 4 Temporal Memory</span>
        <span class='hero-badge'>PHASE 5 Adaptive AI</span>
        <span class='hero-badge'>PHASE 6 Agent Mesh</span>
    </div>
    """, unsafe_allow_html=True)

    scenarios = load_benchmark_scenarios()
    scenario_dict = {f"{s.scenario_id}: {s.title}": s for s in scenarios}

    # ─── Sidebar Controls ───
    st.sidebar.markdown("### Control Panel")
    selected_name = st.sidebar.selectbox("Flight Scenario", list(scenario_dict.keys()))
    scenario = scenario_dict[selected_name]

    st.sidebar.divider()
    llm_provider = st.sidebar.selectbox(
        "LLM Provider Engine",
        ["mock", "groq", "gemini", "openai", "anthropic", "ollama", "huggingface"],
        index=0
    )
    api_key_input = st.sidebar.text_input("API Key (Optional for Mock)", type="password")
    max_retries = st.sidebar.slider("Max Reflection Retries", 1, 5, 3)

    st.sidebar.divider()
    st.sidebar.markdown("#### Execution Mode")
    run_mode = st.sidebar.radio(
        "Select Analysis Pipeline",
        ["Full Pipeline (All 6 Phases)", "Guardrail Only (Phase 1)", "Agent Mesh Only (Phase 6)"],
        index=0
    )

    run_button = st.sidebar.button("Execute Autonomous Flight Plan", use_container_width=True, type="primary")

    # ─── Scenario Metadata Header ───
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mission Type", scenario.mission_type.value.upper())
    col2.metric("UAV Platform", f"{scenario.uav.uav_id}")
    col3.metric("Battery", f"{scenario.uav.battery_capacity_wh} Wh")
    col4.metric("Wind / Gusts", f"{scenario.weather.wind_speed_ms} ± {scenario.weather.gust_amplitude_ms}")
    col5.metric("Risk Level", scenario.risk_level.name)

    # ─── Execute ───
    if run_button:
        client = LLMClient(provider=llm_provider, api_key=api_key_input if api_key_input else None)

        with st.spinner("Executing NeuroSym pipeline across all 6 phases..."):
            progress = st.progress(0, text="Initializing...")

            # Phase 1: Baseline + Neuro-Symbolic Guardrail Agent
            progress.progress(10, text="Phase 1: Running Baseline LLM Agent...")
            baseline_agent = BaselineAgent(client)
            b_proposal, b_val = baseline_agent.run(scenario)

            progress.progress(25, text="Phase 1: Running Neuro-Symbolic Guardrail Agent...")
            neuro_agent = NeuroSymbolicAgent(client, max_retries=max_retries)
            n_proposal, n_val, n_history = neuro_agent.run(scenario)

            # Phase 6: Specialized Agent Mesh
            progress.progress(40, text="Phase 6: Executing Specialized Agent Mesh Pipeline...")
            mesh_controller = MultiAgentMeshController()
            mesh_proposal, mesh_report = mesh_controller.execute_mesh(scenario)

            # Phase 2-5: Closed-Loop Simulation (includes Sim, Memory, Adaptive)
            progress.progress(60, text="Phase 2-5: Running Closed-Loop Simulation...")
            closed_loop_agent = ClosedLoopAgent(client, dt_s=1.0, max_steps=500)
            cl_result = closed_loop_agent.run_mission(scenario)

            progress.progress(100, text="All 6 phases complete!")
            time.sleep(0.5)
            progress.empty()

        st.session_state["b_proposal"] = b_proposal
        st.session_state["b_val"] = b_val
        st.session_state["n_proposal"] = n_proposal
        st.session_state["n_val"] = n_val
        st.session_state["n_history"] = n_history
        st.session_state["mesh_proposal"] = mesh_proposal
        st.session_state["mesh_report"] = mesh_report
        st.session_state["cl_result"] = cl_result
        st.session_state["executed"] = True

    if not st.session_state.get("executed"):
        st.info("Select a scenario and click **Execute Autonomous Flight Plan** to run all 6 phases.")
        return

    # ─── Retrieve results from session state ───
    b_proposal = st.session_state["b_proposal"]
    b_val = st.session_state["b_val"]
    n_proposal = st.session_state["n_proposal"]
    n_val = st.session_state["n_val"]
    n_history = st.session_state["n_history"]
    mesh_proposal = st.session_state["mesh_proposal"]
    mesh_report = st.session_state["mesh_report"]
    cl_result = st.session_state["cl_result"]

    # ═════════════════════════════════════════════════════════════════════
    # TAB LAYOUT — One tab per major feature
    # ═════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Flight Map & Guardrails",
        "Agent Mesh Intelligence",
        "Closed-Loop Simulation",
        "Reflection Inspector",
        "Mission Report"
    ])

    # ═════════════════════════════════════════════════════════════════════
    # TAB 1: Flight Map & Guardrail Results (Phase 1)
    # ═════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<span class="phase-badge">PHASE 1</span> **Symbolic Guardrail Verification & Flight Path Comparison**', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown("##### Unconstrained Baseline LLM")
            if b_val and b_val.is_valid:
                st.success("CERTIFIED SAFE")
            else:
                st.error(f"REJECTED: {b_val.violation_category if b_val else 'Failed'}")
            st.caption(b_proposal.rationale if b_proposal else "")

        with res_col2:
            st.markdown("##### Neuro-Symbolic Agent (Ours)")
            if n_val and n_val.is_valid:
                st.success(f"CERTIFIED SAFE (Attempts: {len(n_history)})")
            else:
                st.error("REJECTED (Max Retries Exceeded)")
            st.caption(n_proposal.rationale if n_proposal else "")

        with res_col3:
            st.markdown("##### Symbolic Telemetry")
            m = n_val.metrics if n_val else {}
            st.metric("Power Draw", f"{m.get('power_watts', 0):.1f} W")
            st.metric("Energy Consumed", f"{m.get('energy_consumed_wh', 0):.1f} Wh",
                      delta=f"Budget: {m.get('usable_battery_wh', 0):.1f} Wh")

        # 2D Interactive Airspace Map
        st.markdown("---")
        fig = go.Figure()

        # Geofence
        airspace_verts = scenario.airspace.vertices + [scenario.airspace.vertices[0]]
        ax, ay = zip(*airspace_verts)
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", name="Authorized Geofence",
                                 line=dict(color="#00e676", width=2, dash="dash")))

        # No-Fly Zones
        for nfz in scenario.no_fly_zones:
            nfz_verts = nfz.polygon + [nfz.polygon[0]]
            nx_v, ny_v = zip(*nfz_verts)
            fig.add_trace(go.Scatter(x=nx_v, y=ny_v, mode="lines", fill="toself",
                                     name=f"NFZ: {nfz.name}",
                                     fillcolor="rgba(255, 82, 82, 0.25)",
                                     line=dict(color="#ff5252", width=2)))

        # Waypoints
        wpx = [wp.x for wp in scenario.waypoints]
        wpy = [wp.y for wp in scenario.waypoints]
        fig.add_trace(go.Scatter(x=wpx, y=wpy, mode="markers+text", name="Waypoints",
                                 text=[f"WP{i+1}" for i in range(len(scenario.waypoints))],
                                 textposition="top center",
                                 marker=dict(size=12, color="#ffd600", symbol="diamond")))

        # Spawn
        fig.add_trace(go.Scatter(x=[scenario.spawn_point.x], y=[scenario.spawn_point.y],
                                 mode="markers+text", name="Spawn Point",
                                 text=["SPAWN"], textposition="bottom center",
                                 marker=dict(size=14, color="#00b0ff", symbol="square")))

        # Baseline Path
        if b_proposal and b_proposal.proposed_path:
            bx = [p.x for p in b_proposal.proposed_path]
            by = [p.y for p in b_proposal.proposed_path]
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines+markers",
                                     name="Baseline Path (Unconstrained)",
                                     line=dict(color="#ff5252", width=3, dash="dot")))

        # Neuro-Symbolic Path
        if n_proposal and n_proposal.proposed_path:
            sx = [p.x for p in n_proposal.proposed_path]
            sy = [p.y for p in n_proposal.proposed_path]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines+markers",
                                     name="Neuro-Symbolic Safe Path",
                                     line=dict(color="#00e676", width=4)))

        # Mesh Agent Path
        if mesh_proposal and mesh_proposal.proposed_path:
            mx = [p.x for p in mesh_proposal.proposed_path]
            my = [p.y for p in mesh_proposal.proposed_path]
            fig.add_trace(go.Scatter(x=mx, y=my, mode="lines+markers",
                                     name="Agent Mesh Path (Phase 6)",
                                     line=dict(color="#7c4dff", width=3, dash="dashdot"),
                                     marker=dict(size=8, symbol="star")))

        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Ground Coordinate X (meters)",
            yaxis_title="Ground Coordinate Y (meters)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(size=11)),
            plot_bgcolor="#0a0e17",
            paper_bgcolor="#0a0e17",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════
    # TAB 2: Agent Mesh Intelligence (Phase 6)
    # ═════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<span class="phase-badge">PHASE 6</span> **Specialized Agent Mesh — Multi-Agent Coordination & Consensus**', unsafe_allow_html=True)

        # Mesh Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Agents Executed", mesh_report["agents_executed"])
        m2.metric("Aggregate Confidence", f"{mesh_report['aggregate_confidence']:.1%}")
        m3.metric("Advisory Flags", len(mesh_report["all_advisory_flags"]))
        m4.metric("Critical Flags", mesh_report["critical_flag_count"])

        # Advisory Flags Display
        if mesh_report["all_advisory_flags"]:
            st.markdown("##### Advisory Flags Raised")
            flags_html = ""
            for f in mesh_report["all_advisory_flags"]:
                css_class = "flag-chip flag-critical" if any(k in f for k in ["CRITICAL", "DENIED", "EMERGENCY"]) else "flag-chip"
                flags_html += f'<span class="{css_class}">{f}</span>'
            st.markdown(flags_html, unsafe_allow_html=True)

        st.markdown("---")

        # Per-Agent Analysis Cards
        st.markdown("##### Specialized Agent Outputs")

        agent_colors = {
            "perception_state": "#00bcd4",
            "mission_planner": "#ffd600",
            "navigator": "#536dfe",
            "risk_assessor": "#ff9100",
            "energy_reasoner": "#00e676"
        }

        for agent_name, summary in mesh_report["agent_summaries"].items():
            color = agent_colors.get(agent_name, "#7c4dff")
            conf_pct = summary["confidence"] * 100

            st.markdown(f"""
            <div class='mesh-agent-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='font-size: 1.05rem; font-weight: 600; color: {color};'>
                        {agent_name.replace('_', ' ').title()}
                    </span>
                    <span style='color: {"#00e676" if conf_pct >= 80 else "#ff9100" if conf_pct >= 50 else "#ff5252"};
                           font-weight: 700; font-size: 0.95rem;'>
                        {conf_pct:.0f}% confidence
                    </span>
                </div>
                <p style='color: #8899aa; font-size: 0.82rem; margin: 6px 0 4px 0;'>{summary["rationale"]}</p>
                <div class='confidence-bar' style='background: linear-gradient(90deg, {color} {conf_pct}%, #1a2035 {conf_pct}%);'></div>
            </div>
            """, unsafe_allow_html=True)

        # Mesh Proposal Summary
        st.markdown("---")
        st.markdown("##### Synthesized Mesh Consensus Proposal")
        ps = mesh_report["proposal_summary"]
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Waypoints", ps["num_waypoints"])
        pc2.metric("Target Speed", f"{ps['target_speed_ms']:.1f} m/s")
        pc3.metric("Est. Duration", f"{ps['estimated_duration_s']:.0f} s")
        pc4.metric("Emergency Action", ps["emergency_action"] or "None")

        # Mesh Execution Pipeline Visualization
        st.markdown("---")
        st.markdown("##### Mesh Execution Pipeline Log")
        for i, step in enumerate(mesh_report["execution_log"]):
            color = agent_colors.get(step["agent"], "#7c4dff")
            st.markdown(f"""
            <div class='timeline-event' style='border-left-color: {color};'>
                <strong style='color: {color};'>Step {i+1}: {step["agent"].replace('_', ' ').title()}</strong>
                <span style='color: #667788; margin-left: 12px;'>Confidence: {step["confidence"]:.0%}</span>
                <br><span style='color: #8899aa; font-size: 0.8rem;'>{step["rationale"]}</span>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # TAB 3: Closed-Loop Simulation (Phase 2-5)
    # ═════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            '<span class="phase-badge">PHASE 2</span> '
            '<span class="phase-badge">PHASE 3</span> '
            '<span class="phase-badge">PHASE 4</span> '
            '<span class="phase-badge">PHASE 5</span> '
            '**Closed-Loop Simulation — Virtual UAV, Temporal Memory & Adaptive Strategy Selection**',
            unsafe_allow_html=True
        )

        # Mission Outcome Metrics
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Mission Status",
                  "COMPLETED" if cl_result["is_completed"]
                  else "ABORTED" if cl_result["is_aborted"]
                  else "VIOLATED" if cl_result["is_violated"]
                  else "TIMEOUT")
        o2.metric("Total Steps", cl_result["total_steps"])
        o3.metric("Elapsed Time", f"{cl_result['elapsed_time_s']:.1f} s")
        o4.metric("Final Battery", f"{cl_result['final_battery_pct']:.1f}%")
        o5.metric("Replans", cl_result["replans_count"])

        # Strategies Used
        st.markdown("---")
        st.markdown("##### Adaptive Strategy Selection (Phase 5)")
        strategy_tag_map = {
            "lightweight_local_slm": ("strategy-tag strategy-slm", "Lightweight Local SLM"),
            "strong_cloud_llm": ("strategy-tag strategy-llm", "Strong Cloud LLM"),
            "deterministic_fallback": ("strategy-tag strategy-det", "Deterministic Fallback"),
            "emergency_deterministic_controller": ("strategy-tag strategy-emg", "Emergency Controller"),
        }
        strat_html = ""
        for s in cl_result["strategies_used"]:
            css, label = strategy_tag_map.get(s, ("strategy-tag", s))
            strat_html += f'<span class="{css}">{label}</span>'
        st.markdown(strat_html, unsafe_allow_html=True)

        # Telemetry Charts from step_history
        st.markdown("---")
        st.markdown("##### Real-Time Telemetry Traces (Phase 2 Simulation Engine + Phase 4 Temporal Memory)")

        step_history = cl_result.get("step_history", [])
        if step_history:
            steps = list(range(len(step_history)))
            battery_trace = [h.get("remaining_battery_wh", 0) for h in step_history]
            speed_trace = [h.get("ground_speed_ms", 0) for h in step_history]
            power_trace = [h.get("power_draw_w", 0) for h in step_history]
            alt_trace = [h.get("altitude_m", 0) for h in step_history]

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Battery Discharge Curve
                fig_batt = go.Figure()
                fig_batt.add_trace(go.Scatter(
                    x=steps, y=battery_trace, mode="lines",
                    name="Battery (Wh)",
                    line=dict(color="#00e676", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 230, 118, 0.08)"
                ))
                fig_batt.update_layout(
                    title="Battery Discharge Curve",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis_title="Step", yaxis_title="Battery (Wh)",
                    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17"
                )
                st.plotly_chart(fig_batt, use_container_width=True)

            with chart_col2:
                # Ground Speed
                fig_speed = go.Figure()
                fig_speed.add_trace(go.Scatter(
                    x=steps, y=speed_trace, mode="lines",
                    name="Ground Speed (m/s)",
                    line=dict(color="#536dfe", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(83, 109, 254, 0.08)"
                ))
                fig_speed.update_layout(
                    title="Ground Speed Trace",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis_title="Step", yaxis_title="Speed (m/s)",
                    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17"
                )
                st.plotly_chart(fig_speed, use_container_width=True)

            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                # Power Draw
                fig_power = go.Figure()
                fig_power.add_trace(go.Scatter(
                    x=steps, y=power_trace, mode="lines",
                    name="Power Draw (W)",
                    line=dict(color="#ff9100", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255, 145, 0, 0.08)"
                ))
                fig_power.update_layout(
                    title="Power Consumption",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis_title="Step", yaxis_title="Power (W)",
                    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17"
                )
                st.plotly_chart(fig_power, use_container_width=True)

            with chart_col4:
                # Altitude
                fig_alt = go.Figure()
                fig_alt.add_trace(go.Scatter(
                    x=steps, y=alt_trace, mode="lines",
                    name="Altitude AGL (m)",
                    line=dict(color="#e040fb", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(224, 64, 251, 0.08)"
                ))
                fig_alt.update_layout(
                    title="Altitude Profile",
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=40, r=20, t=40, b=30),
                    xaxis_title="Step", yaxis_title="Altitude (m)",
                    plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17"
                )
                st.plotly_chart(fig_alt, use_container_width=True)
        else:
            st.warning("No step history available from simulation.")

        # Timeline Events (Phase 3 Closed-Loop + Phase 4 Temporal Memory)
        st.markdown("---")
        st.markdown("##### Chronological Event Timeline (Phase 3 Controller + Phase 4 Memory)")

        timeline = cl_result.get("chronological_timeline", [])
        events = cl_result.get("events_logged", [])

        if events:
            for evt in events:
                st.markdown(f"<div class='timeline-event'>{evt}</div>", unsafe_allow_html=True)
        elif timeline:
            for entry in timeline[:20]:
                st.markdown(f"<div class='timeline-event'>t={entry.get('timestamp_s', '?')}s - {entry.get('event', 'State recorded')}</div>", unsafe_allow_html=True)
        else:
            st.info("No notable events recorded during simulation (nominal mission).")

    # ═════════════════════════════════════════════════════════════════════
    # TAB 4: Reflection Inspector (Phase 1 Detail)
    # ═════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<span class="phase-badge">PHASE 1</span> **Agentic Self-Correction & Reflection Inspector**', unsafe_allow_html=True)
        st.write("Step-by-step audit log of how symbolic guardrail failures trigger diagnostic feedback to force LLM self-correction:")

        if n_history:
            for item in n_history:
                att = item["attempt"]
                val = item["validation"]
                prop = item["proposal"]

                is_valid = val["is_valid"]
                status_label = "CERTIFIED SAFE" if is_valid else "GUARDRAIL REJECTED"

                with st.expander(f"Reflection Step #{att} — {status_label}", expanded=True):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown("**LLM Proposed Action:**")
                        st.json({
                            "target_airspeed_ms": prop["target_airspeed_ms"],
                            "estimated_duration_s": prop["estimated_duration_s"],
                            "emergency_action": prop.get("emergency_action"),
                            "rationale": prop["rationale"],
                            "num_waypoints": len(prop.get("proposed_path", []))
                        })
                    with col_b:
                        st.markdown("**Symbolic Guardrail Verdict:**")
                        if is_valid:
                            st.success(val["diagnostic_message"])
                        else:
                            st.error(val["diagnostic_message"])
                        if val.get("violation_category"):
                            st.caption(f"Category: `{val['violation_category']}` | Error Magnitude: `{val.get('numerical_error', 0):.2f}`")
        else:
            st.info("No reflection history available. Run the pipeline first.")

    # ═════════════════════════════════════════════════════════════════════
    # TAB 5: Full Mission Report
    # ═════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("##### Comprehensive Mission Execution Report")

        st.markdown("---")
        st.markdown("**Phase 1 — Guardrail Comparison:**")
        r1, r2 = st.columns(2)
        r1.metric("Baseline Status", "SAFE" if (b_val and b_val.is_valid) else "REJECTED")
        r2.metric("NeuroSym Status", "SAFE" if (n_val and n_val.is_valid) else "REJECTED")

        st.markdown("**Phase 2-5 — Closed-Loop Simulation:**")
        r3, r4, r5, r6 = st.columns(4)
        r3.metric("Mission Outcome", "COMPLETED" if cl_result["is_completed"] else "FAILED")
        r4.metric("Steps Executed", cl_result["total_steps"])
        r5.metric("Final Battery", f"{cl_result['final_battery_pct']:.1f}%")
        r6.metric("Active Strategies", len(cl_result["strategies_used"]))

        st.markdown("**Phase 6 — Agent Mesh:**")
        r7, r8, r9, r10 = st.columns(4)
        r7.metric("Agents in Mesh", mesh_report["agents_executed"])
        r8.metric("Mesh Confidence", f"{mesh_report['aggregate_confidence']:.1%}")
        r9.metric("Total Flags", len(mesh_report["all_advisory_flags"]))
        r10.metric("Critical Flags", mesh_report["critical_flag_count"])

        st.markdown("---")
        st.markdown("**Raw Data Export:**")
        with st.expander("Full Closed-Loop Result JSON"):
            export = {k: v for k, v in cl_result.items() if k != "step_history"}
            st.json(export)

        with st.expander("Full Mesh Report JSON"):
            st.json(mesh_report)


if __name__ == "__main__":
    main()
