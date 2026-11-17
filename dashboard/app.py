"""Local captain review workspace using the explicitly selected saved classifier."""
import base64
import html
import math
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.data import DashboardDataError, captain_inference, filter_queue, fingerprint, load_dashboard
from dashboard.model_runtime import score_tolerance

st.set_page_config(page_title="Captain Safety Lab", page_icon="▥", layout="wide")
assets = Path(__file__).parent / "assets"
font = base64.b64encode((assets / "Geist.ttf").read_bytes()).decode()
css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8").replace("FONT_DATA", font)
st.html(f"<style>{css}</style>")


@st.cache_data(show_spinner=False)
def cached_data(version):
    return load_dashboard()


try:
    data = cached_data(fingerprint())
except (OSError, ValueError, KeyError, AssertionError) as error:
    st.error(str(error))
    st.stop()

summary = data["summary"]
captains = data["captains"]
validation = summary["validation"]
model_label = summary["model_label"]
st.html(f'<div class="brand"><span>Captain Safety Lab</span><div class="brand-right"><span class="badge">{html.escape(model_label)}</span><span class="muted">CEG2003 · Local workspace</span></div></div>')
st.title("Captain review workspace")
st.caption("March 2024 alert history · April 2024 reported incidents · Historical analysis")


def chart_style(chart, height=220):
    return chart.properties(height=height).configure_view(stroke=None).configure_axis(
        labelFont="Geist", titleFont="Geist", labelColor="#6b7280", titleColor="#6b7280",
        gridColor="#eef0f3", domain=False, tickColor="#e6e8eb", titleFontWeight="normal")


def trend_chart(daily, alarm_types, color="#1d5fbf"):
    observed = pd.DatetimeIndex(sorted(daily["day"].unique()))
    selected = daily.loc[daily["alert_type"].isin(alarm_types)].groupby("day")["recorded_alerts"].sum().reindex(observed, fill_value=0)
    selected = selected.reindex(pd.date_range("2024-03-01", "2024-03-31"))
    frame = selected.rename_axis("day").rename("recorded_alerts").reset_index()
    chart = alt.Chart(frame).mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=20)).encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", tickCount=5, grid=False)),
        y=alt.Y("recorded_alerts:Q", title="Recorded alert rows", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("day:T", title="Day", format="%d %b"), alt.Tooltip("recorded_alerts:Q", title="Recorded rows", format=",")])
    st.altair_chart(chart_style(chart, 195), width="stretch")


def recent_label(value):
    return "Insufficient coverage" if pd.isna(value) else f"{value:+.1f}%"


def show_captain(captain):
    row = captains.loc[captains["captain_id"] == captain].iloc[0]
    selected_types = data["types"].loc[data["types"]["captain_id"] == captain].copy()
    activity = data["daily"].loc[data["daily"]["captain_id"] == captain]
    score_flag = "Above model cutoff" if row["above_model_cutoff"] else "Below model cutoff"
    st.subheader(f"Captain {captain}")
    st.html(f'<div class="captain-line">Relative priority <strong>{row["priority_percentile"]:.1f}/100</strong> · Model score <strong>{row["model_score"]:.3f}</strong> · {score_flag}</div>')
    columns = st.columns(3)
    columns[0].metric("Recorded alert rows", f"{int(row['recorded_alerts']):,}", border=True)
    columns[1].metric("Days with alert records", str(int(row["observed_days"])), border=True)
    columns[2].metric("Rows per observed day", f"{row['alerts_per_observed_day']:,.2f}", border=True,
                      help="Recorded alert rows divided by calendar days containing any alert. This is not alerts per driving day or per hour.")
    st.caption("Repeated sensor records can describe the same period of activity. These totals count recorded rows; alert severity is not supplied.")
    left, right = st.columns(2)
    with left:
        st.subheader("Fatigue and yawn history")
        trend_chart(activity, ["Fatigue Alert", "Yawn"])
    with right:
        st.subheader("Collision-warning history")
        trend_chart(activity, ["Forward Collision Warning", "Headway Monitoring Alert", "Pedestrian Collision Alert"], "#697586")
    st.caption("Days without alert records have unknown driving activity. Recent rate change compares March25–31 with March1–24 using each period's observed alert days.")
    explanations, incident_tab, history_tab = st.tabs(["What influences this score", "Recorded incidents", "Daily records"])
    with explanations:
        result = captain_inference(data, captain)
        direction = lambda value: "raises" if value > 0 else "lowers" if value < 0 else "does not change"
        if summary.get("learned_alert_weights"):
            st.write(f"Compared with the saved reference inputs, the learned alert signal **{direction(result['weighted_alert_input_effect'])}** this score. Recorded-day coverage **{direction(result['observed_days_effect'])}** it.")
            st.caption("These effects average the two orders of replacing the saved reference inputs with this captain's inputs. They explain the ANN's score change, not causation or fault. Recorded-day coverage describes data availability, not driving exposure.")
            st.caption("Type weights were learned from training examples and stay fixed until retraining. Input shares below describe the learned alert mixture, not each alert's effect on accident probability.")
        else:
            st.write(f"The weighted alert rate **{direction(result['weighted_alert_input_effect'])}** the score relative to the training-average input. Recorded-day coverage **{direction(result['observed_days_effect'])}** it.")
            st.caption("The classifier combines these two inputs. Recorded-day coverage describes data availability, not driving exposure. Type weights describe frequency adjustment, not severity or fault.")
        chart = alt.Chart(selected_types).mark_bar(color="#1d5fbf", cornerRadiusEnd=2).encode(
            y=alt.Y("alert_type:N", title=None, sort="-x", axis=alt.Axis(labelLimit=240)),
            x=alt.X("weighted_input_share:Q", title="Share of weighted alert input", axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("alert_type:N", title="Alert type"), alt.Tooltip("recorded_alerts:Q", title="Recorded rows", format=","),
                     alt.Tooltip("weighted_input_share:Q", title="Weighted input share", format=".1%")])
        st.altair_chart(chart_style(chart, 260), width="stretch")
        st.caption("Input shares describe the weighted alert mixture before the classifier. They are not percentages of accident probability.")
        with st.expander("Type counts and weights"):
            view = selected_types[["alert_type", "recorded_alerts", "alerts_per_observed_day", "model_weight", "weighted_input_share"]].sort_values("weighted_input_share", ascending=False)
            st.dataframe(view, hide_index=True, width="stretch", column_config={
                "alert_type": "Alert type", "recorded_alerts": st.column_config.NumberColumn("Recorded rows", format="localized"),
                "alerts_per_observed_day": st.column_config.NumberColumn("Rows / observed day", format="%.2f"),
                "model_weight": st.column_config.NumberColumn("Type weight", format="%.4f"),
                "weighted_input_share": st.column_config.NumberColumn("Weighted input share", format="percent")})
        if st.button("Verify selected score", key="verify_score"):
            if np.isclose(result["score"], row["model_score"], **score_tolerance(data["model"])):
                st.success(f"Saved {model_label} inference matches the displayed score.")
            else:
                st.error("The saved score does not match current inference. Rebuild the dashboard data.")
    with incident_tab:
        outcome = "An eligible April incident report is present." if row["april_reported_incident"] else "No eligible April incident report is present in the workbook."
        st.write(outcome)
        reports = data["incidents"].loc[data["incidents"]["captain_id"] == captain].sort_values("date")
        if reports.empty:
            st.info("No incident reports for this captain appear in the cleaned workbook.")
        else:
            st.dataframe(reports[["report_id", "date", "time", "service", "category"]], hide_index=True, width="stretch",
                         column_config={"report_id": "Report", "date": "Incident date", "time": "Time", "service": "Service", "category": "Source category"})
        st.caption("Jan–May reports are shown as retrospective context. Only eligible April reports define the model outcome. Reported involvement does not establish fault.")
    with history_tab:
        daily_view = activity.pivot(index="day", columns="alert_type", values="recorded_alerts").fillna(0).astype(int)
        daily_view["Total recorded rows"] = daily_view.sum(axis=1)
        st.dataframe(daily_view.sort_index(ascending=False), width="stretch", height=280)
        st.caption("Only days with recorded alerts are listed. A zero for one type means other alert types were recorded that day.")


capacity = st.session_state.get("review_capacity", 10)
top = st.columns(3)
top[0].metric("Captains in the model", f"{len(captains):,}", border=True)
top[1].metric("Eligible April incident captains", "36", border=True)
top[2].metric("Review capacity", f"{int(np.ceil(len(captains) * capacity / 100)):,}", border=True,
              help="Number of highest-ranked captains at the selected review capacity. This is separate from the classifier's cutoff.")
st.html(f'<div class="notice"><strong>Held-out validation</strong> &nbsp; ROC-AUC <strong>{validation["roc_auc"]:.3f}</strong> &nbsp;·&nbsp; Accuracy <strong>{validation["accuracy"]:.2%}</strong> &nbsp;·&nbsp; Incident detection <strong>{validation["incident_detection_recall"]:.2%}</strong><br><span class="section-note">Internal prototype based on 36 reported-incident captains. Supports human review and consideration of training or support.</span></div>')

queue_tab, overview_tab, model_tab = st.tabs(["Captain review", "Alert overview", "Model & data"])
with queue_tab:
    controls = st.columns([1.05, 1.25, 1.0])
    with controls[0]:
        scope = st.selectbox("Queue", ["Review capacity", "Above model cutoff", "All captains"], key="queue_scope")
    with controls[1]:
        search = st.text_input("Find captain", placeholder="Captain ID within this queue", key="captain_search")
    with controls[2]:
        capacity = st.slider("Review capacity (%)", 1, 50, 10, key="review_capacity")
    ranked = filter_queue(captains, scope, capacity, search)
    st.subheader("Prioritised captain queue")
    st.caption("Priority /100 is the relative score percentile among these captains. It is not accident probability. The review-capacity control does not change the model or its validation ratings.")
    if ranked.empty:
        st.info("No captains match this queue selection. Clear the search or choose All captains.")
    else:
        display = ranked[["captain_id", "rank", "priority_percentile", "observed_days", "alerts_per_observed_day", "largest_weighted_type", "recent_rate_change_pct"]].copy()
        display["recent_rate_change_pct"] = display["recent_rate_change_pct"].map(recent_label)
        table = st.dataframe(display, hide_index=True, width="stretch", height=300, on_select="rerun", selection_mode="single-row", key="review_queue",
                             column_config={"captain_id": st.column_config.TextColumn("Captain", width=95, pinned=True),
                                            "rank": st.column_config.NumberColumn("Rank", width=65, format="%d"),
                                            "priority_percentile": st.column_config.ProgressColumn("Priority /100", min_value=0, max_value=100, format="%.1f", width=125),
                                            "observed_days": st.column_config.NumberColumn("Alert days", width=85, format="%d"),
                                            "alerts_per_observed_day": st.column_config.NumberColumn("Rows / observed day", format="%.2f", width=155),
                                            "largest_weighted_type": st.column_config.TextColumn("Largest weighted input", width=225),
                                            "recent_rate_change_pct": st.column_config.TextColumn("Recent rate change", width=145)})
        caption_col, download_col = st.columns([4, 1])
        caption_col.caption(f"{len(ranked):,} captains shown. Select a row or use the captain selector below. Scores come from the saved historical full-data model.")
        download_col.download_button("Export queue", ranked.assign(model=model_label).to_csv(index=False).encode("utf-8-sig"),
                                     file_name=f"{summary['model_id']}_captain_review.csv", mime="text/csv", width="stretch")
        selection = table.selection.rows
        chosen = display.iloc[selection[0]]["captain_id"] if selection and selection[0] < len(display) else None
        if chosen != st.session_state.get("last_queue_selection"):
            st.session_state["last_queue_selection"] = chosen
            if chosen is not None:
                st.session_state["detail_captain"] = chosen
        options = ranked["captain_id"].tolist()
        if st.session_state.get("detail_captain") not in options:
            st.session_state["detail_captain"] = options[0]
        with st.expander("Captain detail and alert history", expanded=True):
            captain = st.selectbox("Captain detail", options, key="detail_captain")
            show_captain(captain)

with overview_tab:
    st.subheader("March alert records")
    overview = st.columns(3)
    overview[0].metric("Recorded rows in model cohort", f"{summary['model_cohort_alert_rows']:,}", border=True)
    overview[1].metric("Captain-days with records", f"{summary['model_observed_captain_days']:,}", border=True)
    overview[2].metric("Original alert types", "10", border=True)
    daily_total = data["daily"].groupby("day", as_index=False)["recorded_alerts"].sum()
    chart = alt.Chart(daily_total).mark_bar(color="#1d5fbf", cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("day:T", title=None, axis=alt.Axis(format="%d %b", tickCount=8, grid=False)),
        y=alt.Y("recorded_alerts:Q", title="Recorded alert rows", axis=alt.Axis(format="~s")),
        tooltip=[alt.Tooltip("day:T", title="Day", format="%d %b"), alt.Tooltip("recorded_alerts:Q", title="Recorded rows", format=",")])
    st.altair_chart(chart_style(chart, 230), width="stretch")
    type_totals = data["types"].groupby("alert_type", as_index=False).agg(recorded_alerts=("recorded_alerts", "sum"), model_weight=("model_weight", "first"))
    type_totals["share"] = type_totals["recorded_alerts"] / type_totals["recorded_alerts"].sum()
    st.subheader("All ten alert types")
    left, right = st.columns([1.2, 1])
    with left:
        chart = alt.Chart(type_totals).mark_bar(color="#697586", cornerRadiusEnd=2).encode(
            y=alt.Y("alert_type:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)),
            x=alt.X("recorded_alerts:Q", title="Recorded rows", axis=alt.Axis(format="~s")),
            tooltip=[alt.Tooltip("alert_type:N", title="Type"), alt.Tooltip("recorded_alerts:Q", title="Rows", format=","), alt.Tooltip("share:Q", title="Share", format=".1%")])
        st.altair_chart(chart_style(chart, 320), width="stretch")
    with right:
        st.dataframe(type_totals.sort_values("recorded_alerts", ascending=False), hide_index=True, width="stretch", height=390,
                     column_config={"alert_type": "Alert type", "recorded_alerts": st.column_config.NumberColumn("Recorded rows", format="localized"),
                                    "model_weight": st.column_config.NumberColumn("Model weight", format="%.4f"), "share": st.column_config.NumberColumn("Share", format="percent")})
    st.caption("Counts cover the 1,541 scored captains. The full cleaned sources contain 3,958,476 rows across 1,544 captains; the three duplicate-only April-report captains have no model score. All original cleaned rows remain preserved.")

with model_tab:
    st.subheader(f"{model_label} model validation")
    columns = st.columns(3)
    columns[0].metric("ROC-AUC", f"{validation['roc_auc']:.3f}", border=True, help=f"Exact value {validation['roc_auc']:.9f}. Internal exploratory comparison on the same dataset.")
    columns[1].metric("Binary accuracy", f"{validation['accuracy']:.2%}", border=True)
    columns[2].metric("Incident detection", f"{validation['incident_detection_recall']:.2%}", border=True)
    if summary.get("learned_alert_weights"):
        st.write(f"**Active ANN:** {summary['architecture']} · **3 hidden layers** · **1 sigmoid output**")
        st.caption("Alert weights are learned during training. This version uses class weighting without SMOTE. Its displayed flags use the training-selected cutoff.")
    st.caption("Means of three complete five-fold tests. Every test captain was excluded from that fold's training and tuning. The same sample has been explored repeatedly; this is internal prototype evidence.")
    target = summary["comparison_targets"]
    target_text = f"ROC-AUC ≥ {target['roc_auc']:.3f}, accuracy ≥ {target['accuracy']:.2%}, incident detection ≥ {target['incident_detection_recall']:.2%}"
    st.caption(("Meets" if summary["targets_met"] else "Does not meet all") + f" comparison benchmarks: {target_text}. These are internal test results.")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Held-out ROC curve")
        x_axis = np.linspace(0, 1, 501)
        curves = []
        for _, group in data["heldout"].groupby("repeat"):
            fpr, tpr, _ = roc_curve(group["actual_april_report"], group["score"])
            curves.append(np.interp(x_axis, fpr, tpr))
        curve_data = pd.DataFrame({"false_positive_rate": x_axis, "detection_rate": np.mean(curves, axis=0)})
        curve = alt.Chart(curve_data).mark_line(color="#1d5fbf", strokeWidth=2).encode(
            x=alt.X("false_positive_rate:Q", title="No-report captains flagged", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y=alt.Y("detection_rate:Q", title="Incident captains detected", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("false_positive_rate:Q", title="False flags", format=".1%"), alt.Tooltip("detection_rate:Q", title="Detection", format=".1%")])
        diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#adb5c1", strokeDash=[5, 5]).encode(x="x:Q", y="y:Q")
        st.altair_chart(chart_style(alt.layer(diagonal, curve), 265), width="stretch")
        interval = summary["uncertainty"]["roc_auc"]
        st.caption(f"Conditional ROC-AUC interval: {interval['low']:.3f}–{interval['high']:.3f}. This does not account for repeated model exploration. The plot averages held-out curves; the metric above uses the saved exact mean AUC.")
    with right:
        st.subheader("What the classifier detected")
        st.dataframe(pd.DataFrame([
            {"Validation outcome": "Reported incident detected", "Average captains": validation["true_positives"]},
            {"Validation outcome": "Reported incident missed", "Average captains": validation["false_negatives"]},
            {"Validation outcome": "Flagged without eligible report", "Average captains": validation["false_positives"]},
            {"Validation outcome": "Unflagged without eligible report", "Average captains": validation["true_negatives"]},
        ]), hide_index=True, width="stretch", column_config={"Average captains": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Counts are averages across the three complete repeats. Review capacity is a separate human workflow setting and does not define this accuracy rating.")
        st.write(f"Always predicting no report would have **97.66% accuracy** and **0% incident detection**. Current precision among model flags is **{validation['precision']:.2%}**.")
    with st.expander("How ranking and model flags work"):
        adjustment = "learned-weight" if summary.get("learned_alert_weights") else "frequency-adjusted"
        st.write(f"The saved {model_label} binary classifier uses one {adjustment} alert rate and the number of days with records. Both original model inputs remain included.")
        st.write(f"The full-data model's cutoff is **{summary['model_cutoff']:.6f}**, currently flagging **{summary['full_fit_model_flags']:,}** captains. The default 10% review capacity shows **155** highest-ranked captains. These are different selections.")
        st.write("Queue scores come from the model refitted on all labelled captains for historical scoring. The performance cards come from held-out tests. Full-fit scores are not independent predictions for these same captains, and the model score is not a calibrated accident probability.")
    with st.expander("Data scope and cleaning", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Dataset": "Cleaned March alert rows", "Count": 3958476},
            {"Dataset": "Cleaned captain IDs", "Count": 1544},
            {"Dataset": "Scored captain IDs", "Count": 1541},
            {"Dataset": "Cleaned incident reports, January–May", "Count": 194},
            {"Dataset": "April source report rows", "Count": 39},
            {"Dataset": "Eligible April reports / positive captains", "Count": 36},
            {"Dataset": "Duplicate-only April captains outside model", "Count": 3},
        ]), hide_index=True, width="stretch")
        st.write("Missing-ID alert rows and incident reports with unmatched employee IDs were excluded during cleaning. No captain IDs were invented or grouped as unknown. Excluded records remain in original sources and the separate audit.")
        st.caption("Eligible April reports have a blank source category or CAT 1. No eligible April report means no incident reported in this workbook. Model outcomes describe reported involvement, not fault.")
st.caption(f"Captain Safety Lab · Active model: {model_label} · Original cleaned data preserved")
