"""Local dashboard for the published March captain safety review."""

from datetime import date, timedelta
from pathlib import Path

import altair as alt
import streamlit as st

from dashboard.data import ALERTS, ROOT, WEEK, fingerprint, load_snapshot
from dashboard.monthly_alerts import display_alert_summary
from dashboard.roc import roc_points


st.set_page_config(page_title="Captain Safety Lab · March review", layout="wide")
st.html(Path(__file__).with_name("style.css"))


@st.cache_data(show_spinner=False)
def saved_dashboard_data(stamp):
    # The file fingerprint in the cache key changes when project outputs are replaced.
    snapshot = load_snapshot()
    return snapshot, display_alert_summary(snapshot, ROOT / ALERTS)


def week_label(start):
    last = start + WEEK - timedelta(days=1)
    first = start.strftime("%d") if start.month == last.month else start.strftime("%d %b")
    return f"{first}–{last:%d %b %Y}"


def show_queue(rows, key, cutoff):
    if rows.empty:
        st.info("No captains match these filters. Try a different ID or select All captains.")
        return None

    def select_queue_captain():
        positions = st.session_state[key]["selection"]["rows"]
        row = rows.iloc[positions[0] if positions else 0]
        st.session_state["flagged_captain"] = int(row.captain_id) if row.flagged else None

    display = rows.assign(captain_id=rows.captain_id.astype(str)).reset_index(drop=True)
    selection = st.dataframe(
        display[["rank", "captain_id", "review_score", "observed_days", "records", "flagged"]],
        hide_index=True, width="stretch", height=247,
        on_select=select_queue_captain, selection_mode="single-row", key=key,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "captain_id": st.column_config.TextColumn("Captain ID"),
            "review_score": st.column_config.ProgressColumn("Review score", min_value=0, max_value=1,
                                                            format="%.3f", width="medium"),
            "observed_days": st.column_config.NumberColumn("Days with alerts", format="%d"),
            "records": st.column_config.NumberColumn("Recorded alerts", format="localized"),
            "flagged": st.column_config.CheckboxColumn("Flagged"),
        })
    selected = selection.selection.rows
    noun = "captain" if len(rows) == 1 else "captains"
    st.caption(f"{len(rows):,} matching {noun} · Scores of {cutoff:.2f} or higher are flagged.")
    return rows.iloc[selected[0] if selected and 0 <= selected[0] < len(rows) else 0]


def review_tab(data, alert_week):
    st.header("Prioritised review queue")
    scope_col, search_col = st.columns([1, 2])
    scope = scope_col.selectbox("Review scope", ["Flagged captains", "All captains", "Below cutoff"])
    query = search_col.text_input("Find captain", placeholder="Enter a captain ID").strip()
    rows = data.queue
    if scope != "All captains":
        rows = rows.loc[rows.flagged == (scope == "Flagged captains")]
    if query:
        rows = rows.loc[rows.captain_id.astype(str).str.contains(query, regex=False)]
    queue_key = f"review_rows_{scope}_{query}"
    row = show_queue(rows, queue_key, data.report["cutoff"])
    flagged = data.queue.loc[data.queue.flagged].set_index("captain_id", drop=False)
    if st.session_state.get("detail_queue_key") != queue_key:
        st.session_state["detail_queue_key"] = queue_key
        if row is not None:
            st.session_state["flagged_captain"] = int(row.captain_id) if row.flagged else None

    def highlight_picked_captain():
        if not rows.empty:
            positions = rows.captain_id.eq(st.session_state["flagged_captain"]).to_numpy().nonzero()[0]
            st.session_state[queue_key] = {"selection": {"rows": positions.tolist()}}

    with st.container(border=True):
        captain = st.selectbox("Choose a flagged captain", flagged.index.tolist(), index=None,
                               key="flagged_captain", on_change=highlight_picked_captain,
                               placeholder="Search by captain ID", disabled=flagged.empty)
        if captain is not None:
            row = flagged.loc[captain]
        if row is None:
            return
        st.subheader(f"Captain {int(row.captain_id)}")
        status = "Flagged for review" if row.flagged else "Below the review cutoff"
        st.write(f"**Rank {int(row['rank']):,} of {len(data.queue):,}** · {status} · Score {row.review_score:.3f}")
        counts = data.counts.loc[row.captain_id]
        observed = counts[counts > 0].sort_values(ascending=False, kind="stable")
        st.caption(f"{int(observed.sum()):,} recorded alerts across {int(row.observed_days)} observed days "
                   f"during {alert_week}. Repeated sensor records count separately.")
        table = observed.rename_axis("Alert type").reset_index(name="Recorded alerts")
        st.dataframe(table, hide_index=True, width="stretch", height="content",
                     column_config={"Recorded alerts": st.column_config.ProgressColumn(
                         "Times recorded", min_value=0, max_value=int(observed.max()), format="%d")})


def fold_chart(predictions):
    options = {"Five-fold mean": None}
    options.update({f"Fold {number}": number for number in sorted(predictions.evaluation_fold.unique())})
    selected = st.selectbox("ROC curve to display", list(options), label_visibility="collapsed")
    points = roc_points(predictions, options[selected])
    axis = alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1], gridColor="#edf1f5",
                    tickSize=0, labelPadding=6, titlePadding=10)
    axes = {
        "x": alt.X("false_positive_rate:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1]), axis=axis),
        "y": alt.Y("detection_rate:Q", title="Detection rate", scale=alt.Scale(domain=[0, 1]), axis=axis),
    }
    chance = alt.Chart(alt.Data(values=[
        {"false_positive_rate": 0, "detection_rate": 0}, {"false_positive_rate": 1, "detection_rate": 1},
    ])).mark_line(color="#9aa7b5", strokeWidth=1, strokeDash=[4, 5]).encode(**axes)
    line = alt.Chart(points).mark_line(color="#3b6691", strokeWidth=2, strokeJoin="round").encode(
        **axes, order="point:Q", tooltip=[alt.Tooltip("false_positive_rate:Q", title="False positive rate", format=".1%"),
                                         alt.Tooltip("detection_rate:Q", title="Detection rate", format=".1%")])
    st.altair_chart((chance + line).properties(height=240).configure_view(stroke=None), width="stretch",
                    alt=f"{selected} ROC curve with a dashed chance reference.")
    st.caption("The dashed diagonal shows chance. With only one or two incident captains per fold, "
               "the detection rate changes in steps.")


def correlation_chart(data):
    alert_types = data.counts.sum().sort_values(ascending=False, kind="stable").index
    features = data.counts[alert_types].copy()
    features["All alerts"] = data.counts.sum(axis=1)
    features["Days with alerts"] = data.queue.set_index("captain_id").observed_days
    names = list(features.columns)
    pairs = features.corr().rename_axis("First").reset_index().melt(
        id_vars="First", var_name="Second", value_name="Correlation")
    matrix = alt.Chart(pairs).encode(
        x=alt.X("First:N", sort=names, title=None, axis=alt.Axis(
            values=names, labelOverlap=False, labelAngle=-55, labelLimit=200)),
        y=alt.Y("Second:N", sort=names, title=None, axis=alt.Axis(
            values=names, labelOverlap=False, labelLimit=200)),
        tooltip=["First:N", "Second:N", alt.Tooltip("Correlation:Q", format=".2f")])
    tiles = matrix.mark_rect(stroke="white").encode(color=alt.condition(
        "isValid(datum.Correlation)", alt.Color(
        "Correlation:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#ab6339", "#f3f6fa", "#235497"]),
        legend=None), alt.value("#ebedef")))
    labels = matrix.mark_text(fontSize=12).transform_calculate(
        valueLabel="isValid(datum.Correlation) ? format(datum.Correlation, '.2f') : '—'"
    ).encode(
        text="valueLabel:N",
        color=alt.condition("abs(datum.Correlation) > 0.70", alt.value("white"), alt.value("#182d46")))
    with st.container(key="alert-correlation-matrix"):
        st.altair_chart((tiles + labels).properties(height=32 * len(names)), width="stretch",
                        alt="Pearson correlations for every alert type, total alerts and days with alerts.")


def model_tab(data, alert_summary):
    report = data.report
    st.header("Model & data")
    st.write(f"**Weighted neural network** · {report['architecture'].replace(' -> ', ' → ')}")
    st.subheader("ROC curve")
    fold_chart(data.predictions)
    st.subheader("Alert feature correlations")
    correlation_chart(data)
    st.caption(f"All {len(data.counts.columns)} alert types, total alerts and days with alerts. "
               f"Each captain contributes one weekly count per type for "
               f"{week_label(date.fromisoformat(report['alert_week_start']))}.")
    with st.expander("March alert totals and model weights", expanded=True):
        st.write(f"**{alert_summary.month_records.sum():,} recorded alerts** · 1–31 March 2024")
        st.dataframe(alert_summary, hide_index=True,
                     width="stretch", height="content", column_config={
                         "alert_type": "Alert type",
                         "month_records": st.column_config.NumberColumn("March alert records", format="localized"),
                         "weight": st.column_config.NumberColumn("Training weight", format="%.4f",
                             help="The model calculated these weights using alerts from 1–14 March.")})


try:
    data, alert_summary = saved_dashboard_data(fingerprint(ROOT))
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the project's verified March review. {error}")
    st.info("Check the cleaned files in data/processed and the saved run in outputs/final. "
            "Run python -m training from the project root to verify or build the saved run.")
    st.stop()

report = data.report
start = date.fromisoformat(report["alert_week_start"])
alert_week = week_label(start)
st.html('<div class="brand-line"><div class="brand-name"><span class="brand-mark" aria-hidden="true">'
        '<i></i><i></i><i></i></span>Captain Safety Lab</div></div>')
st.title("Which captains need a closer review?")
cols = st.columns(4)
cols[0].metric("Scored captains", f"{len(data.queue):,}")
cols[1].metric("Flagged in review queue", f"{int(data.queue.flagged.sum()):,}",
               help="Captains crossing the saved review model’s cutoff.")
cols[2].metric("Mean five-fold ROC-AUC", f"{report['ratings']['roc_auc']:.4f}")
cols[3].metric("Mean five-fold accuracy", f"{report['ratings']['accuracy']:.2%}")
st.caption("Scores guide human review. A flag alone does not establish unsafe driving or fault.")
review, model = st.tabs(["Captain review", "Model & data"])
with review:
    review_tab(data, alert_week)
with model:
    model_tab(data, alert_summary)
