import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import BAR, bundle, chart, md, style
from src.copilot import SUGGESTED_QUESTIONS, Copilot
from src.llm import llm_available
from src.sql_lab import QueryError, run_query, schema, templates
from theme import hero, section

hero("AI Copilot & SQL Lab",
     "Ask the churn data questions in plain English - Claude writes and runs read-only SQL "
     "and answers with the numbers. Prefer to write the SQL yourself? Use the SQL Lab.",
     eyebrow="Generative AI on your analytics layer",
     chips=["Claude tool use", "Read-only SQL sandbox", "Every number is sourced"])

if llm_available():
    tab_ai, tab_sql = st.tabs([":material/auto_awesome: Copilot", ":material/terminal: SQL Lab"])
else:  # no key: open on the tab that works offline
    tab_sql, tab_ai = st.tabs([":material/terminal: SQL Lab", ":material/auto_awesome: Copilot"])


def auto_chart(df: pd.DataFrame, key: str) -> None:
    """Draw a bar chart when a result is one label column plus a numeric column."""
    if len(df) < 2 or len(df) > 40 or df.shape[1] < 2:
        return
    label, numeric = df.columns[0], [c for c in df.columns[1:]
                                      if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return
    business = ("loss", "revenue", "value", "rate", "pct", "churned", "customers", "count")
    value = next((c for c in numeric if any(k in c.lower() for k in business)), numeric[-1])
    fig = go.Figure(go.Bar(x=df[value], y=df[label].astype(str), orientation="h",
                           marker_color=BAR, hovertemplate="%{y}: %{x:,}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    chart(style(fig, min(80 + 28 * len(df), 480), f"{value} by {label}", legend=False), key=key)


def show_steps(steps, prefix: str) -> None:
    for i, step in enumerate(steps):
        title = step.input.get("purpose") or step.tool.replace("_", " ")
        with st.expander(f":material/database: {title}", expanded=False):
            if step.tool == "run_sql":
                st.code(step.input.get("query", ""), language="sql")
            if step.error:
                st.error(step.error)
            elif isinstance(step.result, pd.DataFrame):
                st.dataframe(step.result, hide_index=True)
                auto_chart(step.result, key=f"{prefix}_{i}")
            elif step.result is not None:
                st.json(step.result)


# --------------------------------------------------------------------------- #
# Copilot
# --------------------------------------------------------------------------- #
with tab_ai:
    if not llm_available():
        st.info(
            "**The Copilot needs a Claude API key.** Add `ANTHROPIC_API_KEY` to your environment "
            "(or to Streamlit Cloud *Secrets*) and reload. Until then, the SQL Lab tab gives you "
            "the same database with ready-made queries.", icon=":material/key:")
        st.markdown("Questions it is built for:")
        for q in SUGGESTED_QUESTIONS:
            st.markdown(f"- {q}")
    else:
        state = st.session_state.setdefault("copilot", {"history": [], "turns": []})
        for i, turn in enumerate(state["turns"]):
            with st.chat_message("user"):
                st.markdown(md(turn["question"]))
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.markdown(md(turn["answer"].text))
                show_steps(turn["answer"].steps, prefix=f"t{i}")

        picked = None
        if not state["turns"]:
            st.caption("Try one of these:")
            picked = st.pills("Suggestions", SUGGESTED_QUESTIONS, label_visibility="collapsed")
        question = st.chat_input("Ask about churn, revenue, segments or a customer ID...") or picked
        if question:
            with st.chat_message("user"):
                st.markdown(md(question))
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                with st.spinner("Querying the data..."):
                    try:
                        copilot = Copilot(threshold=bundle()["threshold"])
                        answer = copilot.ask(question, history=state["history"])
                    except Exception as e:  # network, auth, rate limit: show, don't crash
                        st.error(f"The Copilot could not answer: {type(e).__name__}: {e}")
                        st.stop()
                st.markdown(md(answer.text))
                show_steps(answer.steps, prefix=f"t{len(state['turns'])}")
            state["history"] = answer.messages
            state["turns"].append({"question": question, "answer": answer})
        if state["turns"] and st.button("New conversation", icon=":material/restart_alt:"):
            st.session_state.pop("copilot")
            st.rerun()

# --------------------------------------------------------------------------- #
# SQL Lab
# --------------------------------------------------------------------------- #
with tab_sql:
    library = templates()
    left, right = st.columns([3, 1])
    with left:
        name = st.selectbox("Start from a template", list(library),
                            format_func=lambda n: n.replace("_", " ").capitalize())
        sql = st.text_area("SQL (read-only, SQLite dialect)", library[name], height=220,
                           key=f"sql_{name}")
        st.button("Run query", type="primary", icon=":material/play_arrow:")
    with right:
        section("Schema")
        for table, cols in schema().items():
            with st.expander(f":material/table: {table}"):
                st.caption(", ".join(c for c, _ in cols))

    # Queries are read-only, capped and time-limited, so results show immediately;
    # the button just forces a re-run.
    if sql.strip():
        try:
            df, truncated = run_query(sql)
        except QueryError as e:
            st.error(str(e), icon=":material/block:")
        else:
            st.success(f"{len(df):,} rows" + (" (capped at 5,000)" if truncated else ""),
                       icon=":material/check_circle:")
            st.dataframe(df, hide_index=True)
            auto_chart(df, key="sql_chart")
            st.download_button("Download result (CSV)", df.to_csv(index=False),
                               file_name=f"{name}.csv", mime="text/csv",
                               icon=":material/download:")
    st.caption("The database is opened read-only; writes, ATTACH and PRAGMA are blocked and "
               "queries stop after 5 seconds.")
