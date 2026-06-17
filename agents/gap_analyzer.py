from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import GapAnalysis , ResearchGap
from models.state import ResearchState

GAP_ANALYZER_PROMPT = """
You are a Senior Research Director.
Your ONLY responsibility is to identify
knowledge gaps that require additional research.
You are NOT writing the report.
You are NOT improving wording.
You are deciding what information
is still missing.
---------------------------------------
INPUTS
You receive:
1. Research Plan
2. Original Search Queries
3. Completed Research
4. Current Report
5. Critic Feedback
---------------------------------------
YOUR JOB
Identify the minimum additional
research required to substantially
improve the report.
Each gap must contain:
- topic
- expected_section
- reason
- search_query
- priority
---------------------------------------
RULES
Generate AT MOST 3 gaps.
Do NOT duplicate any
existing search query.
Do NOT generate broad queries.
Generate concrete,
research-oriented queries.
Higher priority means
higher impact.
Priority:
1 = Critical
2 = High
3 = Medium
4 = Low
5 = Optional
---------------------------------------
GOOD QUERY
"economic impact of AI in healthcare hospitals"
BAD QUERY
"AI Healthcare"
---------------------------------------
If the report is already
well-supported,
return an empty list.
"""

def gap_analyzer_agent(
    state: ResearchState,
) -> ResearchState:

    planning = state["planning_output"]

    writer = state["writer_output"]

    critic = state["critic_output"]

    research_outputs = state["research_outputs"]

    # -----------------------------
    # Existing planner queries
    # -----------------------------

    planner_queries = planning.search_queries

    # -----------------------------
    # Existing executed queries
    # -----------------------------

    executed_queries = [
        item.search_query
        for item in research_outputs
    ]

    # -----------------------------
    # Research summary
    # -----------------------------

    research_context = ""

    for item in research_outputs:

        findings = "\n".join(
            f"- {finding}"
            for finding in item.key_findings
        )

        research_context += f"""
SEARCH QUERY:
{item.search_query}

SUMMARY:
{item.summary}

KEY FINDINGS:
{findings}

"""
        
    structured_llm = (
        llm.with_structured_output(
            GapAnalysis
        )
    )

    gap_analysis = (
        structured_llm.invoke(
            [
                HumanMessage(
                    content=f"""
    {GAP_ANALYZER_PROMPT}

    TOPIC:
    {planning.topic}

    OBJECTIVE:
    {planning.objective}

    PLANNER SEARCH QUERIES:
    {planner_queries}

    EXECUTED SEARCH QUERIES:
    {executed_queries}

    CURRENT RESEARCH:
    {research_context}

    CURRENT REPORT:
    {writer.report}

    CRITIC WEAKNESSES:
    {critic.weaknesses}

    CRITIC IMPROVEMENT SUGGESTIONS:
    {critic.improvement_suggestions}
    """
                )
            ]
        )
    )
        
    existing_queries = {

        query.lower().strip()

        for query in planner_queries

    }

    existing_queries.update(

        item.search_query.lower().strip()

        for item in research_outputs

    )

    seen = set()

    filtered_gaps = []

    for gap in gap_analysis.gaps:

        query = gap.search_query.lower().strip()

        if query in existing_queries:
            continue

        if query in seen:
            continue

        seen.add(query)

        filtered_gaps.append(gap)

    filtered_gaps.sort(
        key=lambda gap: gap.priority
    )

    gap_analysis.gaps = filtered_gaps[:3]

    state["gap_analysis"] = gap_analysis

    print("\nGap Analyzer completed")

    print(
        f"Gaps Identified: "
        f"{len(gap_analysis.gaps)}"
    )

    for gap in gap_analysis.gaps:

        print(
            f"[P{gap.priority}] "
            f"{gap.topic}"
        )

    return state