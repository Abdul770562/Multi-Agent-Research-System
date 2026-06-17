from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from models.state import ResearchState

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.validator import validator_agent
from agents.writer import writer_agent
from agents.critic import critic_agent
from agents.gap_analyzer import gap_analyzer_agent
from agents.targeted_researcher import (
    targeted_research_agent,
)
from agents.reviser import reviser_agent


MAX_REVISIONS = 2
MIN_APPROVAL_SCORE = 8.0

def validation_router(
    state: ResearchState
):
    return "writer"

MIN_IMPROVEMENT = 0.3


def critic_router(
    state: ResearchState,
):

    critic = state["critic_output"]

    current_score = (
        critic.overall_score
    )

    revision_count = state[
        "revision_count"
    ]

    previous_score = state.get(
        "previous_critic_score",
        0.0,
    )

    if critic.approved:
        print(
            "\nReport Approved."
        )
        return END

    if current_score >= MIN_APPROVAL_SCORE:
        print(
            "\nReached target score."
        )
        return END

    if revision_count >= MAX_REVISIONS:

        print(
            "\nMaximum revisions reached."
        )

        return END

    if (
        revision_count > 0
        and current_score <= previous_score
    ):

        print(
            "\nNo improvement detected."
        )

        return END

    return "gap_analyzer"


def build_graph():

    workflow = StateGraph(
        ResearchState
    )

    workflow.add_node(
        "planner",
        planner_agent,
    )

    workflow.add_node(
        "researcher",
        researcher_agent,
    )

    workflow.add_node(
        "validator",
        validator_agent,
    )

    workflow.add_node(
        "writer",
        writer_agent,
    )

    workflow.add_node(
        "critic",
        critic_agent,
    )

    workflow.add_node(
        "gap_analyzer",
        gap_analyzer_agent,
    )

    workflow.add_node(
        "targeted_researcher",
        targeted_research_agent,
    )

    workflow.add_node(
        "reviser",
        reviser_agent,
    )

    workflow.add_edge(
        START,
        "planner",
    )

    workflow.add_edge(
        "planner",
        "researcher",
    )

    workflow.add_edge(
        "researcher",
        "validator",
    )

    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {
            "writer": "writer",
        },
    )

    workflow.add_edge(
        "writer",
        "critic",
    )

    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "gap_analyzer": "gap_analyzer",
            END: END,
        },
    )

    workflow.add_edge(
        "gap_analyzer",
        "targeted_researcher",
    )

    workflow.add_edge(
        "targeted_researcher",
        "reviser",
    )

    workflow.add_edge(
        "reviser",
        "critic",
    )

    return workflow.compile()