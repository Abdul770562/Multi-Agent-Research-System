from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agents.reviser import reviser_agent
from models.state import ResearchState

from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.validator import validator_agent
from agents.writer import writer_agent
from agents.critic import critic_agent


# def validation_router(
#     state: ResearchState
# ) -> str:

#     validation_output = (
#         state["validation_output"]
#     )

#     if validation_output.is_valid:
#         return "writer"

#     return END

def validation_router(
    state: ResearchState
):
    return "writer"


MAX_REVISIONS = 2

MIN_IMPROVEMENT = 0.3


def critic_router(
    state: ResearchState
):

    critic_output = state[
        "critic_output"
    ]

    current_score = (
        critic_output.overall_score
    )

    previous_score = state[
        "previous_critic_score"
    ]

    revision_count = state[
        "revision_count"
    ]

    if current_score >= 9.5:
        return END

    if revision_count >= MAX_REVISIONS:
        return END

    if (
        revision_count > 0
        and current_score <= previous_score
    ):
        print(
            "No improvement detected."
        )
        return END

    return "reviser"


def build_graph():

    workflow = StateGraph(
        ResearchState
    )

    workflow.add_node(
        "planner",
        planner_agent
    )

    workflow.add_node(
        "researcher",
        researcher_agent
    )

    workflow.add_node(
        "validator",
        validator_agent
    )

    workflow.add_node(
        "writer",
        writer_agent
    )

    workflow.add_node(
        "critic",
        critic_agent
    )

    workflow.add_node(
    "reviser",
    reviser_agent
    )

    workflow.add_edge(
        START,
        "planner"
    )

    workflow.add_edge(
        "planner",
        "researcher"
    )

    workflow.add_edge(
        "researcher",
        "validator"
    )

    workflow.add_conditional_edges(
        "validator",
        validation_router,
        {
            "writer": "writer",
            END: END
        }
    )

    workflow.add_edge(
        "writer",
        "critic"
    )

    workflow.add_edge(
    "reviser",
    "critic"
    )

    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "reviser": "reviser",
            END: END
        }
    )

    return workflow.compile()