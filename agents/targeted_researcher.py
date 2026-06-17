from models.state import ResearchState
from agents.researcher import research_query


MAX_ADDITIONAL_SEARCHES = 3


def targeted_research_agent(
    state: ResearchState
) -> ResearchState:
    """
    Performs targeted research for the
    highest-priority gaps identified by the
    Gap Analyzer.
    """

    gap_analysis = state["gap_analysis"]

    additional_outputs = []

    if not gap_analysis:
        state["additional_research_outputs"] = []
        return state

    print("\nTargeted Research Started")

    for gap in gap_analysis.gaps[:MAX_ADDITIONAL_SEARCHES]:

        print(
            f"\nResearching Gap (P{gap.priority}): "
            f"{gap.topic}"
        )

        result = research_query(
            query=gap.search_query
        )

        if result:

            additional_outputs.append(
                result
            )

    state[
        "additional_research_outputs"
    ] = additional_outputs

    print(
        f"\nTargeted Research completed "
        f"({len(additional_outputs)} new evidence blocks)"
    )

    return state    