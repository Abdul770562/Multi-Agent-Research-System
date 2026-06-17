from graph.workflow import build_graph


def print_separator(title: str):

    print("\n")
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)


def main():

    graph = build_graph()

    initial_state = {

        "query": input(
            "Enter research topic: "
        ),

        "planning_output": None,

        "research_outputs": [],

        "additional_research_outputs": [],

        "gap_analysis": None,

        "validation_output": None,

        "writer_output": None,

        "critic_output": None,

        "revision_count": 0,

        "previous_critic_score": 0.0,

        "report_versions": []

    }

    result = graph.invoke(
        initial_state
    )

    writer = result.get(
        "writer_output"
    )

    critic = result.get(
        "critic_output"
    )

    planning = result.get(
        "planning_output"
    )

    print_separator(
        "MULTI AGENT RESEARCH SYSTEM"
    )

    print(
        f"Topic              : {planning.topic}"
    )

    print(
        f"Research Iterations: {result['revision_count']}"
    )

    print(
        f"Final Score        : {critic.overall_score:.1f}/10"
    )

    print(
        f"Approved           : {critic.approved}"
    )

    print_separator(
        "RESEARCH REPORT"
    )

    if writer:

        print(
            writer.report
        )

    else:

        print(
            "No report generated."
        )

    print_separator(
        "REFERENCES"
    )

    if writer and writer.references:

        for index, reference in enumerate(
            writer.references,
            start=1
        ):

            print(
                f"{index}. {reference}"
            )

    else:

        print(
            "No references available."
        )

    print_separator(
        "FINAL CRITIC REVIEW"
    )

    print(
        f"Overall Score      : {critic.overall_score:.1f}/10"
    )

    print(
        f"Structure Score    : {critic.structure_score:.1f}/10"
    )

    print(
        f"Clarity Score      : {critic.clarity_score:.1f}/10"
    )

    print(
        f"Research Quality   : {critic.research_quality_score:.1f}/10"
    )

    print(
        f"Completeness Score : {critic.completeness_score:.1f}/10"
    )

    print(
        f"Approved           : {critic.approved}"
    )

    print("\nStrengths:")

    for strength in critic.strengths:

        print(
            f"• {strength}"
        )

    print("\nWeaknesses:")

    for weakness in critic.weaknesses:

        print(
            f"• {weakness}"
        )

    print("\nImprovement Suggestions:")

    for suggestion in critic.improvement_suggestions:

        print(
            f"• {suggestion}"
        )

    print("\n")

    print("=" * 80)
    print("Research Completed Successfully".center(80))
    print("=" * 80)


if __name__ == "__main__":

    main()