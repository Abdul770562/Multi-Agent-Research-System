from graph.workflow import (
    build_graph
)


def main():

    

    graph = build_graph()

    initial_state = {
        "query":
        input(
            "Enter research topic: "
        ),

        "planning_output":
        None,

        "research_outputs":
        [],

        "validation_output":
        None,

        "writer_output":
        None,

        "critic_output":
        None,

        "revision_count": 
        0,
        "previous_critic_score": 
        0.0,
        "report_versions": 
        []
    }

    result = graph.invoke(
        initial_state
    )

    print("\n")

    print("=" * 80)

    print("\nFINAL STATE:\n")
    print(result.keys())

    print("\nWRITER OUTPUT:")
    print(result.get("writer_output"))

    print("\nCRITIC OUTPUT:")
    print(result.get("critic_output"))

    # print(
    #     result["writer_output"].report
    # )

    writer_output = result.get(
    "writer_output"
    )

    if writer_output:

        print(
            writer_output.report
        )

    else:

        print(
            "No report generated."
        )

    print("=" * 80)

    print("\nCritic Review:\n")

    print(
        result["critic_output"]
    )


if __name__ == "__main__":
    main()