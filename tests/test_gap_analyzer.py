from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.validator import validator_agent
from agents.writer import writer_agent
from agents.critic import critic_agent
from agents.gap_analyzer import gap_analyzer_agent


def main():

    state = {
        "query": "Artificial Intelligence in Healthcare",

        "planning_output": None,
        "research_outputs": [],

        "validation_output": None,

        "writer_output": None,
        "critic_output": None,

        "gap_analysis": None,

        "revision_count": 0,
        "previous_critic_score": 0.0,
        "report_versions": [],
    }

    print("\nRunning Planner...")
    state = planner_agent(state)

    print("\nRunning Researcher...")
    state = researcher_agent(state)

    print("\nRunning Validator...")
    state = validator_agent(state)

    print("\nRunning Writer...")
    state = writer_agent(state)

    print("\nRunning Critic...")
    state = critic_agent(state)

    print("\nRunning Gap Analyzer...")
    state = gap_analyzer_agent(state)

    gap_analysis = state["gap_analysis"]

    print("\n" + "=" * 80)
    print("GAP ANALYSIS")
    print("=" * 80)

    print(f"\nTotal Gaps: {len(gap_analysis.gaps)}")

    print("\nOverall Rationale:")
    print(gap_analysis.rationale)

    print("\nIdentified Gaps:\n")

    for i, gap in enumerate(gap_analysis.gaps, start=1):

        print(f"Gap #{i}")
        print(f"Priority          : P{gap.priority}")
        print(f"Topic             : {gap.topic}")
        print(f"Expected Section  : {gap.expected_section}")
        print(f"Reason            : {gap.reason}")
        print(f"Search Query      : {gap.search_query}")
        print("-" * 80)


if __name__ == "__main__":
    main()