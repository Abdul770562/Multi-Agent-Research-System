import re

from langchain_core.messages import HumanMessage

from config.llm import llm

from models.outputs import (
    WriterOutput,
    ResearchOutput,
)

from models.outputs import (
    GapAnalysis,
)

from models.state import (
    ResearchState,
)

from utils.research import (
    merge_research_outputs,
)



REVISION_PROMPT = """
You are a Senior Research Editor.

Your responsibility is NOT to rewrite the report.

Your responsibility is to improve ONLY the
parts of the report that are supported by
new research evidence.

You are given:

1. Original report

2. Complete research corpus

3. Research gaps

4. Critic weaknesses

5. Critic improvement suggestions

===============================

GROUNDING RULES

You may ONLY use information
contained in the supplied
research corpus.

Never invent:

- facts
- citations
- URLs
- organizations
- studies
- statistics
- percentages
- case studies

If evidence does not exist,
do not add it.

===============================

REVISION RULES

Preserve:

- report title
- structure
- headings
- references
- valid content

Improve ONLY the sections
identified by the Gap Analyzer.

Address every:

- weakness
- improvement suggestion

Expand sections ONLY if
supporting evidence exists.

Do not create a new references section.

Do not remove references.

Return the COMPLETE updated report.
"""

def build_research_context(
    research_outputs: list[ResearchOutput],
) -> str:
    """
    Builds a formatted research corpus
    for the reviser.
    """

    context = ""

    for item in research_outputs:

        findings = "\n".join(
            f"- {finding}"
            for finding in item.key_findings
        )

        sources = "\n".join(
            f"- {source}"
            for source in item.sources
        )

        context += f"""

==================================================

SEARCH QUERY

{item.search_query}

SUMMARY

{item.summary}

KEY FINDINGS

{findings}

SOURCES

{sources}

"""

    return context


def build_gap_context(
    gap_analysis: GapAnalysis,
) -> str:
    """
    Formats Gap Analyzer output for
    the reviser.
    """

    context = ""

    for gap in gap_analysis.gaps:

        context += f"""

----------------------------------------

TOPIC

{gap.topic}

EXPECTED SECTION

{gap.expected_section}

REASON

{gap.reason}

SEARCH QUERY

{gap.search_query}

PRIORITY

{gap.priority}

"""

    return context

def build_critic_context(
    state: ResearchState,
) -> str:
    """
    Formats critic feedback.
    """

    critic = state["critic_output"]

    weaknesses = "\n".join(
        f"- {item}"
        for item in critic.weaknesses
    )

    improvements = "\n".join(
        f"- {item}"
        for item in critic.improvement_suggestions
    )

    return f"""
OVERALL SCORE

{critic.overall_score}

----------------------------------------

WEAKNESSES

{weaknesses}

----------------------------------------

IMPROVEMENT SUGGESTIONS

{improvements}
"""

def get_allowed_sources(
    research_outputs: list[ResearchOutput],
) -> set[str]:
    """
    Collect every source URL that
    the reviser is allowed to use.
    """

    allowed = set()

    for item in research_outputs:
        allowed.update(item.sources)

    return allowed

def validate_grounding(
    report: str,
    allowed_sources: set[str],
) -> tuple[str, set[str]]:
    """
    Removes URLs that were not
    present in the supplied
    research corpus.
    """

    urls_found = set(
        re.findall(
            r"https?://[^\s)\]]+",
            report,
        )
    )

    invalid_urls = (
        urls_found - allowed_sources
    )

    for url in invalid_urls:

        report = report.replace(
            url,
            "[REMOVED_UNAUTHORIZED_SOURCE]",
        )

    return report, invalid_urls

def reviser_agent(
    state: ResearchState,
) -> ResearchState:
    """
    Improves the report using
    additional research collected
    after Gap Analysis.
    """

    writer_output = state[
        "writer_output"
    ]

    gap_analysis = state[
        "gap_analysis"
    ]

    # ----------------------------------
    # Merge Original + Additional Research
    # ----------------------------------

    merged_research = (
        merge_research_outputs(
            state["research_outputs"],
            state[
                "additional_research_outputs"
            ],
        )
    )

    # ----------------------------------
    # Build LLM Context
    # ----------------------------------

    research_context = (
        build_research_context(
            merged_research
        )
    )

    gap_context = (
        build_gap_context(
            gap_analysis
        )
    )

    critic_context = (
        build_critic_context(
            state
        )
    )

    # ----------------------------------
    # Allowed Sources
    # ----------------------------------

    allowed_sources = (
        get_allowed_sources(
            merged_research
        )
    )

    print(
        "\nStarting Revision..."
    )

    print(
        f"Research Blocks : "
        f"{len(merged_research)}"
    )

    print(
        f"Gaps            : "
        f"{len(gap_analysis.gaps)}"
    )

    print(
        f"Allowed Sources : "
        f"{len(allowed_sources)}"
    )

    structured_llm = (
        llm.with_structured_output(
            WriterOutput
        )
    )

    # ----------------------------------
    # Generate Revised Report
    # ----------------------------------

    try:

        revised_output = (
            structured_llm.invoke(
                [
                    HumanMessage(
                        content=f"""
{REVISION_PROMPT}

==================================================

CURRENT REPORT

{writer_output.report}

==================================================

COMPLETE RESEARCH CORPUS

{research_context}

==================================================

RESEARCH GAPS

{gap_context}

==================================================

CRITIC FEEDBACK

{critic_context}

==================================================

Revise ONLY the sections
identified in the Gap Analysis.

Keep all valid information.

Use ONLY evidence present
inside the research corpus.

Return the complete
improved report.
"""
                    )
                ]
            )
        )

    except Exception as e:

        raise RuntimeError(
            f"Revision failed: {e}"
        )
    
    revised_output.references = (
        writer_output.references
    )

    (
        revised_output.report,
        invalid_urls,
    ) = validate_grounding(
        revised_output.report,
        allowed_sources,
    )

    if invalid_urls:

        print(
            "\nWARNING:"
        )

        print(
            "Reviser introduced "
            "unauthorized URLs."
        )

        for url in invalid_urls:

            print(
                f" - {url}"
            )

        print()

    old_words = len(writer_output.report.split())
    new_words = len(revised_output.report.split())

    print(
        f"Word Count: "
        f"{old_words} -> {new_words}"
    )

    print(
        f"Removed "
        f"{len(invalid_urls)} "
        f"unauthorized URLs."
    )

    # ----------------------------------
    # Update State
    # ----------------------------------

    state[
        "writer_output"
    ] = revised_output

    state[
        "revision_count"
    ] += 1

    state[
        "report_versions"
    ].append(
        revised_output.report
    )

    previous_score = state.get(
        "previous_critic_score",
        0.0,
    )

    previous_words = len(
        writer_output.report.split()
    )

    revised_words = len(
        revised_output.report.split()
    )

    print()

    print("=" * 60)
    print("Revision Summary")
    print("=" * 60)

    print(
        f"Revision Number : "
        f"{state['revision_count']}"
    )

    print(
        f"Previous Critic Score : "
        f"{previous_score}"
    )

    print(
        f"Word Count : "
        f"{previous_words}"
        f" -> "
        f"{revised_words}"
    )

    print(
        f"Research Blocks Used : "
        f"{len(merged_research)}"
    )

    print(
        f"Gaps Addressed : "
        f"{len(gap_analysis.gaps)}"
    )

    print(
        f"Report Versions : "
        f"{len(state['report_versions'])}"
    )

    print(
        "=" * 60
    )

    return state