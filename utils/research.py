from models.outputs import ResearchOutput


def merge_research_outputs(
    original: list[ResearchOutput],
    additional: list[ResearchOutput],
) -> list[ResearchOutput]:
    """
    Merge original and additional research outputs.

    Rules
    -----
    1. Preserve original research order.
    2. Ignore duplicate search queries.
    3. Remove duplicate source URLs.
    """

    merged: list[ResearchOutput] = []

    seen_queries: set[str] = set()

    for output in original + additional:

        normalized_query = (
            output.search_query
            .strip()
            .lower()
        )

        if normalized_query in seen_queries:
            continue

        seen_queries.add(
            normalized_query
        )

        # Remove duplicate URLs while
        # preserving order.
        output.sources = list(
            dict.fromkeys(output.sources)
        )

        merged.append(output)

    return merged