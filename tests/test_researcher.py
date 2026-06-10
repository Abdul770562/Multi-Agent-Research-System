# tests/test_researcher.py

from agents.planner import (
    planner_agent
)

from agents.researcher import (
    researcher_agent
)

state = {
    "query":
    "Artificial Intelligence in Healthcare",

    "planning_output":
    None,

    "research_outputs":
    [],

    "validation_output":
    None,

    "writer_output":
    None,

    "critic_output":
    None
}

state = planner_agent(
    state
)

state = researcher_agent(
    state
)

for output in state[
    "research_outputs"
]:
    print(output)