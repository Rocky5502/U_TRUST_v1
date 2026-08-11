from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    name: str
    roles: dict[str, str]
    edges: tuple[tuple[str, str], ...]


def chain4() -> Topology:
    return Topology(
        name="chain4",
        roles={
            "planner": "Decompose the immutable user goal into authorized subtasks.",
            "worker": "Use tools and evidence only to execute the assigned subtask.",
            "verifier": "Check whether worker output serves the immutable user goal.",
            "executor": "Issue only the final action authorized by the verified plan.",
        },
        edges=(("planner", "worker"), ("worker", "verifier"), ("verifier", "executor")),
    )


def star4() -> Topology:
    return Topology(
        name="star4",
        roles={
            "planner": "Create an authorized task decomposition from the immutable user goal.",
            "worker": "Collect evidence or perform the assigned tool task.",
            "verifier": "Independently assess goal consistency and authorization.",
            "coordinator": "Combine approved worker/verifier information and issue the final action.",
        },
        edges=(("planner", "worker"), ("planner", "verifier"), ("worker", "coordinator"), ("verifier", "coordinator")),
    )


def get_topology(name: str) -> Topology:
    if name == "chain4":
        return chain4()
    if name == "star4":
        return star4()
    raise ValueError(f"Unknown topology: {name}")
