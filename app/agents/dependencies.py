
from dataclasses import dataclass

from app.agents.models import AgentAction


@dataclass(frozen=True)
class WorkflowGraph:
    version: str
    dependencies: dict[AgentAction, frozenset[AgentAction]]

    def prerequisites_for(
        self, action: AgentAction
    ) -> frozenset[AgentAction]:
        return self.dependencies.get(action, frozenset())


class DependencyValidator:
    def __init__(self, graph: WorkflowGraph):
        self.graph = graph

    def can_execute(
        self,
        action: AgentAction,
        completed: set[AgentAction],
    ) -> bool:
        required = self.graph.prerequisites_for(action)
        return required.issubset(completed)

    def missing_prerequisites(
        self,
        action: AgentAction,
        completed: set[AgentAction],
    ) -> frozenset[AgentAction]:
        required = self.graph.prerequisites_for(action)
        return required - completed
