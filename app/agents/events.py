from dataclasses import dataclass
from enum import Enum


class WorkflowEventType(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class WorkflowCompletionEvent:
    event_id: str
    agent_run_id: str
    workflow_id: str
    event_type: WorkflowEventType
