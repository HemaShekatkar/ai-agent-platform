from abc import ABC, abstractmethod

from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError

from app.workflows.models import (
    ToolWorkflowRequest,
    ToolWorkflowResult,
)
from app.workflows.tool_execution_workflow import ToolExecutionWorkflow


class ToolWorkflowRunner(ABC):

    @abstractmethod
    async def start(
        self,
        request: ToolWorkflowRequest,
    ) -> str:
        """Start or reconnect to a workflow. Return its ID."""
        ...

    @abstractmethod
    async def is_complete(
        self,
        workflow_id: str,
    ) -> bool:
        """Check whether the workflow has reached a terminal state."""
        ...

    @abstractmethod
    async def get_result(
        self,
        workflow_id: str,
    ) -> ToolWorkflowResult:
        """Retrieve the completed workflow result."""
        ...


class TemporalToolWorkflowRunner(ToolWorkflowRunner):

    def __init__(self, client: Client, task_queue: str):
        self.client = client
        self.task_queue = task_queue

    async def start(
        self,
        request: ToolWorkflowRequest,
    ) -> str:

        workflow_id = f"tool-{request.execution_id}"

        try:
            await self.client.start_workflow(
                ToolExecutionWorkflow.run,
                request,
                id=workflow_id,
                task_queue=self.task_queue,
            )
        except WorkflowAlreadyStartedError:
            pass

        return workflow_id

    async def is_complete(self, workflow_id: str) -> bool:
        handle = self.client.get_workflow_handle(workflow_id)
        description = await handle.describe()

        return description.status.name in {
            "COMPLETED",
            "FAILED",
            "CANCELED",
            "TERMINATED",
            "TIMED_OUT",
        }

    async def get_result(
        self,
        workflow_id: str,
    ) -> ToolWorkflowResult:
        handle = self.client.get_workflow_handle(
            workflow_id,
            result_type=ToolWorkflowResult,
        )
        return await handle.result()
