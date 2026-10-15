class ToolNotFoundError(Exception):
    pass


class DuplicateToolError(Exception):
    pass


class ToolAuthorizationError(Exception):
    pass


class ToolInputValidationError(Exception):
    pass


class ToolApprovalRequiredError(Exception):
    pass


class ToolApprovalRejectedError(Exception):
    pass


class ToolExecutionError(Exception):
    pass


class TransientToolError(ToolExecutionError):
    pass


class PermanentToolError(ToolExecutionError):
    pass
