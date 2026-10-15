# User-facing outcome messages, shared by the graph and resumption so the
# same agent state always reads the same way however it was reached

ACTION_AWAITING_COMPLETION = "Action is awaiting completion."
ACTION_COMPLETED = "Action completed successfully."


def action_ended(status: str) -> str:
    return f"Action ended with status: {status}"
