import math
from collections.abc import Callable
from dataclasses import dataclass

from app.retrieval.models import RetrievalResult


SECTION_SEPARATOR = "\n\n"


def estimate_tokens(text: str) -> int:
    # Rough heuristic (~4 chars per token); pass a real tokenizer for accuracy
    return math.ceil(len(text) / 4)


@dataclass
class ContextBudget:
    model_context_limit: int
    system_prompt_tokens: int
    history_tokens: int
    tool_definition_tokens: int
    question_tokens: int
    output_reservation_tokens: int

    @property
    def retrieval_tokens(self) -> int:
        return (
            self.model_context_limit
            - self.system_prompt_tokens
            - self.history_tokens
            - self.tool_definition_tokens
            - self.question_tokens
            - self.output_reservation_tokens
        )


@dataclass
class ExcludedChunk:
    chunk_id: str
    reason: str


@dataclass
class ContextItem:
    chunk_id: str
    text: str


@dataclass
class ContextResult:
    text: str
    items: list[ContextItem]
    included_chunk_ids: list[str]
    excluded_chunks: list[ExcludedChunk]
    budget_tokens: int
    used_tokens: int


class ContextBuilder:

    def __init__(
        self,
        count_tokens: Callable[[str], int] = estimate_tokens,
    ):
        self.count_tokens = count_tokens

    def build(
        self,
        results: list[RetrievalResult],
        budget: ContextBudget,
    ) -> ContextResult:

        # The budget is per request: question and history size vary per call
        if budget.retrieval_tokens <= 0:
            raise ValueError(
                "No token budget available for retrieval context"
            )

        max_tokens = budget.retrieval_tokens

        included = []
        excluded = []
        sections = []
        items = []

        used_tokens = 0

        # Results are expected in ranked order, so higher-scoring chunks
        # claim budget first; a smaller later chunk may still fit
        for result in results:
            chunk = result.chunk

            section = (
                f"[chunk_id={chunk.id}]\n"
                f"{chunk.text}"
            )

            # Count the joined text, since separators cost tokens and
            # token counts are not strictly additive across strings
            candidate_tokens = self.count_tokens(
                SECTION_SEPARATOR.join([*sections, section])
            )

            if candidate_tokens > max_tokens:
                excluded.append(
                    ExcludedChunk(
                        chunk_id=chunk.id,
                        reason="BUDGET_EXCEEDED",
                    )
                )
                continue

            sections.append(section)
            included.append(chunk.id)
            used_tokens = candidate_tokens
            items.append(
                ContextItem(
                    chunk_id=chunk.id,
                    text=chunk.text,
                )
            )

        return ContextResult(
            text=SECTION_SEPARATOR.join(sections),
            items=items,
            included_chunk_ids=included,
            excluded_chunks=excluded,
            budget_tokens=max_tokens,
            used_tokens=used_tokens,
        )
