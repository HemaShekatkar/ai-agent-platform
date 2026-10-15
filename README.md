# Production AI Agent Platform

A reference implementation for building reliable production AI agents with governed model routing, RAG, context budgeting, structured generation, groundedness validation, bounded recovery, evaluations, and end-to-end observability.

## Architecture

```
        ┌─────────────────────────────────────────────────────────────────┐
        │ Observability                                                   │
        │ traces • metrics • evals • versions                             │
        │                                                                 │
        │                ┌───────────────────────┐                        │
        │                │     User / Client     │                        │
        │                └───────────┬───────────┘                        │
        │                            │                                    │
        │                            ▼                                    │
        │                ┌───────────────────────┐                        │
        │                │      Agent Layer      │                        │
        │                │ orchestration/policy  │                        │
        │                └───────────┬───────────┘                        │
        │                            │                                    │
        │        ┌───────────────────┼────────────────────┐               │
        │        │                   │                    │               │
        │        ▼                   ▼                    ▼               │
        │ ┌────────────────┐ ┌───────────────┐   ┌───────────────┐        │
        │ │   Retrieval    │ │ Model Gateway │   │ Tools/Actions │        │
        │ │ hybrid search  │ │ routing       │   │ APIs/workflow │        │
        │ │ ACL/filtering  │ │ fallback      │   │ systems       │        │
        │ │ reranking      │ │ governance    │   └───────────────┘        │
        │ └───────┬────────┘ └───────┬───────┘                            │
        │         │                  │                                    │
        │         ▼                  ▼                                    │
        │ ┌────────────────┐  ┌──────────────┐                            │
        │ │Context Builder │  │     LLM      │                            │
        │ │token budgeting │  │ structured   │                            │
        │ │evidence        │  │ output       │                            │
        │ └───────┬────────┘  └──────┬───────┘                            │
        │         │                  │                                    │
        │         └────────┬─────────┘                                    │
        │                  ▼                                              │
        │        ┌───────────────────┐                                    │
        │        │    Validation     │                                    │
        │        │ citations         │                                    │
        │        │ groundedness      │                                    │
        │        │ business rules    │                                    │
        │        └─────────┬─────────┘                                    │
        │                  │                                              │
        │         ┌────────┴────────┐                                     │
        │         │                 │                                     │
        │       PASS              FAIL                                    │
        │         │                 │                                     │
        │         ▼                 ▼                                     │
        │    Decision        bounded retry                                │
        │                          │                                      │
        │                      still fails                                │
        │                          │                                      │
        │                          ▼                                      │
        │                    HUMAN_REVIEW                                 │
        └─────────────────────────────────────────────────────────────────┘
```

## What's implemented

The diagram shows the target architecture. This repository currently implements:

| Component | Where | Notes |
|---|---|---|
| Agent orchestration | `app/agents/onboarding_agent.py` | retrieve → build context → generate → validate → one bounded retry → `HUMAN_REVIEW` |
| Retrieval | `app/retrieval/retriever.py` | in-memory lexical retrieval with ACL, region, document-type and effective-date filtering |
| Context builder | `app/retrieval/context.py` | per-request token budgeting with budget-exclusion tracking |
| Model gateway | `app/gateway.py`, `app/routing/` | capability and data-classification routing, weighted ranking, transient-failure fallback, latency budgets |
| Structured generation | `app/gateway.py`, `app/models/` | typed Pydantic outputs via `generate_structured` |
| Validation | `app/agents/onboarding.py`, `app/evals/groundedness.py` | citation validation and an LLM-judge groundedness check |
| Evaluations | `app/evals/` | exact-match and LLM-judge evaluators, retrieval metrics (recall@k, precision@k, MRR), release gating |
| Observability | `app/observability/` | spans with parent/child structure and a versioned attribute convention for each stage |

Not yet implemented: hybrid search, reranking, and tools/actions.

All examples and test data are synthetic. Partner names, policies and documents are fictional.

## Getting started

Requires Python 3.14.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run the tests

The test suite uses fakes for every model provider and needs no API key. Workflow tests start a local Temporal dev server; the first run downloads it (cached afterwards), and nothing else leaves your machine.

```bash
pytest
```

### PostgreSQL integration tests (optional)

The event-inbox and agent-lock tests in `tests/agents/test_postgres_*.py` run against a real PostgreSQL database and are skipped unless `TEST_DATABASE_URL` is set, either exported in your shell or as a line in your git-ignored `.env` (see `.env.example`). Point it at a disposable database; the tests apply every migration in `db/` and empty the tables before each test.

```bash
createdb agent_platform_test
TEST_DATABASE_URL=postgresql://localhost:5432/agent_platform_test pytest
```

### Run the live demo (optional)

The demo calls a real model and needs an OpenAI API key.

```bash
cp .env.example .env    # then set OPENAI_API_KEY in .env
python -m app.main
```

`.env` is ignored by git. Never commit real credentials.

## License

MIT. See [LICENSE](LICENSE).
