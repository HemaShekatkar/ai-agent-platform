from app.agents.multi_agent_models import ReviewEvidence


def is_review_stale(
    review: ReviewEvidence,
    current_policy_version: str,
    current_document_version: str,
    current_jurisdiction: str,
) -> bool:

    return (
        review.policy_version != current_policy_version
        or review.document_version != current_document_version
        or review.jurisdiction != current_jurisdiction
    )


def get_reviews_to_rerun(
    reviews: dict[str, ReviewEvidence],
    current_versions: dict[str, dict[str, str]],
) -> list[str]:

    rerun = []

    for reviewer, review in reviews.items():

        versions = current_versions.get(reviewer)

        if versions is None:
            rerun.append(reviewer)
            continue

        if is_review_stale(
            review,
            current_policy_version=versions["policy_version"],
            current_document_version=versions["document_version"],
            current_jurisdiction=versions["jurisdiction"],
        ):
            rerun.append(reviewer)

    return rerun
