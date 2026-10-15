from app.tools.fingerprint import action_fingerprint


def test_fingerprint_ignores_dictionary_key_order():
    # Key order is incidental (e.g. it can change after a JSON round trip
    # through Temporal); the same action must keep the same fingerprint
    first = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123", "region": "US"},
    )
    second = action_fingerprint(
        "approve_partner",
        {"region": "US", "partner_id": "partner-123"},
    )

    assert first == second


def test_fingerprint_ignores_nested_key_order():
    first = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123", "terms": {"tier": "gold", "days": 30}},
    )
    second = action_fingerprint(
        "approve_partner",
        {"terms": {"days": 30, "tier": "gold"}, "partner_id": "partner-123"},
    )

    assert first == second


def test_fingerprint_changes_with_arguments():
    first = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123"},
    )
    second = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-456"},
    )

    assert first != second


def test_fingerprint_changes_with_tool_name():
    first = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123"},
    )
    second = action_fingerprint(
        "delete_partner",
        {"partner_id": "partner-123"},
    )

    assert first != second


def test_fingerprint_is_stable_hex_digest():
    fingerprint = action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123"},
    )

    # SHA-256 hex: safe to store and compare as a plain string
    assert len(fingerprint) == 64
    assert fingerprint == action_fingerprint(
        "approve_partner",
        {"partner_id": "partner-123"},
    )
