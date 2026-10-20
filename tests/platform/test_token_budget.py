
import pytest

from app.platform.token_budget import TokenBudgetManager


def test_reservation_reduces_available_budget():
    manager = TokenBudgetManager({"tenant-a": 1000})

    reservation = manager.reserve("tenant-a", 400)

    assert reservation is not None
    assert manager.available("tenant-a") == 600


def test_prevents_over_reservation():
    manager = TokenBudgetManager({"tenant-a": 1000})

    manager.reserve("tenant-a", 700)

    assert manager.reserve("tenant-a", 400) is None


def test_reconciliation_returns_unused_budget():
    manager = TokenBudgetManager({"tenant-a": 1000})

    reservation = manager.reserve("tenant-a", 500)
    manager.reconcile(reservation.reservation_id, 300)

    assert manager.available("tenant-a") == 700


def test_tenant_budgets_are_isolated():
    manager = TokenBudgetManager({
        "tenant-a": 1000,
        "tenant-b": 1000,
    })

    manager.reserve("tenant-a", 900)

    assert manager.available("tenant-b") == 1000


def test_cancel_releases_reservation():
    manager = TokenBudgetManager({"tenant-a": 1000})

    reservation = manager.reserve("tenant-a", 400)
    manager.cancel(reservation.reservation_id)

    assert manager.available("tenant-a") == 1000


def test_actual_usage_cannot_exceed_reservation():
    manager = TokenBudgetManager({"tenant-a": 1000})

    reservation = manager.reserve("tenant-a", 400)

    with pytest.raises(ValueError):
        manager.reconcile(reservation.reservation_id, 500)

    assert manager.available("tenant-a") == 600


def test_unknown_tenant_rejected():
    manager = TokenBudgetManager({"tenant-a": 1000})

    with pytest.raises(ValueError):
        manager.reserve("tenant-b", 100)
