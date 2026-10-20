
from dataclasses import dataclass


@dataclass(frozen=True)
class TenantQuota:
    tenant_id: str
    max_concurrent: int


class TenantAdmissionController:
    def __init__(
        self,
        global_limit: int,
        quotas: list[TenantQuota],
    ):
        if global_limit <= 0:
            raise ValueError("global_limit must be positive")

        self.global_limit = global_limit
        self.quotas = {}

        for quota in quotas:
            if quota.max_concurrent <= 0:
                raise ValueError("tenant quota must be positive")
            if quota.tenant_id in self.quotas:
                raise ValueError("duplicate tenant quota")
            self.quotas[quota.tenant_id] = quota.max_concurrent

        self.active_by_tenant = {
            tenant_id: 0 for tenant_id in self.quotas
        }
        self.global_active = 0

    def acquire(self, tenant_id: str) -> bool:
        if tenant_id not in self.quotas:
            return False

        if self.global_active >= self.global_limit:
            return False

        if (
            self.active_by_tenant[tenant_id]
            >= self.quotas[tenant_id]
        ):
            return False

        self.active_by_tenant[tenant_id] += 1
        self.global_active += 1
        return True

    def release(self, tenant_id: str) -> None:
        if tenant_id not in self.quotas:
            raise ValueError("unknown tenant")

        if self.active_by_tenant[tenant_id] <= 0:
            raise ValueError("no active request to release")

        self.active_by_tenant[tenant_id] -= 1
        self.global_active -= 1
