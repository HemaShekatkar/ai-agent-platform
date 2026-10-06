from dataclasses import dataclass
from enum import Enum


class Capability(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class LatencyTier(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class CostTier(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PII = "pii"


@dataclass
class ModelConfig:
    name: str
    capability: Capability
    latency: LatencyTier
    cost: CostTier
    expected_latency_ms: int
    allowed_data: set[DataClassification]


@dataclass
class ModelRequest:
    task: str
    minimum_capability: Capability
    latency_requirement: LatencyTier
    cost_tier: CostTier
    timeout_ms: int
    data_classification: DataClassification


@dataclass
class RoutingPolicy:
    latency_weight: float
    cost_weight: float
    # Recorded on each GatewayTrace so routing decisions are attributable
    version: str = "v1"
