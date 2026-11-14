import math
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0
    ref: str = None

    def dist(self, other):
        dx, dy, dz = abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z)
        return dx + dy + dz, math.sqrt(dx * dx + dy * dy + dz * dz)

@dataclass
class Location:
    id: str
    location: 'Point' = None
    health_factor: float = 1.0
    zone: str = None
    thermal_class: str = None

@dataclass
class Interface:
    name: str
    cost_per_meter: float
    latency_per_meter: float
    weight_per_meter: float
    capacity: float = float('inf')
    port_cost: float = 0.0
    hop_latency_ms: float = 0.0
    frame_kbit: float = float('inf')
    max_length_m: float = float('inf')

@dataclass
class Sensor:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    rate_hz: float = 10.0

@dataclass
class Actuator:
    id: str
    type: str
    interface: str
    volume: float
    location: 'Point' = None
    rate_hz: float = 10.0

@dataclass
class SoftwareComponent:
    id: str
    domain: str
    cpu_req: int
    ram_req: int
    rom_req: int
    asil_req: int
    hw_required: list
    interface_required: list = None
    sensors: list = None
    actuators: list = None
    redundant_with: str = None
    rate_hz: float = 10.0
    msg_kbit: float = None
    tops_req: float = 0.0
    fail_operational: bool = None
    allowed_cells: list = None

    def __post_init__(self):
        if self.interface_required is None:
            self.interface_required = []
        if self.sensors is None:
            self.sensors = []
        if self.actuators is None:
            self.actuators = []

@dataclass
class CandidateECU:
    id: str
    cpu_cap: int
    ram_cap: int
    rom_cap: int
    max_partitions: int
    cost: int
    type: str
    asil_level: int
    hw_offered: list
    interface_offered: list
    location: 'Point' = None
    system_overhead_percent: float = 0.0
    partition_reservation_percent: float = 0.0
