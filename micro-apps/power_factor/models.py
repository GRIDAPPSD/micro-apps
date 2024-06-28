from dataclasses import dataclass, field, astuple
import cimgraph.data_profile.rc4_2021 as cim_dp
import numpy as np


@dataclass
class PhaseSwitch:
    a: bool = False
    b: bool = False
    c: bool = False

    def __array__(self):
        return np.array(astuple(self))

    def __len__(self):
        return astuple(self).__len__()

    def __getitem__(self, item):
        return astuple(self).__getitem__(item)


@dataclass
class ComplexPower:
    real: float = 0.0
    imag: float = 0.0

    def __array__(self):
        return np.array(astuple(self))

    def __len__(self):
        return astuple(self).__len__()

    def __getitem__(self, item):
        return astuple(self).__getitem__(item)


@dataclass
class PhasePower:
    a: ComplexPower = field(default=ComplexPower())
    b: ComplexPower = field(default=ComplexPower())
    c: ComplexPower = field(default=ComplexPower())

    def __array__(self):
        return np.array(astuple(self))

    def __len__(self):
        return astuple(self).__len__()

    def __getitem__(self, item):
        return astuple(self).__getitem__(item)


@dataclass
class PhaseMap:
    s1: cim_dp.PhaseCode | None = None
    s2: cim_dp.PhaseCode | None = None


@dataclass
class SimulationMeasurement:
    measurement_mrid: str
    value: int | None = None
    angle: float | None = None
    magnitude: float | None = None


@dataclass
class MeasurementInfo:
    mrid: str
    value_type: str
    phase: cim_dp.PhaseCode


def update_va(
        info: MeasurementInfo,
        val: SimulationMeasurement,
        power: PhasePower) -> PhasePower:
    if val.angle is None or val.magnitude is None:
        return power

    mag = float(val.magnitude)
    ang = float(val.angle)
    val = ComplexPower(
        real=mag * np.cos(np.deg2rad(ang)),
        imag=mag * np.sin(np.deg2rad(ang))
    )

    if info.phase == cim_dp.PhaseCode.A:
        power.a = val
        return power

    if info.phase == cim_dp.PhaseCode.B:
        power.b = val
        return power

    if info.phase == cim_dp.PhaseCode.C:
        power.c = val
        return power

    return power


def find_nearest(array: np.array, value: float) -> int:
    return (np.abs(array - value)).argmin()


def update_pnv(
        info: MeasurementInfo,
        val: SimulationMeasurement,
        phase_map: PhaseMap) -> PhaseMap:
    if val.angle is None or val.magnitude is None:
        return phase_map

    ang = float(val.angle)
    phases = np.array([0.0, -120.0, 120.0])
    phase_id = find_nearest(phases, ang)

    if info.phase == cim_dp.PhaseCode.s1:
        if phase_id == 0:
            phase_map.s1 = cim_dp.PhaseCode.A
            return phase_map

        if phase_id == 1:
            phase_map.s1 = cim_dp.PhaseCode.B
            return phase_map

        if phase_id == 2:
            phase_map.s1 = cim_dp.PhaseCode.C
            return phase_map

    if info.phase == cim_dp.PhaseCode.s2:
        if phase_id == 0:
            phase_map.s2 = cim_dp.PhaseCode.A
            return phase_map

        if phase_id == 1:
            phase_map.s2 = cim_dp.PhaseCode.B
            return phase_map

        if phase_id == 2:
            phase_map.s2 = cim_dp.PhaseCode.C
            return phase_map

    return phase_map


@dataclass
class Compensators:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements_va: dict[PhasePower] = field(default_factory=dict)
    measurements_pnv: dict[PhaseMap] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    sections_attribute: str = "ShuntCompensator.sections"


@dataclass
class PowerElectronics:
    units: dict[str] = field(default_factory=dict)
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements_va: dict[PhasePower] = field(default_factory=dict)
    measurements_pnv: dict[PhaseMap] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    p_attribute: str = "PowerElectronicsConnection.p"
    q_attribute: str = "PowerElectronicsConnection.q"


@dataclass
class Generators:
    units: dict[str] = field(default_factory=dict)
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements_va: dict[PhasePower] = field(default_factory=dict)
    measurements_pnv: dict[PhaseMap] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    p_attribute: str = "RotatingMachine.p"
    q_attribute: str = "RotatingMachine.q"


@dataclass
class Switches:
    normal: dict[PhaseSwitch] = field(default_factory=dict)
    measurements_va: dict[PhaseSwitch] = field(default_factory=dict)
    measurements_pnv: dict[PhaseMap] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    open_attribute: str = "Switch.open"


@dataclass
class Consumers:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements_va: dict[PhasePower] = field(default_factory=dict)
    measurements_pnv: dict[PhaseMap] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
