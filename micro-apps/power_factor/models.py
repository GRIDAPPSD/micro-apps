from dataclasses import dataclass, field, astuple
import cimgraph.data_profile.rc4_2021 as cim_dp
import numpy as np


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
class SimulationMeasurement:
    measurement_mrid: str
    value: int | None = None
    angle: float | None = None
    magnitude: float | None = None


@dataclass
class MeasurementInfo:
    mrid: str
    phase: cim_dp.SinglePhaseKind


def convert_va(val: SimulationMeasurement) -> ComplexPower:
    if val.angle is None or val.magnitude is None:
        return ComplexPower()

    mag = float(val.magnitude)
    ang = float(val.angle)

    return ComplexPower(
        real=mag * np.cos(np.deg2rad(ang)),
        imag=mag * np.sin(np.deg2rad(ang))
    )


def find_nearest(array: np.array, value: float) -> int:
    return (np.abs(array - value)).argmin()


def convert_pnv(info: MeasurementInfo, val: SimulationMeasurement) -> PhasePower:
    if val.angle is None or val.magnitude is None:
        return PhasePower()

    mag = float(val.magnitude)
    ang = float(val.angle)
    print("ang 1: ", ang)
    phases = np.array([0.0, 120.0, -120.0])
    phase_id = find_nearest(phases, ang)
    ang -= phases[phase_id]
    print("ang 2: ", ang)
    phase = ComplexPower(
        real=mag * np.cos(np.deg2rad(ang)),
        imag=mag * np.sin(np.deg2rad(ang))
    )

    power = PhasePower()
    if info.phase == cim_dp.SinglePhaseKind.A:
        power.a = phase
        return power

    if info.phase == cim_dp.SinglePhaseKind.B:
        power.b = phase
        return power

    if info.phase == cim_dp.SinglePhaseKind.C:
        power.c = phase
        return power

    if phase_id == 0:
        power.a = phase
        return power

    if phase_id == 1:
        power.b = phase
        return power

    if phase_id == 2:
        power.c = phase
        return power

    print("conversion failed: ", info, val)

    return PhasePower()


@dataclass
class Compensators:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements: dict[PhasePower] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    sections_attribute: str = "ShuntCompensator.sections"


@dataclass
class PowerElectronics:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements: dict[PhasePower] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    p_attribute: str = "PowerElectronicsConnection.p"
    q_attribute: str = "PowerElectronicsConnection.q"


@dataclass
class Generators:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements: dict[PhasePower] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
    p_attribute: str = "RotatingMachine.p"
    q_attribute: str = "RotatingMachine.q"


@dataclass
class Consumers:
    ratings: dict[PhasePower] = field(default_factory=dict)
    measurements: dict[PhasePower] = field(default_factory=dict)
    measurement_map: dict[MeasurementInfo] = field(default_factory=dict)
