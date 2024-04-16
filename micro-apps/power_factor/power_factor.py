import time
import os
import traceback
import logging
from pprint import pprint
import networkx as nx
import numpy as np
import cvxpy as cp
from pandas import DataFrame
import importlib
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from dataclasses import dataclass, asdict
import json
from typing import Dict, Tuple, NamedTuple
from cimgraph import utils
from cimgraph.databases import ConnectionParameters, BlazegraphConnection
from cimgraph.models import FeederModel
from cimgraph.data_profile.rc4_2021 import ACLineSegment
from cimgraph.data_profile.rc4_2021 import ACLineSegmentPhase
from cimgraph.data_profile.rc4_2021 import Terminal
from cimgraph.data_profile.rc4_2021 import ConnectivityNode
from cimgraph.data_profile.rc4_2021 import EnergyConsumer
from cimgraph.data_profile.rc4_2021 import EnergyConsumerPhase
from cimgraph.data_profile.rc4_2021 import PowerElectronicsConnection
from cimgraph.data_profile.rc4_2021 import PowerElectronicsConnectionPhase
from cimgraph.data_profile.rc4_2021 import LinearShuntCompensator
from cimgraph.data_profile.rc4_2021 import LinearShuntCompensatorPhase
from cimgraph.data_profile.rc4_2021 import TransformerTank
from cimgraph.data_profile.rc4_2021 import TransformerTankEnd
from cimgraph.data_profile.rc4_2021 import LoadBreakSwitch
from cimgraph.data_profile.rc4_2021 import PositionPoint
from cimgraph.data_profile.rc4_2021 import CoordinateSystem
from cimgraph.data_profile.rc4_2021 import Location
from cimgraph.data_profile.rc4_2021 import SinglePhaseKind
from cimgraph.data_profile.rc4_2021 import Measurement
from gridappsd import GridAPPSD, DifferenceBuilder
import gridappsd.utils
from gridappsd.simulation import Simulation
from gridappsd.simulation import PowerSystemConfig
from gridappsd.simulation import ApplicationConfig
from gridappsd.simulation import SimulationConfig
from gridappsd.simulation import SimulationArgs
from gridappsd.simulation import ModelCreationConfig
from gridappsd.simulation import ServiceConfig
from gridappsd import topics as t

IEEE123_APPS = "_E3D03A27-B988-4D79-BFAB-F9D37FB289F7"
IEEE13 = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
IEEE123 = "_C1C3E687-6FFD-C753-582B-632A27E28507"
IEEE123PV = "_A3BC35AA-01F6-478E-A7B1-8EA4598A685C"
SOURCE_BUS = '150r'
OUT_DIR = "outputs"
ROOT = os.getcwd()


logging.basicConfig(level=logging.DEBUG,
                    filename=f"{ROOT}/{OUT_DIR}/log.txt", filemode='w')
log = logging.getLogger(__name__)


class ControlAttributes(BaseModel):
    switch: str = "Switch.open"
    capacitor: str = "ShuntCompensator.sections"
    inverter_p: str = "PowerElectronicsConnection.p"
    inverter_q: str = "PowerElectronicsConnection.q"
    generator_p: str = "RotatingMachine.p"
    generator_q: str = "RotatingMachine.q"
    transformer: str = "TapChanger.step"


class PhaseMRID(NamedTuple):
    mrid: str
    phase: int


def generate_graph() -> nx.Graph:
    graph = nx.Graph()

    network.get_all_edges(cim.ACLineSegment)
    network.get_all_edges(cim.ConnectivityNode)
    network.get_all_edges(cim.Terminal)

    line: ACLineSegment
    for line in network.graph[ACLineSegment].values():
        terminals = line.Terminals
        bus_1 = terminals[0].ConnectivityNode.name
        bus_2 = terminals[1].ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=float(line.length))

    xfmr: TransformerTank
    for xfmr in network.graph[TransformerTank].values():
        ends = xfmr.TransformerTankEnds
        bus_1 = ends[0].Terminal.ConnectivityNode.name
        bus_2 = ends[1].Terminal.ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=0.0)

    switch: LoadBreakSwitch
    for switch in network.graph[LoadBreakSwitch].values():
        terminals = switch.Terminals
        bus_1 = terminals[0].ConnectivityNode.name
        bus_2 = terminals[1].ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=0.0)

    return graph


def get_graph_positions() -> dict:
    loc = utils.get_all_bus_locations(network)
    return {d['name']: [float(d['x']), float(d['y'])]
            for d in loc.values()}


def get_compensators(mrids: dict) -> Tuple[dict, dict]:
    compensators = {}
    if LinearShuntCompensator not in network.graph:
        return mrids, compensators

    shunt: LinearShuntCompensator
    for shunt in network.graph[LinearShuntCompensator].values():

        mrid = shunt.mRID
        nom_v = float(shunt.nomU)
        susceptance = -float(shunt.bPerSection) * nom_v**2
        power = np.zeros((3, 2))

        if not shunt.ShuntCompensatorPhase:
            measurement = Measurement
            for measurement in shunt.Measurements:
                measurement_mrid = measurement.mRID
                mrids[measurement_mrid] = PhaseMRID(mrid, 3)
                power[0] = [0, susceptance]
                power[1] = [0, susceptance]
                power[2] = [0, susceptance]

        else:
            phase: LinearShuntCompensatorPhase
            for phase in shunt.ShuntCompensatorPhase:
                measurement = Measurement
                for measurement in phase.Measurements:
                    measurement_mrid = measurement.mRID
                    if phase.phase == SinglePhaseKind.A:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 0)
                        power[0] = [0, susceptance]
                    if phase.phase == SinglePhaseKind.B:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 1)
                        power[1] = [0, susceptance]
                    if phase.phase == SinglePhaseKind.C:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 2)
                        power[2] = [0, susceptance]
        compensators[mrid] = power

    return mrids, compensators


def get_consumers(mrids: dict) -> Tuple[dict, dict]:
    consumers = {}
    if EnergyConsumer not in network.graph:
        return mrids, consumers

    load: EnergyConsumer
    for load in network.graph[EnergyConsumer].values():
        mrid = load.mRID
        power = np.zeros((3, 2))
        if not load.EnergyConsumerPhase:
            measurement = Measurement
            for measurement in load.Measurements:
                measurement_mrid = measurement.mRID
                mrids[measurement_mrid] = PhaseMRID(mrid, 3)
                power[0] = [load.p, load.q]
                power[1] = [load.p, load.q]
                power[2] = [load.p, load.q]

        else:
            phases = load.EnergyConsumerPhase
            phase: EnergyConsumerPhase
            for phase in phases:
                measurement = Measurement
                for measurement in phase.Measurements:
                    measurement_mrid = measurement.mRID
                    if phase.phase == SinglePhaseKind.A:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 0)
                        power[0] = [phase.p, phase.q]
                    if phase.phase == SinglePhaseKind.B:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 1)
                        power[1] = [phase.p, phase.q]
                    if phase.phase == SinglePhaseKind.C:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 2)
                        power[2] = [phase.p, phase.q]
        consumers[mrid] = power
    return mrids, consumers


def get_power_electronics(mrids: dict) -> Tuple[dict, dict]:
    electronics = {}
    if PowerElectronicsConnection not in network.graph:
        return mrids, electronics

    pec: PowerElectronicsConnection
    for pec in network.graph[PowerElectronicsConnection].values():

        mrid = pec.mRID
        power = np.zeros((3, 2))

        if not pec.PowerElectronicsConnectionPhases:
            measurement = Measurement
            for measurement in pec.Measurements:
                measurement_mrid = measurement.mRID
                mrids[measurement_mrid] = PhaseMRID(mrid, 3)
                power[0] = [pec.ratedS, pec.ratedS]
                power[1] = [pec.ratedS, pec.ratedS]
                power[2] = [pec.ratedS, pec.ratedS]

        else:
            phases = pec.PowerElectronicsConnectionPhases
            phase: PowerElectronicsConnectionPhase
            for phase in phases:
                measurement = Measurement
                for measurement in phase.Measurements:
                    measurement_mrid = measurement.mRID
                    if phase.phase == SinglePhaseKind.A:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 0)
                        power[0] = [pec.ratedS, pec.ratedS]
                    if phase.phase == SinglePhaseKind.B:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 1)
                        power[1] = [pec.ratedS, pec.ratedS]
                    if phase.phase == SinglePhaseKind.C:
                        mrids[measurement_mrid] = PhaseMRID(mrid, 2)
                        power[2] = [pec.ratedS, pec.ratedS]
        electronics[mrid] = power

    return mrids, electronics


def map_power_electronics() -> dict:
    map = {}
    if not network.graph[PowerElectronicsConnection]:
        return map

    pec: PowerElectronicsConnection
    for pec in network.graph[PowerElectronicsConnection].values():
        name = pec.mRID
        bus = pec.Terminals[0].ConnectivityNode.name
        map[name] = bus
    return map


def dist_matrix(ders: dict, dists: dict) -> np.ndarray:
    phases = 3
    map = [dists[bus] for der, bus in ders.items()]
    map = map / np.linalg.norm(map)
    mat = np.array([map,]*phases).transpose()
    return mat


def dict_to_array(d: dict) -> np.ndarray:
    results = d.values()
    data = list(results)
    return np.array(data)


class PowerFactor(object):
    def __init__(self, gapps: GridAPPSD, sim: Simulation):
        self.gapps = gapps
        self.mrids = {}
        self.mrids, self.compensators = get_compensators(self.mrids)
        pprint(self.compensators)
        self.mrids, self.consumers = get_consumers(self.mrids)
        pprint(self.consumers)
        self.mrids, self.electronics = get_power_electronics(self.mrids)
        pprint(self.electronics)
        exit()

        # graph = generate_graph()
        # dist = nx.shortest_path_length(graph, source=SOURCE_BUS)
        # electronics_map = map_power_electronics()
        # self.dist = dist_matrix(electronics_map, dist)
        # plot_graph(graph, dist)

        sim.add_onmeasurement_callback(self.on_measurement)
        sim.start_simulation()

    def on_measurement(self, sim: Simulation, timestamp: dict, measurements: dict) -> None:
        print(timestamp)

        for k, v in measurements.items():

            if k in self.mrids.keys():
                print(self.mrids[k], v)

    def reset_compensators(self) -> None:
        load = dict_to_array(self.consumers)[:, :, 1]
        load = np.sum(load, axis=0)

        print(load)

        for k, v in self.compensators.items():
            diff_builder = DifferenceBuilder(sim.simulation_id)
            diff_builder.add_difference(
                k, ControlAttributes.capacitor, 0, 1
            )
            topic = t.simulation_input_topic(sim.simulation_id)
            message = diff_builder.get_message()
            print(topic)
            print(message)
            self.gapps.send(topic, message)

    def get_phase_vars(self, load_mult: float) -> np.ndarray:
        # End of March
        # check compensator first to see if we need them based on the load
        # greedy algo on 3 phase and then individual phases
        # terminal voltage should be the lowest (next-step)

        load = dict_to_array(self.consumers)[:, :, 1]
        load = np.sum(load, axis=0) * load_mult

        comp = dict_to_array(self.compensators)[:, :, 1]
        comp = np.sum(comp, axis=0)

        return load

    def get_bounds(self, solar_mult: float) -> np.ndarray:
        apparent = dict_to_array(self.electronics)[:, :, 0]
        real = apparent * solar_mult
        reactive = np.sqrt(abs(real**2 - apparent**2))
        return reactive


def dispatch(self, load: float, solar: float) -> np.ndarray:

    bounds = self.get_bounds(solar)
    A = np.clip(bounds, a_min=0, a_max=1)
    b = self.get_phase_vars(load)

    # constrain values to lesser of ders and network
    total_der = np.sum(bounds, axis=0)
    direction = np.clip(b, a_max=1, a_min=-1)
    print(b, direction)
    b = np.array([min(vals) for vals in zip(abs(total_der), abs(b))])
    b = b*direction
    print(b)

    # Construct the problem.
    m, n = np.shape(A)
    x = cp.Variable((m, n))

    cost = cp.sum(self.dist.T@cp.abs(x))
    objective = cp.Minimize(cost)
    constraints = [
        cp.sum(A.T@x, axis=0) == b,
        -bounds <= x,
        x <= bounds]

    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status == 'optimal':
        return np.round(x.value, 0)

    print(prob.status)

    return prob.status


def plot(setpoints: np.ndarray, profiles: DataFrame) -> None:

    solar = profiles["Solar"]
    load = profiles["Loadshape"]
    time = profiles["Time"]

    fig, ax = plt.subplots()

    ax.plot(time, solar, label='Solar')
    ax.plot(time, load, label='Load')
    print(np.shape(setpoints))
    for der in setpoints.transpose():
        norm = np.linalg.norm(der)
        ax.plot(time, der/norm)

    ax.set(xlabel='Time (5 min)', ylabel='Output (%)')
    fig.legend()
    plt.savefig('outputs/dispatch.png', dpi=400)


def dispatch_serialize(dispatch: list) -> np.ndarray:
    n = len(dispatch)
    m = len(dispatch[0])

    setpoints = np.zeros((n, m))
    for idx, row in enumerate(dispatch):
        setpoints[idx] = np.sum(row, axis=1)

    return setpoints


class ModelInfo(BaseModel):
    modelName: str
    modelId: str
    stationName: str
    stationId: str
    subRegionName: str
    subRegionId: str
    regionName: str
    regionId: str


if __name__ == "__main__":
    try:
        cim_profile = 'rc4_2021'
        cim = importlib.import_module('cimgraph.data_profile.' + cim_profile)
        mrid = IEEE123

        feeder = cim.Feeder(mRID=mrid)
        params = ConnectionParameters(
            url="http://localhost:8889/bigdata/namespace/kb/sparql",
            cim_profile=cim_profile)

        gapps = GridAPPSD(username='system', password='manager')
        addr, port = gridappsd.utils.get_gridappsd_address()

        bg = BlazegraphConnection(params)
        network = FeederModel(
            connection=bg,
            container=feeder,
            distributed=False)
        utils.get_all_data(network)

        models = gapps.query_model_info()['data']['models']
        for model in models:
            if model['modelId'] == mrid:
                system = ModelInfo(**model)

        system_config = PowerSystemConfig(
            GeographicalRegion_name=system.regionId,
            SubGeographicalRegion_name=system.subRegionId,
            Line_name=system.modelId
        )

        model_config = ModelCreationConfig(
            load_scaling_factor=1,
            schedule_name="ieeezipload",
            z_fraction=0,
            i_fraction=1,
            p_fraction=0,
            randomize_zipload_fractions=False,
            use_houses=False
        )

        start = datetime(2024, 1, 1)
        epoch = datetime.timestamp(start)
        duration = timedelta(minutes=7).total_seconds()

        sim_args = SimulationArgs(
            start_time=epoch,
            duration=duration,
            simulator="GridLAB-D",
            timestep_frequency=1000,
            timestep_increment=1000,
            run_realtime=False,
            simulation_name=system.modelName,
            power_flow_solver_method="NR",
            model_creation_config=asdict(model_config)
        )

        sim_config = SimulationConfig(
            power_system_config=asdict(system_config),
            simulation_config=asdict(sim_args)
        )

        sim = Simulation(gapps, asdict(sim_config))
        app = PowerFactor(gapps, sim)
        # gapps.subscribe(t.simulation_output_topic(sim.simulation_id), app)
        # profiles = read_csv('data/time-series.csv')
        # dispatch = []
        # for idx, row in profiles.iterrows():
        # print("Iteration:\n", row)
        # setpoints = app.dispatch(row['Loadshape'], row['Solar'])
        # dispatch.append(setpoints)
#
        # dispatch = dispatch_serialize(dispatch)
        # plot(dispatch, profiles)

        while sim._running_or_paused:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                log.debug("Exiting sample")
                sim.stop()
                break

    except Exception as e:
        log.debug(e)
        log.debug(traceback.format_exc())
        sim.stop()
