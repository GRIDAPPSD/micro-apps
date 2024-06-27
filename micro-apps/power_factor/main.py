import os
import traceback
import logging
from pprint import pprint
import networkx as nx
import numpy as np
import cvxpy as cp
import importlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import cimgraph.utils
from cimgraph.databases import ConnectionParameters
from cimgraph.databases import BlazegraphConnection
from cimgraph.databases import GridappsdConnection
from cimgraph.models import FeederModel
import cimgraph.data_profile.rc4_2021 as cim_dp
from gridappsd import GridAPPSD, DifferenceBuilder
from gridappsd.simulation import Simulation
from gridappsd.simulation import PowerSystemConfig
from gridappsd.simulation import ApplicationConfig
from gridappsd.simulation import SimulationConfig
from gridappsd.simulation import SimulationArgs
from gridappsd.simulation import ModelCreationConfig
from gridappsd.simulation import ServiceConfig
from gridappsd import topics as t
import models
import query

IEEE123_APPS = "_E3D03A27-B988-4D79-BFAB-F9D37FB289F7"
IEEE13 = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
IEEE123 = "_C1C3E687-6FFD-C753-582B-632A27E28507"
IEEE123PV = "_E407CBB6-8C8D-9BC9-589C-AB83FBF0826D"
SOURCE_BUS = "150r"
OUT_DIR = "outputs"
ROOT = os.getcwd()


logging.basicConfig(level=logging.DEBUG,
                    filename=f"{ROOT}/{OUT_DIR}/log.txt", filemode='w')
log = logging.getLogger(__name__)


def remap_phases(
        info: models.MeasurementInfo,
        map: models.PhaseMap) -> models.MeasurementInfo:
    if info.phase == cim_dp.PhaseCode.s1:
        info.phase = map.s1
        return info

    if info.phase == cim_dp.PhaseCode.s2:
        info.phase = map.s2
        return info

    return info


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
    gapps: GridAPPSD
    sim: Simulation
    network: cimgraph.GraphModel
    compensators: models.Compensators
    consumers: models.Consumers
    electronics: models.PowerElectronics

    def __init__(self, gapps: GridAPPSD, sim: Simulation, network: cimgraph.GraphModel):
        self.gapps = gapps
        self.switches = query.get_switches(network)
        pprint(self.switches)
        log.debug(self.switches)
        self.compensators = query.get_compensators(network)
        log.debug(self.compensators)
        self.consumers = query.get_consumers(network)
        log.debug(self.consumers)
        self.electronics = query.get_power_electronics(network)
        log.debug(self.electronics)

        (graph, source_bus) = query.generate_graph(network)
        print(source_bus)
        paths = nx.shortest_path_length(graph, source=SOURCE_BUS)
        self.dist = dist_matrix(query.map_power_electronics(network), paths)

        sim.add_onmeasurement_callback(self.on_measurement)
        self.last_dispatch = 0

    def on_measurement(self, sim: Simulation, timestamp: dict, measurements: dict) -> None:
        if timestamp - self.last_dispatch >= 10:
            self.last_dispatch = timestamp
            self.toggle_switches()
            # self.dispatch()

        print(f"{timestamp}: on_measurement")
        for k, v in measurements.items():
            if k in self.switches.measurement_map:
                val = models.SimulationMeasurement(**v)
                if val.value is not None:
                    info = self.switches.measurement_map[k]

            if k in self.consumers.measurement_map:
                val = models.SimulationMeasurement(**v)
                if val.value is not None:
                    continue

                info = self.consumers.measurement_map[k]
                if info.value_type == "PNV":
                    old = self.consumers.measurements_pnv[info.mrid]
                    new = models.update_pnv(info, val, old)
                    self.consumers.measurements_pnv[info.mrid] = new

                if info.value_type == "VA":
                    map = self.consumers.measurements_pnv[info.mrid]
                    info = remap_phases(info, map)
                    self.consumers.measurement_map[k] = info

                    old = self.consumers.measurements_va[info.mrid]
                    new = models.update_va(info, val, old)
                    self.consumers.measurements_va[info.mrid] = new

            if k in self.compensators.measurement_map:
                val = models.SimulationMeasurement(**v)
                if val.value is not None:
                    continue

                info = self.compensators.measurement_map[k]
                if info.value_type == "PNV":
                    old = self.compensators.measurements_pnv[info.mrid]
                    new = models.update_pnv(info, val, old)
                    self.compensators.measurements_pnv[info.mrid] = new

                if info.value_type == "VA":
                    map = self.compensators.measurements_pnv[info.mrid]
                    info = remap_phases(info, map)
                    self.compensators.measurement_map[k] = info

                    old = self.compensators.measurements_va[info.mrid]
                    new = models.update_va(info, val, old)
                    self.compensators.measurements_va[info.mrid] = new

            if k in self.electronics.measurement_map:
                val = models.SimulationMeasurement(**v)
                if val.value is not None:
                    continue

                info = self.electronics.measurement_map[k]
                if info.value_type == "PNV":
                    old = self.electronics.measurements_pnv[info.mrid]
                    new = models.update_pnv(info, val, old)
                    self.electronics.measurements_pnv[info.mrid] = new

                if info.value_type == "VA":
                    map = self.electronics.measurements_pnv[info.mrid]
                    info = remap_phases(info, map)
                    self.electronics.measurement_map[k] = info

                    old = self.electronics.measurements_va[info.mrid]
                    new = models.update_va(info, val, old)
                    self.electronics.measurements_va[info.mrid] = new

    def toggle_switches(self) -> None:
        diff_builder = DifferenceBuilder(sim.simulation_id)
        v: models.MeasurementInfo
        for k, v in self.switches.measurement_map.items():
            if v.phase == "A":
                print(k, v)
                diff_builder.add_difference(
                    k, models.Switches.open_attribute, True, False
                )

        topic = t.simulation_input_topic(sim.simulation_id)
        message = diff_builder.get_message()
        log.debug(message)
        self.gapps.send(topic, message)

    def set_compensators(self) -> None:
        loads = dict_to_array(self.consumers.measurements)[:, :, 1]
        total_load = np.sum(loads, axis=0)

        comps_abc = {k: v for k, v in self.compensators.ratings.items()
                     if v.a.imag != 0 and v.b.imag != 0 and v.c.imag != 0}
        comps_a = {k: v for k, v in self.compensators.ratings.items()
                   if v.b.imag == 0 and v.c.imag == 0}
        comps_b = {k: v for k, v in self.compensators.ratings.items()
                   if v.a.imag == 0 and v.c.imag == 0}
        comps_c = {k: v for k, v in self.compensators.ratings.items()
                   if v.a.imag == 0 and v.b.imag == 0}

        diff_builder = DifferenceBuilder(sim.simulation_id)
        for k, v in comps_abc.items():
            vars = np.sum(np.array(v), axis=1)
            net = total_load + vars
            if net.all() > 0:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 0, 1
                )
                total_load = net
            else:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 1, 0
                )

        for k, v in comps_a.items():
            vars = np.sum(np.array(v), axis=1)
            net = total_load + vars
            if net[0] > 0:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 0, 1
                )
                total_load = net
            else:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 1, 0
                )

        for k, v in comps_b.items():
            vars = np.sum(np.array(v), axis=1)
            net = total_load + vars
            if net[1] > 0:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 0, 1
                )
                total_load = net
            else:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 1, 0
                )

        for k, v in comps_c.items():
            vars = np.sum(np.array(v), axis=1)
            net = total_load + vars
            if net[2] > 0:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 0, 1
                )
                total_load = net
            else:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 1, 0
                )

        topic = t.simulation_input_topic(sim.simulation_id)
        message = diff_builder.get_message()
        log.debug(message)
        self.gapps.send(topic, message)

    def set_electronics(self, dispatch: np.ndarray) -> None:
        diff_builder = DifferenceBuilder(sim.simulation_id)
        for k, v in self.electronics.measurements.items():
            vars = np.sum(np.array(v), axis=1)
            total_load = 0
            net = total_load + vars
            if net[2] > 0:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 0, 1
                )
                total_load = net
            else:
                diff_builder.add_difference(
                    k, models.Compensators.sections_attribute, 1, 0
                )

        topic = t.simulation_input_topic(sim.simulation_id)
        message = diff_builder.get_message()
        log.debug(message)
        self.gapps.send(topic, message)

    def get_phase_vars(self) -> np.ndarray:
        load = dict_to_array(self.consumers.measurements)[:, :, 1]
        total_load = np.sum(load, axis=0)

        comp = dict_to_array(self.compensators.measurements)[:, :, 1]
        total_comp = np.sum(comp, axis=0)

        print("Total consumers: ", total_load)
        print("Total compensators: ", total_comp)

        return total_load + total_comp

    def get_bounds(self) -> np.ndarray:
        apparent = dict_to_array(self.electronics.ratings)[:, :, 0]
        real = dict_to_array(self.electronics.measurements)[:, :, 0]
        print("PECS: ", apparent, real)
        reactive = np.sqrt(abs(real**2 - apparent**2))
        return reactive

    def dispatch(self) -> np.ndarray:
        self.set_compensators()

        bounds = self.get_bounds()
        print("Bounds: ", bounds)
        A = np.clip(bounds, a_min=0, a_max=1)
        b = self.get_phase_vars()

        # constrain values to lesser of ders and network
        total_der = np.sum(bounds, axis=0)
        direction = np.clip(b, a_max=1, a_min=-1)
        b = np.array([min(vals) for vals in zip(abs(total_der), abs(b))])
        b = b*direction

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


@dataclass
class ModelInfo:
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
        mrid = IEEE123PV

        feeder = cim.Feeder(mRID=mrid)

        params = ConnectionParameters(
            url="http://localhost:8889/bigdata/namespace/kb/sparql",
            cim_profile=cim_profile)
        bgc = BlazegraphConnection(params)

        # params = ConnectionParameters(
        #    cim_profile=cim_profile,
        #    iec61970_301=7
        # )
        # gac = GridappsdConnection(params)
        gapps = GridAPPSD(username='system', password='manager')
        network = FeederModel(
            connection=bgc,
            container=feeder,
            distributed=False)

        cimgraph.utils.get_all_data(network)
        cimgraph.utils.get_all_measurement_data(network)

        model_info = gapps.query_model_info()['data']['models']
        for model in model_info:
            if model['modelId'] == mrid:
                system = ModelInfo(**model)

        system_config = PowerSystemConfig(
            GeographicalRegion_name=system.regionId,
            SubGeographicalRegion_name=system.subRegionId,
            Line_name=system.modelId
        )

        model_config = ModelCreationConfig(
            load_scaling_factor=1,
            schedule_name="",
            z_fraction=0,
            i_fraction=1,
            p_fraction=0,
            randomize_zipload_fractions=False,
            use_houses=False
        )

        start = datetime(2023, 1, 1, 4)
        epoch = datetime.timestamp(start)
        duration = timedelta(seconds=60).total_seconds()

        sim_args = SimulationArgs(
            start_time=epoch,
            duration=duration,
            simulator="GridLAB-D",
            timestep_frequency=1000,
            timestep_increment=1000,
            run_realtime=True,
            simulation_name=system.modelName,
            power_flow_solver_method="NR",
            model_creation_config=asdict(model_config)
        )

        sim_config = SimulationConfig(
            power_system_config=asdict(system_config),
            simulation_config=asdict(sim_args)
        )
        sim = Simulation(gapps, asdict(sim_config))

        app = PowerFactor(gapps, sim, network)

        sim.run_loop()

    except KeyboardInterrupt:
        sim.stop()

    except Exception as e:
        log.debug(e)
        log.debug(traceback.format_exc())
