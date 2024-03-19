import networkx as nx
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pandas import read_csv, read_table, DataFrame
import importlib
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


IEEE123_APPS = "_E3D03A27-B988-4D79-BFAB-F9D37FB289F7"
IEEE13 = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
IEEE123 = "_C1C3E687-6FFD-C753-582B-632A27E28507"
IEEE123PV = "_A3BC35AA-01F6-478E-A7B1-8EA4598A685C"
SOURCE_BUS = '150r'


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


def plot_graph(G: nx.Graph, dist: dict) -> None:
    loc = utils.get_all_bus_locations(network)
    pos = {d['name']: [float(d['x']), float(d['y'])]
           for d in loc.values()}

    # nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=20,
        nodelist=list(dist.keys()),
        node_color=list(dist.values()),
        cmap=plt.cm.plasma
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=2, font_family="sans-serif")

    # edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.4)

    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=2)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/graph.png", dpi=400)


def get_compensators() -> dict:
    compensators = {}
    shunt: LinearShuntCompensator
    for shunt in network.graph[LinearShuntCompensator].values():

        name = shunt.name
        nom_v = float(shunt.nomU)
        susceptance = -float(shunt.bPerSection) * nom_v**2
        power = np.zeros((3, 2))

        if not shunt.ShuntCompensatorPhase:
            power[0] = [0, susceptance]
            power[1] = [0, susceptance]
            power[2] = [0, susceptance]

        else:
            phase: LinearShuntCompensatorPhase
            for phase in shunt.ShuntCompensatorPhase:
                if phase.phase == SinglePhaseKind.A:
                    power[0] = [0, susceptance]
                if phase.phase == SinglePhaseKind.B:
                    power[1] = [0, susceptance]
                if phase.phase == SinglePhaseKind.C:
                    power[2] = [0, susceptance]
        compensators[name] = power

    return compensators


def get_consumers() -> dict:
    consumers = {}
    load: EnergyConsumer
    for load in network.graph[EnergyConsumer].values():
        name = load.name
        power = np.zeros((3, 2))

        if not load.EnergyConsumerPhase:
            power[0] = [load.p, load.q]
            power[1] = [load.p, load.q]
            power[2] = [load.p, load.q]

        else:
            phases = load.EnergyConsumerPhase
            phase: EnergyConsumerPhase
            for phase in phases:
                if phase.phase == SinglePhaseKind.A:
                    power[0] = [phase.p, phase.q]
                if phase.phase == SinglePhaseKind.B:
                    power[1] = [phase.p, phase.q]
                if phase.phase == SinglePhaseKind.C:
                    power[2] = [phase.p, phase.q]
        consumers[name] = power
    return consumers


def get_power_electronics() -> dict:
    electronics = {}
    pec: PowerElectronicsConnection
    for pec in network.graph[PowerElectronicsConnection].values():
        name = pec.name
        power = np.zeros((3, 2))

        if not pec.PowerElectronicsConnectionPhases:
            power[0] = [pec.ratedS, pec.ratedS]
            power[1] = [pec.ratedS, pec.ratedS]
            power[2] = [pec.ratedS, pec.ratedS]

        else:
            phases = pec.PowerElectronicsConnectionPhases
            phase: PowerElectronicsConnectionPhase
            for phase in phases:
                if phase.phase == SinglePhaseKind.A:
                    power[0] = [pec.ratedS, pec.ratedS]
                if phase.phase == SinglePhaseKind.B:
                    power[1] = [pec.ratedS, pec.ratedS]
                if phase.phase == SinglePhaseKind.C:
                    power[2] = [pec.ratedS, pec.ratedS]
        electronics[name] = power

    return electronics


def map_power_electronics() -> dict:
    map = {}
    pec: PowerElectronicsConnection
    for pec in network.graph[PowerElectronicsConnection].values():
        name = pec.name
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
    def __init__(self):
        self.compensators = get_compensators()
        self.consumers = get_consumers()
        self.electronics = get_power_electronics()

        graph = generate_graph()
        dist = nx.shortest_path_length(graph, source=SOURCE_BUS)
        electronics_map = map_power_electronics()
        self.dist = dist_matrix(electronics_map, dist)
        print(self.dist)
        plot_graph(graph, dist)

    def get_phase_vars(self, load_mult: float) -> np.ndarray:
        load = dict_to_array(self.consumers)[:, :, 1]
        load = np.sum(load, axis=0) * load_mult

        comp = dict_to_array(self.compensators)[:, :, 1]
        comp = np.sum(comp, axis=0)

        return load + comp

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
        result = prob.solve(solver=cp.CLARABEL, verbose=False)

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


if __name__ == "__main__":
    cim_profile = 'rc4_2021'
    cim = importlib.import_module('cimgraph.data_profile.' + cim_profile)

    feeder = cim.Feeder(mRID=IEEE123_APPS)
    params = ConnectionParameters(
        url="http://localhost:8889/bigdata/namespace/kb/sparql",
        cim_profile=cim_profile)

    try:
        bg = BlazegraphConnection(params)
        print(bg)
        network = FeederModel(
            connection=bg,
            container=feeder,
            distributed=False)
        utils.get_all_data(network)

        app = PowerFactor()
        profiles = read_csv('data/time-series.csv')
        dispatch = []
        for _, row in profiles.iterrows():
            print("Iteration:\n", row)
            setpoints = app.dispatch(row['Loadshape'], row['Solar'])
            dispatch.append(setpoints)

        dispatch = dispatch_serialize(dispatch)
        plot(dispatch, profiles)

    except Exception as e:
        raise e
