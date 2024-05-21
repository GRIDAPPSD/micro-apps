import networkx as nx
import cimgraph.utils
import cimgraph.models
import cimgraph.data_profile.rc4_2021 as cim_dp
import models
import logging

log = logging.getLogger(__name__)


def generate_graph(network: cimgraph.models.GraphModel) -> nx.Graph:
    graph = nx.Graph()

    network.get_all_edges(cim_dp.ACLineSegment)
    network.get_all_edges(cim_dp.ConnectivityNode)
    network.get_all_edges(cim_dp.Terminal)

    line: cim_dp.ACLineSegment
    for line in network.graph[cim_dp.ACLineSegment].values():
        terminals = line.Terminals
        bus_1 = terminals[0].ConnectivityNode.name
        bus_2 = terminals[1].ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=float(line.length))

    xfmr: cim_dp.TransformerTank
    for xfmr in network.graph[cim_dp.TransformerTank].values():
        ends = xfmr.TransformerTankEnds
        bus_1 = ends[0].Terminal.ConnectivityNode.name
        bus_2 = ends[1].Terminal.ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=0.0)

    switch: cim_dp.LoadBreakSwitch
    for switch in network.graph[cim_dp.LoadBreakSwitch].values():
        terminals = switch.Terminals
        bus_1 = terminals[0].ConnectivityNode.name
        bus_2 = terminals[1].ConnectivityNode.name
        graph.add_edge(bus_1, bus_2, weight=0.0)

    return graph


def get_graph_positions(network: cimgraph.models.GraphModel) -> dict:
    loc = cimgraph.utils.get_all_bus_locations(network)
    return {d['name']: [float(d['x']), float(d['y'])]
            for d in loc.values()}


def get_compensators(network: cimgraph.models.GraphModel) -> models.Compensators:
    compensators = models.Compensators()
    if cim_dp.LinearShuntCompensator not in network.graph:
        return compensators

    shunt: cim_dp.LinearShuntCompensator
    for shunt in network.graph[cim_dp.LinearShuntCompensator].values():

        mrid = shunt.mRID
        nom_v = float(shunt.nomU)
        p_imag = -float(shunt.bPerSection) * nom_v**2

        if not shunt.ShuntCompensatorPhase:
            measurement = cim_dp.Measurement
            for measurement in shunt.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "VA":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                compensators.measurement_map[measurement.mRID] = info

            compensators.measurements[mrid] = models.PhasePower()

            power = models.ComplexPower(real=0.0, imag=p_imag/3)
            phases = models.PhasePower(a=power, b=power, c=power)
            compensators.ratings[mrid] = phases

        else:
            ratings = models.PhasePower()
            phase: models.LinearShuntCompensatorPhase
            for phase in shunt.ShuntCompensatorPhase:
                power = models.ComplexPower(real=0.0, imag=p_imag)
                if phase.phase == cim_dp.SinglePhaseKind.A:
                    ratings.a = power
                if phase.phase == cim_dp.SinglePhaseKind.B:
                    ratings.b = power
                if phase.phase == cim_dp.SinglePhaseKind.C:
                    ratings.c = power

            compensators.ratings[mrid] = ratings

            measurement = cim_dp.Measurement
            for measurement in shunt.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "VA":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                compensators.measurement_map[measurement.mRID] = info

            compensators.measurements[mrid] = models.PhasePower()

    return compensators


def get_consumers(network: cimgraph.GraphModel) -> models.Consumers:
    consumers = models.Consumers()
    if cim_dp.EnergyConsumer not in network.graph:
        return consumers

    load: cim_dp.EnergyConsumer
    for load in network.graph[cim_dp.EnergyConsumer].values():
        mrid = load.mRID

        if not load.EnergyConsumerPhase:
            measurement = cim_dp.Measurement
            for measurement in load.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "PNV":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                consumers.measurement_map[measurement.mRID] = info

            consumers.measurements[mrid] = models.PhasePower()

            p = float(load.p)
            q = float(load.q)
            power = models.ComplexPower(real=p/3, imag=q/3)
            ratings = models.PhasePower(a=power, b=power, c=power)
            consumers.ratings[mrid] = ratings

        else:
            ratings = models.PhasePower()
            phase: cim_dp.EnergyConsumerPhase

            for phase in load.EnergyConsumerPhase:
                power = models.ComplexPower(real=phase.p, imag=phase.q)

                if phase.phase == cim_dp.SinglePhaseKind.A:
                    ratings.a = power
                if phase.phase == cim_dp.SinglePhaseKind.B:
                    ratings.b = power
                if phase.phase == cim_dp.SinglePhaseKind.C:
                    ratings.c = power

            consumers.ratings[mrid] = ratings

            measurement = cim_dp.Measurement
            for measurement in load.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "PNV":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                consumers.measurement_map[measurement.mRID] = info

            consumers.measurements[mrid] = models.PhasePower()

    return consumers


def get_power_electronics(network: cimgraph.GraphModel) -> models.PowerElectronics:
    electronics = models.PowerElectronics()
    if cim_dp.PowerElectronicsConnection not in network.graph:
        return electronics

    pec: cim_dp.PowerElectronicsConnection
    for pec in network.graph[cim_dp.PowerElectronicsConnection].values():
        mrid = pec.mRID

        if not pec.PowerElectronicsConnectionPhases:
            measurement = cim_dp.Measurement
            for measurement in pec.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "PNV":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                electronics.measurement_map[measurement.mRID] = info

            electronics.measurements[mrid] = models.PhasePower()

            rated_s = float(pec.ratedS)
            power = models.ComplexPower(real=rated_s/3, imag=rated_s/3)
            ratings = models.PhasePower(a=power, b=power, c=power)
            electronics.ratings[mrid] = ratings

        else:
            ratings = models.PhasePower()
            phase: cim_dp.PowerElectronicsConnectionPhase
            for phase in pec.PowerElectronicsConnectionPhases:
                power = models.ComplexPower(
                    real=float(phase.p), imag=float(phase.q))
                if phase.phase == cim_dp.SinglePhaseKind.A:
                    ratings.a = power
                if phase.phase == cim_dp.SinglePhaseKind.B:
                    ratings.b = power
                if phase.phase == cim_dp.SinglePhaseKind.C:
                    ratings.c = power
            electronics.ratings[mrid] = ratings

            measurement = cim_dp.Measurement
            for measurement in pec.Measurements:
                print(measurement.name, measurement.measurementType,
                      measurement.phases)
                if measurement.measurementType != "PNV":
                    continue

                info = models.MeasurementInfo(
                    mrid=mrid, phase=measurement.phases)
                electronics.measurement_map[measurement.mRID] = info

            electronics.measurements[mrid] = models.PhasePower()

    return electronics


def map_power_electronics(network: cimgraph.GraphModel) -> dict:
    map = {}
    if cim_dp.PowerElectronicsConnection not in network.graph:
        return map

    pec: cim_dp.PowerElectronicsConnection
    for pec in network.graph[cim_dp.PowerElectronicsConnection].values():
        name = pec.mRID
        bus = pec.Terminals[0].ConnectivityNode.name
        map[name] = bus
    return map
