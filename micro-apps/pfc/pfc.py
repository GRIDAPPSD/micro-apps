import logging
import gridappsd.field_interface.agents.agents as agents_mod
from cimgraph.data_profile import CIM_PROFILE
cim_profile = CIM_PROFILE.RC4_2021.value
agents_mod.set_cim_profile(cim_profile)
cim = agents_mod.cim

log = logging.getLogger(__name__)


def init_cim(network_area) -> None:
    network_area.get_all_attributes(cim.ACLineSegment)
    network_area.get_all_attributes(cim.ACLineSegmentPhase)
    network_area.get_all_attributes(cim.BaseVoltage)
    network_area.get_all_attributes(cim.SvEstVoltage)
    network_area.get_all_attributes(cim.SvVoltage)
    network_area.get_all_attributes(cim.Equipment)
    network_area.get_all_attributes(cim.PerLengthPhaseImpedance)
    network_area.get_all_attributes(cim.PhaseImpedanceData)
    network_area.get_all_attributes(cim.WireSpacingInfo)
    network_area.get_all_attributes(cim.WirePosition)
    network_area.get_all_attributes(cim.OverheadWireInfo)
    network_area.get_all_attributes(cim.ConcentricNeutralCableInfo)
    network_area.get_all_attributes(cim.TapeShieldCableInfo)
    network_area.get_all_attributes(cim.TransformerTank)
    network_area.get_all_attributes(cim.TransformerTankEnd)
    network_area.get_all_attributes(cim.TransformerTankInfo)
    network_area.get_all_attributes(cim.TransformerEndInfo)
    network_area.get_all_attributes(cim.ShortCircuitTest)
    network_area.get_all_attributes(cim.NoLoadTest)
    network_area.get_all_attributes(cim.PowerElectronicsConnection)
    network_area.get_all_attributes(cim.PowerElectronicsConnectionPhase)
    network_area.get_all_attributes(cim.EnergyConsumer)
    network_area.get_all_attributes(cim.EnergyConsumerPhase)
    network_area.get_all_attributes(cim.Analog)
    network_area.get_all_attributes(cim.EnergyConsumerPhase)
    network_area.get_all_attributes(cim.Terminal)


def query_line_info(network_area):
    if cim.ACLineSegment not in network_area.typed_catalog:
        return None

    line_ids = list(network_area.typed_catalog[cim.ACLineSegment].keys())

    for line_id in line_ids:
        line = network_area.typed_catalog[cim.ACLineSegment][line_id]
        print(line)

        from_bus = line.Terminals[0].ConnectivityNode.name
        print(from_bus)

        to_bus = line.Terminals[1].ConnectivityNode.name
        print(to_bus)


if __name__ == "__main__":

    print("Hello from pfc")
