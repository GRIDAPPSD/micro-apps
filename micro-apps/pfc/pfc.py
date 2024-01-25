from enum import Enum
from cimgraph import utils
from cimgraph.databases import ConnectionParameters, BlazegraphConnection
from cimgraph.models import FeederModel
import importlib


class LineName(Enum):
    IEEE13 = "_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62"
    IEEE123 = "_C1C3E687-6FFD-C753-582B-632A27E28507"
    IEEE123PV = "_A3BC35AA-01F6-478E-A7B1-8EA4598A685C"


if __name__ == "__main__":
    cim_profile = 'rc4_2021'
    cim = importlib.import_module('cimgraph.data_profile.' + cim_profile)

    feeder = cim.Feeder(mRID=LineName.IEEE13.value)
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
        network.get_all_edges(cim.ACLineSegment)
        network.pprint(cim.ACLineSegment, show_empty=False, json_ld=False)
    except Exception as e:
        raise e
