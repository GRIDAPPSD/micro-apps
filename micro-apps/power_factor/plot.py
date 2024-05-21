import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame


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


def plot_graph(G: nx.Graph, dist: dict, pos: dict) -> None:

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
