import networkx as nx
import matplotlib.pyplot as plt


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
