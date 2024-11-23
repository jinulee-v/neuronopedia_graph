from pyvis.network import Network
import networkx as nx
import json
import os
import pickle

PALETTE = ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', '#8be0a4', '#b497e7', '#c3c3c3']

def visualize_graph(G: nx.Graph, title='Graph', save_path="graphs", labels = None, thresh=0):
    # Initialize a Pyvis Network object
    directed = nx.is_directed(G)
    net = Network(notebook=True, directed=directed, height="750px", width="100%", cdn_resources='in_line')

    # net.set_options(json.dumps({
    #     "layout": {
    #         "hierarchical": {
    #             "enabled": True,
    #             "direction": "DU",
    #             "sortMethod": "directed"
    #         }
    #     }
    # }))

    # Load the NetworkX graph into Pyvis with customized edge thickness
    def id(node):
        return f"{node[0]}_{node[1]}" # graph layer and feature idx
    def label(node):
        # labels = [{"0": xx, "1": yy, ..}, {"0": aa, "1": bb, ..}, ...]
        if labels is not None:
            return labels[node[0]][node[1]]
        return node
    ids = set()
    for u, v, data in G.edges(data=True):
        strength = data['weight']
        if strength > thresh:
            if id(u) not in ids:
                net.add_node(id(u), label=id(u), title=label(u), shape='box', size=20, color=PALETTE[u[0] % len(PALETTE)])
                ids.add(id(u))
            if id(v) not in ids:
                net.add_node(id(v), label=id(v), title=label(v), shape='box', size=20, color=PALETTE[v[0] % len(PALETTE)])
                ids.add(id(v))
            net.add_edge(id(u), id(v), width=strength * 10, label=f"{strength:.3f}")  # Adjust multiplier for thickness scaling

    # Show the interactive graph and save it as an HTML string
    print("Visualizing graph...")
    net.show(".graph.html")

    # Read the generated HTML file and display it inline
    with open(".graph.html", "r") as f:
        graph_html = f.read()
    
    # Delete temporary .graph.html file
    os.remove(".graph.html")

    with open(f"{save_path}/{title}.html", "w") as file:
        file.write(graph_html)
    with open(f"{save_path}/{title}.pkl", "wb") as file:
        pickle.dump(G, file)