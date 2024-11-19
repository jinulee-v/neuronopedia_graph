from pyvis.network import Network
import networkx as nx
import os
import pickle

def visualize_bipartite_graph(G: nx.Graph, title='Graph', save_path="graphs", thresh=0):
    # Initialize a Pyvis Network object
    directed = nx.is_directed(G)
    net = Network(notebook=True, directed=directed, height="750px", width="100%", cdn_resources='in_line')

    # Load the NetworkX graph into Pyvis with customized edge thickness
    for u, v, data in G.edges(data=True):
        strength = data['weight']
        if strength > thresh:
            net.add_node(-u, label=u, title=u, shape='box', size=20, color="skyblue")
            net.add_node(v, label=v, title=v, shape='box', size=20, color="pink")
            net.add_edge(-u, v, width=strength * 10, label=f"{strength:.3f}")  # Adjust multiplier for thickness scaling

    # Show the interactive graph and save it as an HTML string
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


def create_graph_conditional_prob(activations_from, lats_for_sents_from, activations_to, lats_for_sents_to):
    graph = {}
    for x in activations_from.keys(): # enumerate nodes
        inner_lst = set()
        for sent_tok in activations_from[x]: # for each sentence it activates on
            if sent_tok in lats_for_sents_to:
                inner_lst.update(lats_for_sents_to[sent_tok])
        for y in inner_lst: # for latents those sentences activate on
            if x == y:
                continue
            st1, st2 = activations_from[x], activations_to[y]
            intersection = len(st1.intersection(st2))
            union = len(st1)
            if intersection > 0:
                graph[str(x)] = graph.get(x, {})
                graph[str(x)][str(y)] = intersection / union
    
    # Initialize a NetworkX graph
    G = nx.DiGraph()
    # Add edges and nodes with strength-based thickness
    for node_i, neighbors in graph.items():
        for node_j, strength in neighbors.items():
            G.add_edge(node_i, node_j, weight=strength)
    return G

def create_bipartite_graph(activations_from, lats_for_sents_from, activations_to, lats_for_sents_to, mode="symmetric_jaccard"):
    if mode == "conditional_prob":
        return create_graph_conditional_prob(activations_from, lats_for_sents_from, activations_to, lats_for_sents_to)
    else:
        raise ValueError(f"Invalid mode: {mode}")