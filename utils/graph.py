import networkx as nx
from collections import defaultdict

################ Graph for single SAE ################

def create_graph_symmetric_jaccard(activations, lats_for_sents, thresh=0):
    graph = defaultdict(dict)

    for x in activations.keys(): # enumerate nodes
        inner_lst = set()
        for sent_tok in activations[x]: # for each sentence it activates on
            inner_lst.update(lats_for_sents[sent_tok])
        for y in inner_lst: # for latents those sentences activate on
            if x == y:
                continue
            fst, snd = (x, y) if x < y else (y, x)
            # assert not (fsts in graph and snds in graph[fsts])
            st1, st2 = activations[fst], activations[snd]
            intersection = len(st1.intersection(st2))
            union = len(st1.union(st2))
            if intersection / union > thresh:
                graph[(0, fst)][(0, snd)] = intersection / union
                
    # Initialize a NetworkX graph
    G = nx.Graph()
    # Add edges and nodes with strength-based thickness
    for node_i, neighbors in graph.items():
        for node_j, strength in neighbors.items():
            G.add_edge(node_i, node_j, weight=strength)
    return G

def create_graph_conditional_prob(activations, lats_for_sents, thresh=0):
    graph = defaultdict(dict)

    for x in activations.keys(): # enumerate nodes
        inner_lst = set()
        for sent_tok in activations[x]: # for each sentence it activates on
            inner_lst.update(lats_for_sents[sent_tok])
        for y in inner_lst: # for latents those sentences activate on
            if x == y:
                continue
            st1, st2 = activations[x], activations[y]
            intersection = len(st1.intersection(st2))
            union = len(st1)
            if intersection / union > thresh:
                graph[(0, x)][(0, y)] = intersection / union
    
    # Initialize a NetworkX graph
    sum = 0
    for g in graph.values():
        sum += len(g)
    print("SAE graph generated")
    print("Number of edges", sum)
    G = nx.DiGraph()
    # Add edges and nodes with strength-based thickness
    for node_i, neighbors in graph.items():
        for node_j, strength in neighbors.items():
            G.add_edge(node_i, node_j, weight=strength)
    return G

def create_graph(activations, lats_for_sents, mode="symmetric_jaccard", thresh=0):
    if mode == "symmetric_jaccard":
        return create_graph_symmetric_jaccard(activations, lats_for_sents, thresh=thresh)
    elif mode == "conditional_prob":
        return create_graph_conditional_prob(activations, lats_for_sents, thresh=thresh)
    else:
        raise ValueError(f"Invalid mode: {mode}")

################ Graph for multiple SAEs ################

def create_layered_graph_conditional_prob(activations, lats_for_sents, thresh=0):
    graph = defaultdict(dict)
    assert len(activations) == len(lats_for_sents)

    for i in range(len(activations)-1):
        activations_from = activations[i]
        activations_to = activations[i+1]
        lats_for_sents_from = lats_for_sents[i]
        lats_for_sents_to = lats_for_sents[i+1]

        for x in activations_from.keys(): # enumerate nodes
            inner_lst = set()
            for sent_tok in activations_from[x]: # for each sentence it activates on
                if sent_tok in lats_for_sents_to:
                    inner_lst.update(lats_for_sents_to[sent_tok])
            for y in inner_lst: # for latents those sentences activate on
                st1, st2 = activations_from[x], activations_to[y]
                intersection = len(st1.intersection(st2))
                union = len(st1)
                if intersection / union > thresh:
                    graph[(i, x)][(i+1, y)] = intersection / union

    sum = 0
    for g in graph.values():
        sum += len(g)
    print("SAE graph generated")
    print("Number of edges", sum)
    # Initialize a NetworkX graph
    G = nx.DiGraph()
    # Add edges and nodes with strength-based thickness
    for node_i, neighbors in graph.items():
        for node_j, strength in neighbors.items():
            G.add_edge(node_i, node_j, weight=strength)
    return G

def create_layered_graph(activations, lats_for_sents, mode="conditional_prob", thresh=0):
    if mode == "conditional_prob":
        return create_layered_graph_conditional_prob(activations, lats_for_sents, thresh)
    else:
        raise ValueError(f"Invalid mode: {mode}")