import argparse
import json

import sys
sys.path.append(".")
from utils.neuronopedia import get_neuronopedia_activations, extract_neuronopedia_features
from utils.bipartite_graph import create_bipartite_graph, visualize_bipartite_graph


def main(args):
    sentences_from, activations_from, lats_for_sents_from = extract_neuronopedia_features(args.model, args.layer_from, args.num_features_from, args.max_features_from)
    sentences_to, activations_to, lats_for_sents_to = extract_neuronopedia_features(args.model, args.layer_to, args.num_features_to, args.max_features_to)
    
    graph = create_bipartite_graph(activations_from, lats_for_sents_from, activations_to, lats_for_sents_to, mode=args.mode)
    visualize_bipartite_graph(
        graph,
        title=f'{args.model}_{args.layer_from}_{args.layer_to}',
        save_path=f"graphs/twosae/{args.mode}",
        thresh=args.thresh
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize neuron activations into graphs')

    # SAE configuration
    parser.add_argument('--model', type=str, default='gemma-2-2b', help='Model name')
    parser.add_argument('--layer_from', type=str, default='20-gemmascope-res-65k', help='Layer name')
    parser.add_argument('--layer_to', type=str, default='20-gemmascope-res-16k', help='Layer name')
    parser.add_argument('--max_features_from', type=int, default=65536, help='Number of max features of the given model/layer/index')
    parser.add_argument('--max_features_to', type=int, default=16384, help='Number of max features of the given model/layer/index')

    # Graph configuration
    parser.add_argument('--num_features_from', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--num_features_to', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--mode', type=str, default="conditional_prob", choices=["conditional_prob"], help='Mode of graph creation')
    parser.add_argument('--thresh', type=float, default=0.1, help='Threshold for edge weights')
    args = parser.parse_args()

    # Default command args
    if args.num_features_from is None:
        args.num_features_from = args.max_features_from
    assert args.num_features_from <= args.max_features_from
    if args.num_features_to is None:
        args.num_features_to = args.max_features_to
    assert args.num_features_to <= args.max_features_to

    main(args)
