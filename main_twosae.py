import argparse
from pathlib import Path

import sys
sys.path.append(".")
from utils.neuronopedia_dump import extract_neuronopedia_features
from utils.graph import create_layered_graph
from utils.visualize import visualize_graph


def main(args):
    Path(f"graphs/twosae/{args.mode}").mkdir(parents=True, exist_ok=True)

    descriptions_from, activations_from, lats_for_sents_from = extract_neuronopedia_features(args.model_from, args.layer_from, args.num_features_from, args.max_features_from)
    descriptions_to, activations_to, lats_for_sents_to = extract_neuronopedia_features(args.model_to, args.layer_to, args.num_features_to, args.max_features_to)
    
    graph = create_layered_graph(
        [activations_from, activations_to],
        [lats_for_sents_from, lats_for_sents_to],
        thresh=args.thresh,
        mode=args.mode
    )
    visualize_graph(
        graph,
        title=f'{args.layer_from}_{args.layer_to}',
        labels=[descriptions_from, descriptions_to],
        save_path=f"graphs/twosae/{args.mode}",
        thresh=args.thresh
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize neuron activations into graphs')

    # SAE configuration
    parser.add_argument('--model_from', type=str, default='gemma-2-2b-gemmascope-res-16k', help='Model name')
    parser.add_argument('--model_to', type=str, default='gemma-2-2b-gemmascope-res-16k', help='Model name')
    parser.add_argument('--layer_from', type=str, default='19-gemmascope-res-16k', help='Layer name')
    parser.add_argument('--layer_to', type=str, default='20-gemmascope-res-16k', help='Layer name')
    parser.add_argument('--max_features_from', type=int, default=16384, help='Number of max features of the given model/layer/index')
    parser.add_argument('--max_features_to', type=int, default=16384, help='Number of max features of the given model/layer/index')

    # Graph configuration
    parser.add_argument('--num_features_from', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--num_features_to', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--mode', type=str, default="conditional_prob", choices=["conditional_prob"], help='Mode of graph creation')
    parser.add_argument('--thresh', type=float, default=0.3, help='Threshold for edge weights')
    args = parser.parse_args()

    # Default command args
    if args.num_features_from is None:
        args.num_features_from = args.max_features_from
    assert args.num_features_from <= args.max_features_from
    if args.num_features_to is None:
        args.num_features_to = args.max_features_to
    assert args.num_features_to <= args.max_features_to

    main(args)
