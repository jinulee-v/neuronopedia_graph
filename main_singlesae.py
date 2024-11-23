import argparse
from pathlib import Path

import sys
sys.path.append(".")
from utils.neuronopedia_dump import extract_neuronopedia_features
from utils.graph import create_graph
from utils.visualize import visualize_graph


def main(args):
    Path(f"graphs/singlesae/{args.mode}").mkdir(parents=True, exist_ok=True)

    descriptions, activations, lats_for_sents = extract_neuronopedia_features(args.model, args.layer, args.num_features, args.max_features)
    
    graph = create_graph(activations, lats_for_sents, mode=args.mode, thresh=args.thresh)
    visualize_graph(
        graph,
        title=f'{args.model}_{args.layer}_{args.num_features}',
        labels=[descriptions],
        save_path=f"graphs/singlesae/{args.mode}",
        thresh=args.thresh
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize neuron activations into graphs')

    # SAE configuration
    parser.add_argument('--model', type=str, default='gemma-2-2b-gemmascope-res-16k', help='Model name')
    parser.add_argument('--layer', type=str, default='20-gemmascope-res-16k', help='Layer name')
    parser.add_argument('--max_features', type=int, default=16384, help='Number of max features of the given model/layer/index')

    # Graph configuration
    parser.add_argument('--num_features', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--mode', type=str, default='symmetric_jaccard', choices=["symmetric_jaccard", "conditional_prob"], help='Mode of graph creation')
    parser.add_argument('--thresh', type=float, default=0.3, help='Threshold for edge weights')
    args = parser.parse_args()

    # Default command args
    if args.num_features is None:
        args.num_features = args.max_features
    assert args.num_features <= args.max_features

    main(args)
