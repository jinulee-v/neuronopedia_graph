import argparse
from pathlib import Path

import sys
sys.path.append(".")
from utils.neuronopedia_dump import extract_neuronopedia_features
from utils.graph import create_layered_graph
from utils.visualize import visualize_graph

LAYERS = {
    # "gemma-2-2b-gemmascope-res-16k": [f"{i}-gemmascope-res-16k" for i in range(26)],
    "gemma-2-2b-gemmascope-res-16k": [f"{i}-gemmascope-res-16k" for i in range(19,22)],
}

def main(args):
    Path(f"graphs/saeflow/{args.mode}").mkdir(parents=True, exist_ok=True)

    descriptions = []
    activations = []
    lats_for_sents = []
    for layer in LAYERS[args.model]:
        descriptionsh, activationsh, lats_for_sentsh = extract_neuronopedia_features(args.model, layer, args.num_features, args.max_features)
        descriptions.append(descriptionsh)
        activations.append(activationsh)
        lats_for_sents.append(lats_for_sentsh)
    
    graph = create_layered_graph(
        activations,
        lats_for_sents,
        thresh=args.thresh,
        mode=args.mode
    )
    visualize_graph(
        graph,
        title=f'{args.model}_{args.layer_to}',
        labels = descriptions,
        save_path=f"graphs/saeflow/{args.mode}",
        thresh=args.thresh
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize neuron activations into graphs')

    # SAE configuration
    parser.add_argument('--model', type=str, default='gemma-2-2b-gemmascope-res-16k', help='Model name')
    parser.add_argument('--max_features', type=int, default=16384, help='Number of max features of the given model/layer/index')

    # Graph configuration
    parser.add_argument('--num_features', type=int, default=None, help='Number of features to include in the graph') # equivalent to num_nodes
    parser.add_argument('--mode', type=str, default="conditional_prob", choices=["conditional_prob"], help='Mode of graph creation')
    parser.add_argument('--thresh', type=float, default=0.5, help='Threshold for edge weights')
    args = parser.parse_args()

    # Default command args
    assert args.model in LAYERS
    if args.num_features is None:
        args.num_features = args.max_features
    assert args.num_features <= args.max_features

    main(args)
