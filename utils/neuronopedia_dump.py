import json
import os
from tqdm import tqdm
from collections import defaultdict
import gc

def extract_neuronopedia_features(model, layer, num_features, max_features, parallel=False):
    descriptions, activations, lats_for_sents = {}, defaultdict(set), defaultdict(set)
    assert num_features == max_features

    base_dir = os.path.join("data", model, layer)
    files = os.listdir(base_dir)
    files = [f for f in files if f.endswith(".json")] # Skip Zone.identifier etc

    for file in tqdm(files):
        file = os.path.join(base_dir, file)

        # Load data
        with open(file) as f:
            results = json.load(f)
        for result in results:
            feature_idx = int(result["index"])
            # Extract feature description
            if len(result["explanations"]) > 0:
                descriptions[feature_idx] = result["explanations"][0]["description"]
            else:
                descriptions[feature_idx] = "No description available"
            # Create map between sentence-token pair and feature
            for sentence in result["activations"]:
                sent_id = hash('¢'.join(sentence['tokens'])) # imperfect hash function for optimization
                for j, act_val in enumerate(sentence['values']):
                    if act_val > 0:
                        activations[feature_idx].add((sent_id, j, act_val))
                        lats_for_sents[(sent_id, j)].add(feature_idx)
        del results

    return descriptions, activations, lats_for_sents
