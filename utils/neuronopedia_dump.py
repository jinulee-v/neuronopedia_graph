import json
import os
from tqdm import tqdm

def extract_neuronopedia_features(model, layer, num_features, max_features, parallel=False):
    descriptions, sentences, activations, lats_for_sents = {}, {}, {}, {}
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
            activationsh = result["activations"]
            feature_idx = int(result["index"])
            if len(result["explanations"]) > 0:
                descriptions[feature_idx] = result["explanations"][0]["description"]
            else:
                descriptions[feature_idx] = "No description available"
            for sentence in activationsh:
                tokens = sentence['tokens']
                sent = '¢'.join(tokens)
                if sent not in sentences: sentences[sent] = (len(sentences), 1)
                else: sentences[sent] = (sentences[sent][0], sentences[sent][1]+1)
                idx = sentences[sent][0] # sentence index
                nonzero_idxes = [j for j in range(len(sentence['values'])) if sentence['values'][j] > 0]
                if len(nonzero_idxes) > 0:
                    activations[feature_idx] = activations.get(feature_idx, set())
                    for x in nonzero_idxes:
                        activations[feature_idx].add((idx, x))
                        lats_for_sents[(idx, x)] = lats_for_sents.get((idx, x), set())
                        lats_for_sents[(idx, x)].add(feature_idx)

    return descriptions, activations, lats_for_sents
