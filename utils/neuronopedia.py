import http.client
from urllib.parse import urlparse
import json
import os
import dotenv
dotenv.load_dotenv()

from tqdm import tqdm
import random

headers = { 'X-Api-Key':  os.environ["NEURONOPEDIA_API_KEY"] }

def getResponse(conn):
    res = conn.getresponse()
    # print(res.status, res.reason)  # Should print 200 OK if successful
    if res.status == 307:
        location = res.getheader("Location")
        # print("Redirecting to:", location)

        # Parse the new URL
        url = urlparse(location)
        conn = http.client.HTTPSConnection(url.hostname)

        # Send request to the new URL
        conn.request("GET", url.path, headers=headers)
        res = conn.getresponse()
    data=res.read()
    result = data.decode("utf-8")
    result = json.loads(result)

    conn.close()

    return result

def get_neuronopedia_activations(model, layer, index):
    CACHE_DIR = os.environ.get("CACHE_DIR", ".cache")
    if f"{model}_{layer}_{index}.json" in os.listdir(CACHE_DIR):
        try:
            with open(f"{CACHE_DIR}/{model}_{layer}_{index}.json", "r") as f:
                data = json.load(f)
                if data is not None:
                    return data
        except:
            pass
    conn = http.client.HTTPSConnection("www.neuronpedia.org")
    conn.request("GET", "/api/feature/"+model+"/"+layer+"/"+str(index), headers=headers)
    res = getResponse(conn)

    with open(f"{CACHE_DIR}/{model}_{layer}_{index}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if res is None:
        print(f"Failed to get neuronopedia activations for {model}/{layer}/{index}")
    return res

def extract_neuronopedia_features(model, layer, num_features, max_features):
    sentences, activations, lats_for_sents = {}, {}, {}
    if num_features == max_features:
        rand_subset = range(max_features)
    else:
        rand_subset = random.sample(range(max_features), k=num_features)
    fresh_is = set()
    cnt=0
    for i in tqdm(rand_subset):
        try:
            resh = get_neuronopedia_activations(model, layer, i)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(f"Failed to get neuronopedia activations for {model}/{layer}/{i}")
            print(e)
            continue
        activationsh = resh['activations']
        for sentence in activationsh:
            tokens = sentence['tokens']
            sent = '¢'.join(tokens)
            if sent not in sentences: sentences[sent] = (len(sentences), 1)
            else: sentences[sent] = (sentences[sent][0], sentences[sent][1]+1)
            idx = sentences[sent][0] # sentence index
            nonzero_idxes = [j for j in range(len(sentence['values'])) if sentence['values'][j] > 0]
            if len(nonzero_idxes) > 0:
                activations[i] = activations.get(i, set())
                for x in nonzero_idxes:
                    activations[i].add((idx, x))
                    lats_for_sents[(idx, x)] = lats_for_sents.get((idx, x), set())
                    lats_for_sents[(idx, x)].add(i)
                fresh_is.add(i)
        cnt+=1

    return sentences, activations, lats_for_sents
