## Preparation

1. Download Gemmascope dump (`gemma-2-2b-gemmascope-res-16k.zip`, `gemma-2-2b-gemmascope-res-65k.zip`) from [Neuronpedia dump](https://neuronpedia-exports.s3.amazonaws.com/index.html).

2. Unzip the files in the following directory.

data \\
├ gemma-2-2b-gemmascope-res-16k
│ └ 0-gemmascope-res-16k
│ ...
└ gemma-2-2b-gemmascope-res-65k

3. Install Python dependencies.

```sh
pip install -r requirements.txt
```

## Running clustering experiments

### Single SAE co-firing analysis

```sh
python main_singlesae.py --layer 20-gemmascope-res-16k
```

### Two SAEs co-firing analysis

```sh
python main_twosae.py --layer_from 20-gemmascope-res-65k --max_features_from 65536 --layer_from 20-gemmascope-res-16k --max_features_from 16384
```