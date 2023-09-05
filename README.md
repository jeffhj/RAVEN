# RAVEN

This repository contains pretrained models and code for pretraining and evaluation for RAVEN.



## Installation

The codebase uses the following dependencies:

* python 3 (tested with 3.8)
* fairscale (tested with 0.4.6)
* transformers (tested with 4.18.0)
* numpy (tested with 1.22.4)
* faiss (tested with 1.7.2)


***Recommend***: you can use the `Dockerfile` to build a docker image that meets all the above requirements



## Training RAVEN

**Download corpora**

```
python preprocessing/download_corpus.py --corpus enwiki-dec2021 --output_directory ${DATA_DIR}
```



**Download pretrained Atlas checkpoint**

```
python preprocessing/download_model.py --model {model download key} --output_directory ${DATA_DIR} 
```

model download key

- `models/atlas/xl` for Atlas 3B
- `models/atlas/xxl` for Atlas 11B



**Train RAVEN**

xl (3B)

```
sbatch example_scripts/train-raven-xl.sh
```

xxl (11B)

```
sbatch example_scripts/train-raven-xxl.sh
```



*Note*: 

- For the first run, please set `PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}` to initialize model weights with Atlas checkpoint. For the following runs, set `PRETRAINED_MODEL=none` so the latest checkpoint can be loaded
- If there is no prebuilt index, don't use the `load_index_path` option, and an index will be built automatically. You can use the `save_index_path` option to save the index for later usage.
- We use a batch size of 64 for both 3B and 11B model training. batch_size = #nodes $\times$ ntasks-per-node $\times$ per_gpu_batch_size



*Training time:*

- Training RAVEN (xl, 5k steps) takes about 1 day on 8 A100 GPUs
- Training RAVEN (xxl, 5k steps) takes about 2-3 days on 32 A100 GPUs



## In-Context Learning with RAVEN

**NQ**

```
sbatch example_scripts/evaluate-nq.sh
```



**TQA**

```
sbatch example_scripts/evaluate-tqa.sh
```



**MMLU**

```
sbatch example_scripts/evaluate-mmlu.sh
```



*Note*: 

- You may refer to [Atlas](https://github.com/facebookresearch/atlas) to download and preprocess the dataset.
- If there is no prebuilt index, don't use the `load_index_path` option, and an index will be built automatically. You can use the `save_index_path` option to save the index for later usage.
- The default batch_size in the script is for `xl`. Please set a smaller batch_size, e.g., 2, for `xxl`



**Prompting Strategies:**

- ***Standard In-Context Learning***

set `n_shots` and `fusion=0`, e.g., `n_shots=0` for 0-shot, `n_shots=5` for 5-shot



- ***Fusion-In-Context Learning***

set `n_shots` and `fusion`, e.g., `n_shots=64, fusion=5` for [64-shot, 5-fusion]



## Ack and License

This repository is adapted from Meta's [Atlas](https://github.com/facebookresearch/atlas), a retrieval-augmented encoder-decoder language model on which RAVEN is based.

**Code License**

The majority of the code is licensed under [CC-BY-NC](./LICENSE), however portions of the project are available under separate license terms: huggingface transformers is licensed under the [Apache 2.0 license](https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE), which covers `src/modeling_bert.py` and `src/modeling_t5.py`.

**Data License**

The wikipedia-derived data used in the repository, such as the corpora and indices available from `download_corpus.py` and `download_index.py` are licensed according to [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). 