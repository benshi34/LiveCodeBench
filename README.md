# LiveCodeBench: Modified for Reasoning-Retrieval

See [When Does Retrieval Help with Reasoning-Intense Tasks?]()

To reproduce other results from this paper, see::

* [USACO](https://github.com/benshi34/USACOnew)
* [BRIGHT](https://github.com/xlang-ai/BRIGHT)

## Installation
You can clone the repository using the following command:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
```

We recommend using poetry for managing dependencies. You can install poetry and the dependencies using the following commands:

```bash
pip install poetry
poetry install
```

The default setup does not install [`vllm`](https://vllm.ai/). To install `vllm` as well you can use:

```bash
poetry install --with with-gpu
```

## Data
We add all retrieval artifacts in `assets/data`.

## Inference and Evaluation
To reproduce paper results, utilize the following command:

`python -m lcb_runner.runner.main --model [INSERT MODEL] --scenario codegeneration --evaluate --n 1 --cot_code_generation --corpus_json [INSERT CORPUS DICT PATH] --num_docs [INSERT NUM DOCS] --corpus_name [INSERT CORPUS NAME] --retrieval_setting 2 --multiprocess 16`

For example, to get optimal retrieval results with 2 documents in context using GPT-4o for inference, utilize the following command:

`python -m lcb_runner.runner.main --model gpt-4o-2024-05-13 --scenario codegeneration --evaluate --n 1 --cot_code_generation --corpus_json assets/data/lcb_optimal_topic_retrieval_p2_corpus_dict.json --num_docs 2 --corpus_name lcb_optimal --retrieval_setting 2 --multiprocess 16`
