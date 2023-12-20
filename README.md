# CoDI-Eval

The repository for AAAI 2024 main track paper "Benchmarking Large Language Models on Controllable Generation under Diversified Instructions"

---

## Quick Start

### Setup

We use `python 3.11` in this project. You can create a virtual environment through `conda`. The libraries required in this project are listed in `requirements.txt`

```shell
pip install -r requirements.txt
```

If you need to query OpenAI LLMs, input your OpenAI API Key in `./utils/query_llms.py` (line 16, openai.api_key = 'your OpenAI API Key'). Other LLMs that need to be accessed via API can be set up using a similar approach.

If you need to perform an evaluation for the Toxicity Avoidance task, you should input your Perspective API Key in `./utils/perspective.py` (line 11, PERSPECTIVE_API_KEY = 'your Perspective API Key'). The application of Perspective API Key can be found at https://www.perspectiveapi.com/

### Usage

**Step 1:  Generate model output according to instructions**

You first need to change the configuration in `generate.sh`. If you want to use open-source LLMs or your own LLMs, you are required to change the model_path parameter, i.e., the local path or the Huggingface path of the model.

```shell
model="gpt-3.5-turbo-1106"  # model name
model_category="api"  # choices=['api', 'open']
model_path=""  # Only open source models require this parameter
task="length"  # target CTG task, choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode="generate"  # Zero-shot or Few shot, choices=['generate', 'generate_few_shot']

python generate.py --model=$model --model_category=$model_category --task=$task --mode='generate' --model_path=$model_path
```

**Step 2: Generate evaluation results**

After generating the model output, you can then use `evaluate.sh` to generate the evaluation results.

```shell
model="gpt-3.5-turbo-1106"  # model name
task="length"  # target CTG task, choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode='evaluate'  # Zero shot or Few shot, choices=['evaluate', 'evaluate_few_shot']

python evaluate.py --model=$model --task=$task --mode=$mode
```

**Step 3: Generate evaluation results**

Once LLM's evaluation results on all CTG tasks have been generated, the final CoDI-Eval score can be obtained using `statistic.py`.

You first need to modify line 324 in `statistic.py` (LLMs = []) and list the LLMs you want to evaluate. Then, depending on whether your experiment results in a zero-shot or a few-shot, run `python statistic.py --save_fold=results_zero_shot` or `python statistic.py --save_fold=results_few_shot`. The final statistical results will be in `f'./{args.save_fold}/statistic.csv'`.