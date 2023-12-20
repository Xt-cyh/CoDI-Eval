model="gpt-3.5-turbo-1106"  # model name
model_category="api"  # choices=['api', 'open']
model_path=""  # Only open source models require this parameter
task="length"  # choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode="generate"  # choices=['generate', 'generate_few_shot']

python generate.py --model=$model --model_category=$model_category --task=$task --mode='generate' --model_path=$model_path
