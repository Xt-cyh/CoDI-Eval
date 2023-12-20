model="gpt-3.5-turbo-1106"  # model name
task="length"  # choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"]
mode='evaluate'  # choices=['evaluate', 'evaluate_few_shot']

python evaluate.py --model=$model --task=$task --mode=$mode