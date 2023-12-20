model="gpt-3.5-turbo-1106"
task="length"
mode='evaluate'

python evaluate.py --model=$model --task=$task --mode=$mode