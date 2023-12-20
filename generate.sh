model="gpt-3.5-turbo-1106"
model_category="api"
# Only open source models require this parameter
model_path=''
task="length"
mode='generate'

python generate.py --model=$model --model_category=$model_category --task=$task --mode='generate' --model_path=$model_path
