import os
import re
import json
import random
import time
import argparse
import torch
import openai
import matplotlib.pyplot as plt
from scipy.special import expit, softmax
from tqdm import tqdm
import pandas as pd
import numpy as np
from rouge import Rouge

from transformers import AutoModelForSequenceClassification
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

from utils.query_huggingface import query, parse_topic_dict, parse_sentiment_dict
from utils.perspective import detect_toxic
from utils.query_llms import *
from utils.eval_utils import *


topic_label = {
    'arts_&_culture':0, 'business_&_entrepreneurs':1, 'celebrity_&_pop culture':2, 'daily life':3, 'family':4, 
    'fashion_&_style':5, 'film_&_tv_&_video':6, 'fitness_&_health':7, 'food_&_dining':8, 'gaming':9,
    'learning_&_educational':10, 'music':11, 'social concern':12, 'other_hobbies':13, 'relationships':14,
    'science_&_technology':15, 'sports':16, 'travel_&_adventure':17, 'youth_&_student life':18
}
label_topic = {value: key for key, value in topic_label.items()}
label_sentiment1 = {
    0:'negative sentiment', 1:'neutral sentiment', 2:'positive sentiment'
}
label_sentiment2 = {
    0:'anger', 1:'disgust', 2:'fear', 3:'joy', 4:'neutral sentiment', 5:'sadness', 6:'surprise'
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_test_data(args):
    tasks = []
    with open(f'./data/instructions/{args.task}.jsonl', 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


def query_api_model(args, prompt, system_content):
    if 'gpt' in args.model:
        return query_chatgpt(model=args.model, prompt=prompt, system_content=system_content, max_tokens=args.max_len, top_p=args.top_p)
    elif 'qwen' in args.model:
        return query_qwen(args, prompt, system_content)


def get_open_model(args):
    tokenizer = None
    model = None
    if 'GLM' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    elif 'baichuan' in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    return model, tokenizer


def generate(args):
    system_content = "You are performing a test of controlled text generation. Generate text according to the following instruction and generate whatever you want, no other requirements:"
    
    result_dir = f'./{args.save_fold}/{args.model}/{args.task}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = result_dir + f'generation.jsonl'

    if args.mode == 'generate_few_shot':
        # read few shot demos
        system_content = "You are performing a test of controlled text generation. Generate text according to the following 6 instruction and generate whatever you want, no other requirements:"
        with open(f'./data/few_shot/{args.task}.txt', 'r') as f:
            few_shot_examples = f.read()

    instructions = get_test_data(args)
    tokenizer = None

    if args.task == 'detoxic':
        args.max_len = 75

    if args.model_category == 'api':
        for i in tqdm(range(len(instructions))):
            task = instructions[i]
            prompt = None
            if args.mode == 'generate':
                prompt = f"Instruction: {task['instruction']}\nResponse:"
            else:
                prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
            
            res = query_api_model(args, prompt, system_content)

            result_dict = get_result_dict(args, task, res)
            # save results
            with open(result_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    else:
        model, tokenizer = get_open_model(args)
        model.to(args.device)
        model.eval()
        for i in tqdm(range(len(instructions))):
            task = instructions[i]
            prompt = None
            if args.mode == 'generate':
                prompt = f"Instruction: {task['instruction']}\nResponse:"
                if 'llama2' in args.model:
                    prompt = get_llama2_prompt(prompt, system_content)
                else:
                    prompt = system_content + '\n' + prompt
            else:
                prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
                if 'llama2' in args.model:
                    prompt = get_llama2_prompt(prompt, system_content)
                else:
                    prompt = system_content + '\n' + prompt

            if 'GLM' in args.model:
                res = query_chatglm(model, tokenizer, prompt, max_tokens=args.max_len, top_p=args.top_p)
            else:
                context_tokens = tokenizer(prompt, return_tensors='pt')
                input_ids = context_tokens.input_ids.to(args.device)
                attention_mask = context_tokens.attention_mask.to(args.device)
                generation_output = model.generate(
                    input_ids=input_ids,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    output_scores=True,
                    do_sample=True,
                    use_cache=True,
                    max_new_tokens=args.max_len
                )
                output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
                res = output[len(prompt):].strip()

            result_dict = get_result_dict(args, task, res)
            # save results
            with open(result_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model", default='ChatGPT', type=str)
    parser.add_argument("--model_path", default='../models/Llama-2-13b-chat-hf', type=str)
    parser.add_argument("--model_category", default='api', type=str, choices=['api', 'open'])
    parser.add_argument("--task", default='sentiment', type=str, choices=["sentiment", "topic", "multi", "length", "keyword", "detoxic"])
    parser.add_argument("--mode", default='generate', type=str, choices=['generate', 'generate_few_shot'])
    parser.add_argument("--save_fold", default='results_zero_shot', type=str)
    # decoding
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_len", default=300, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    # gpu
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')

    args = parser.parse_args()
    set_seed(args)

    generate(args)
