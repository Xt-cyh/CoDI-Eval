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
LLMs = ["ChatGPT","WizardLM","llama2-13b-chat","llama2-7b-chat","ChatGLM-6B","ChatGLM2-6B","llama-7B","llama-13B","llama2-7B","llama2-13B","alpaca","baichuan-7B","baichuan-13B-chat","vicuna-7B","vicuna-13B","gpt4all-13b-snoozy","rwkv-7B","rwkv-14B"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_test_data(args):
    tasks = []
    with open(args.test_data, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


def generate(args):
    system_content = "You are performing a test of controlled text generation. Generate text according to the following 6 instruction and generate whatever you want, no other requirements:"
    result_dir = f'./{args.save_fold}/{args.model}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_path = result_dir + f'{args.task}.jsonl'

    if args.mode == 'generate_few_shot':
        # read few shot examples
        with open(f'./data/few_shot/{args.task}.txt', 'r') as f:
            few_shot_examples = f.read()

    instructions = get_test_data(args)
    tokenizer = None
    max_tokens=300

    if args.task == 'detoxic':
        max_tokens = 75
        args.max_len = 75

    if args.model == 'ChatGPT':
        for i in tqdm(range(len(instructions))):
            task = instructions[i]
            prompt = None
            if args.mode == 'generate':
                prompt = f"Instruction: {task['instruction']}\nResponse:"
            else:
                prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
            response = query_chatgpt(prompt, system_content=system_content, max_tokens=max_tokens, top_p=args.top_p)
            res = response["choices"][0]["message"]["content"]

            result_dict = get_result_dict(args, task, res)
            # save results
            with open(result_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')
        return

    elif 'ChatGLM' in args.model:
        # load model
        tokenizer = None
        model = None
        if args.model == 'ChatGLM-6B':
            tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
            model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
        else:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
            model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
        model.to(args.device)
        model = model.eval()

        for i in tqdm(range(len(instructions))):
            task = instructions[i]
            prompt = None
            if args.mode == 'generate':
                prompt = f"Instruction: {task['instruction']}\nResponse:"
            else:
                prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
            prompt = system_content + '\n' + prompt
            response = query_chatglm(model, tokenizer, prompt, max_tokens=max_tokens, top_p=args.top_p)

            result_dict = get_result_dict(args, task, response)
            # save results
            with open(result_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')
        return

    if args.model == 'alpaca': 
        tokenizer = LlamaTokenizer.from_pretrained("../models/alpaca")
        model = LlamaForCausalLM.from_pretrained("../models/alpaca")
    elif args.model == 'vicuna-7B':
        tokenizer = LlamaTokenizer.from_pretrained("../models/vicuna-7B")
        model = LlamaForCausalLM.from_pretrained("../models/vicuna-7B")
    elif args.model == 'vicuna-13B':
        tokenizer = LlamaTokenizer.from_pretrained("../models/vicuna-13B")
        model = LlamaForCausalLM.from_pretrained("../models/vicuna-13B")
    elif args.model == 'gpt4all-13b-snoozy':
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/gpt4all-13b-snoozy")
        model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-13b-snoozy")
    elif args.model == 'llama-7B': 
        tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    elif args.model == 'llama-13B': 
        tokenizer = LlamaTokenizer.from_pretrained("../models/llama-13b-hf")
        model = LlamaForCausalLM.from_pretrained("../models/llama-13b-hf")
    elif args.model == 'llama-gpt4-7B':
        tokenizer = AutoTokenizer.from_pretrained("../models/llama_gpt4")
        model = AutoModelForCausalLM.from_pretrained("../models/llama_gpt4")
    elif args.model == 'alpaca-13B': 
        tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-13b")
        model = AutoModelForCausalLM.from_pretrained("chavinlo/alpaca-13b")
    elif args.model == 'alpaca-gpt4-13B':
        tokenizer = AutoTokenizer.from_pretrained("chavinlo/gpt4-x-alpaca")
        model = AutoModelForCausalLM.from_pretrained("chavinlo/gpt4-x-alpaca")
    elif args.model == 'baichuan-7B':
        tokenizer = AutoTokenizer.from_pretrained("../models/baichuan-7B", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("../models/baichuan-7B", device_map="auto", trust_remote_code=True)
    elif args.model == 'baichuan-13B-chat':
        tokenizer = AutoTokenizer.from_pretrained("../models/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("../models/Baichuan-13B-Chat", device_map="auto", trust_remote_code=True)
    elif args.model == 'rwkv-7B':
        model_id = "RWKV/rwkv-raven-7b"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif args.model == 'rwkv-14B':
        model_id = "RWKV/rwkv-raven-14b"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif args.model == 'WizardLM':
        tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardLM-13B-V1.2")
        model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-13B-V1.2")
    elif args.model == 'llama2-7b-chat':
        tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("../models/Llama-2-7b-chat-hf")
    elif args.model == 'llama2-13b-chat':
        tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-13b-chat-hf")
        model = AutoModelForCausalLM.from_pretrained("../models/Llama-2-13b-chat-hf")
    elif args.model == 'llama2-7b':
        tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-7b-hf")
        model = AutoModelForCausalLM.from_pretrained("../models/Llama-2-7b-hf")
    elif args.model == 'llama2-13b':
        tokenizer = AutoTokenizer.from_pretrained("../models/Llama-2-13b-hf")
        model = AutoModelForCausalLM.from_pretrained("../models/Llama-2-13b-hf")

    model.to(args.device)
    model.eval()

    for i in tqdm(range(len(instructions))):
        task = instructions[i]
        prompt = None
        if args.mode == 'generate':
            prompt = f"Instruction: {task['instruction']}\nResponse:"
        else:
            prompt = few_shot_examples + f"\n6.Instruction: {task['instruction']}\nResponse:"
        prompt = system_content + '\n' + prompt
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
    parser.add_argument("--task", default='sentiment', type=str)
    parser.add_argument("--mode", default='cal_sbl', type=str)
    parser.add_argument("--save_fold", default='results_zero_shot', type=str)
    # decoding
    parser.add_argument("--top_k", default=200, type=int)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_len", default=300, type=int)
    parser.add_argument("--min_len", default=300, type=int)
    parser.add_argument("--length_penalty", default=1, type=int)
    parser.add_argument("--repetition_penalty", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    # gpu
    parser.add_argument("--gpudevice", type=str, default='6')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')

    args = parser.parse_args()
    set_seed(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpudevice

    generate(args)
