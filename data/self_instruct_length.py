import os
import re
import openai
import json
import random
import time
import pandas as pd
import numpy as np
import argparse
import torch
from rouge import Rouge
from tqdm import tqdm

from utils.query_chatgpt import query_chatgpt
from utils.length_eval import number2word, get_mode, get_interval


sent_style = [
    'Formal', 'informal', 'concise', 'verbose', 'polite', 'causal'
]
sent_style_extend = [
    "Flowery","Ornate","Poetic","Sparse","Rambling","Jargon-filled","Technical","Pithy","Witty",
    "Sarcastic","Humorous","Dramatic","Melodramatic","Colloquial","Idiomatic","Figurative",
    "Metaphorical","Symbolic","Rhetorical","Persuasive","Eloquent","Laconic","Elevated","Simplistic","Sententious"
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_tasks(args):
    tasks = []
    with open(args.instruction_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))
    return tasks


def self_instruct(tasks_pool, prompt, pre_nums):
    '''
    expand instructions from seed tasks
    '''
    seed_task_num = len(tasks_pool)
    modes = ['equal to', 'at most', 'at least', 'between', 'around']
    mode_num = {
        'equal to':0, 'at most':0, 'at least':0, 'between':0, 'around':0
    }
    for idx in range(200):
        if idx < 1:
            instruction_selected = random.sample(tasks_pool, pre_nums)
        else:
            instruction_selected = random.sample(tasks_pool[:seed_task_num], pre_nums)
        random.shuffle(instruction_selected)

        pre_instructions = ''
        i = 1
        for instruct in instruction_selected:
            pre_instructions += f'{i}' + ':[' + instruct + ']\n' # Task 
            i += 1
        prompt_to_query = prompt + pre_instructions
        prompt_to_query += f'{i}:'
        print(prompt_to_query)

        n_completions = 1
        response = query_chatgpt(prompt_to_query, n_completions=n_completions)

        # extract new instructions
        for j in range(n_completions):
            res = response["choices"][j]["message"]["content"]
            print(res)
            pattern = r"\[(.+?)\]"
            matches = re.findall(pattern, res)
            # fail to generate new instructions
            if len(matches) == 0:
                continue

            print(f'-----extend {len(tasks_pool)} instructions-----')
            # save instructions
            with open(args.source_path, 'a+') as f:
                for txt in matches:
                    data = {'instruction': txt}
                    # parser length control mode
                    mode = get_mode(txt)
                    if mode in ['equal to', 'at most', 'at least', 'between', 'around']:
                        data['mode'] = mode
                    else:
                        continue
                    # label balance
                    mode_num[mode] += 1
                    if mode_num[mode] > 16:
                        continue
                    tasks_pool.append(txt)
                    f.write(json.dumps(data))
                    f.write('\n')

        if len(tasks_pool) >= 100:
            break
        time.sleep(1)
    return tasks_pool


def diversify(tasks_pool, prompt, pre_nums):
    '''
    Diversify seed tasks, use iCL
    '''
    diversified_instructions = []
    tot_instructions = 0

    for idx in range(3000):
        prompt_to_query = ''
        example = random.sample(tasks_pool + diversified_instructions, 1)[0]
        mode = example['mode']
        example = example['instruction']

        style = None
        method = random.random()
        if method < 0.2:
            style = random.choice(sent_style_extend)
        elif method < 0.7:
            style = random.choice(sent_style)

        if style is not None:
            style_query = f', be {style}'
        else:
            style_query = ''
        prompt_to_query = prompt.format(instruction=example, style=style_query)
        print(prompt_to_query)

        n_completions = 1
        response = query_chatgpt(prompt_to_query, n_completions=n_completions)
        
        # extract new instructions
        for j in range(n_completions):
            res = response["choices"][j]["message"]["content"]
            print(res)

            pattern = r"\[(.+?)\]"
            match = re.search(pattern, res)
            # fail to generate new instructions
            if match:
                res = match.group(1)
            else:
                continue

            data = {'instruction': res, 'mode': mode}
            # filter rep
            rouge = Rouge()
            scores = []
            for i in diversified_instructions:
                scores.append(rouge.get_scores(res, i['instruction'], avg=True)['rouge-l']['f'])
            if len(scores) > 0 and max(scores) > 0.95:
                continue

            # filter no {number} or both have {number} and {number1}/{number2}
            if ('{number}' not in data['instruction']) or ('{number1}' not in data['instruction']):
                continue
            if '{number}' in data['instruction'] and ('{number1}' in data['instruction'] or '{number2}' in data['instruction']):
                continue

            tot_instructions += 1
            diversified_instructions.append(data)
            # save instructions
            with open(args.target_path, 'a+') as f:
                f.write(json.dumps(data))
                f.write('\n')
            print(f'-----extend {len(diversified_instructions)} instructions-----')

        if tot_instructions >= 1900:
            break
        time.sleep(1)

    return diversified_instructions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--seed_tasks", default='./seed_task/length/length.json', type=str)
    parser.add_argument("--source_path", default='./instructions/length/length.jsonl', type=str)
    parser.add_argument("--target_path", default='./instructions/length/length_diversified.jsonl', type=str)
    parser.add_argument("--mode", default='diversify', type=str)

    args = parser.parse_args()
    set_seed(args)

    if args.mode == 'self-instruct':
        prompt = "There are five basic categories of length-controlled text generation: 'equal to', 'around', 'at most', 'at least', 'between'. " \
        "In the instructions of length-controlled text generation, you need to include a number represented by {number} in the task instruction. " \
        "If you want to control the length range, provide two numbers, {number1} and {number2}. " \
        "Do not include other requirements, except for length control. " \
        "The counting units for length is only words. Each instruction needs to contain only one control category." \
        "\n\nPlease comp up with 10 length-controlled text generation instructions, be concise:\n"

        seed_tasks = json.load(open(args.seed_tasks, 'r'))
        random.shuffle(seed_tasks)
        # save instructions
        with open(args.source_path, 'a+') as f:
            for txt in seed_tasks:
                data = {'instruction': txt}
                mode = get_mode(txt)
                if mode in ['equal to', 'at most', 'at least', 'between', 'around']:
                    data['mode'] = mode
                else:
                    continue
                f.write(json.dumps(data))
                f.write('\n')
        instructions = self_instruct(seed_tasks, prompt, 8)

    elif args.mode == 'diversify':
        seed_tasks = []
        with open(args.source_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                seed_tasks.append(data)

        pre_nums = 3
        prompt = "There are five basic categories of length-controlled text generation: 'equal to', 'around', 'at most', 'at least', 'between'. " \
        "In the instructions of length-controlled text generation, you need to include a number represented by {{number}} in the task instruction. " \
        "If you want to control the length range, provide two numbers, {{number1}} and {{number2}}. " \
        "Do not include other requirements, except for length control. " \
        "The counting units for length is only words. Each instruction needs to contain only one control category. " \
        "Note that the rewritten text is placed in [], for example, A:[]." \
        "\n\nPlease rewrite the length-controlled text generation instructions as required:\n" \
        "Q:'Provide me with a very short text, no longer than {{number}} words:' rewrite this instruction, be concise\nA:[Write less than {{number}} words text:]\n" \
        "Q:'Please create a piece of writing that is between {{number1}} to {{number2}} words long.' rewrite this instruction, be informal\nA:[Hey there! Write something for me, just make sure it's not too short or too long. Maybe between {{number1}} to {{number2}} words? Thanks!]\n" \
        "Q:'I require a text about {{number}} words:' rewrite this instruction\nA:[Could you please provide me with a text of around {{number}} words long?]\n" \
        "Q:'{instruction}' rewrite this instruction{style}\nA:"
        instructions = diversify(seed_tasks, prompt, pre_nums)

