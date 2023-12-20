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

from utils.query_chatgpt import query_chatgpt


topics = [
    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop culture', 'daily life', 'family', 
    'fashion_&_style', 'film_&_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming',
    'learning_&_educational', 'music', 'social concern', 'relationships',
    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student life'
]
sentiments = ['anger', 'disgust', 'fear', 'joy', 'negative sentiment', 'neutral sentiment', 'positive sentiment', 'sadness', 'surprise']
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


def get_keywords(args):
    keywords = []
    with open(args.keyword_path, 'r') as f:
        cnt = 0
        for line in f.readlines():
            data = json.loads(line)
            klist = data['words']
            word_selected = random.sample(klist, 2)
            # query llm for word substitution
            flag = False
            for i in range(2):
                word = word_selected[i]
                prompt = f"Use one word or phrase to substitute '{word}', put it in []:"
                response = query_chatgpt(prompt)
                res = response["choices"][0]["message"]["content"]
                pattern = r"\[(.+?)\]"
                match = re.search(pattern, res)
                if match:
                    res = match.group(1)
                else:
                    flag = True
                    break
                data[f'word{i+1}'] = word
                data[f'sub{i+1}'] = res
            
            if flag:
                continue
            keywords.append(data)
            with open(args.save_path, 'a+') as f:
                f.write(json.dumps(data))
                f.write('\n')
            cnt += 1
            print(f'finish {cnt}')
            time.sleep(1)

    return keywords


def self_instruct(tasks_pool, prompt, pre_nums):
    '''
    expand instructions from seed tasks
    '''
    seed_task_num = len(tasks_pool)
    for idx in range(100):
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
                    tasks_pool.append(txt)
                    f.write(json.dumps(data))
                    f.write('\n')

        if len(tasks_pool) >= 100:
            break

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

            data = {'instruction': res}
            # filter rep
            rouge = Rouge()
            scores = []
            for i in diversified_instructions:
                scores.append(rouge.get_scores(res, i['instruction'], avg=True)['rouge-l']['f'])
            if len(scores) > 0 and max(scores) > 0.95:
                continue

            # filter no {keywords} and {nonkeywords}
            if ('{keywords}' not in data['instruction']):
                continue

            tot_instructions += 1
            diversified_instructions.append(data)
            # save instructions
            with open(args.target_path, 'a+') as f:
                f.write(json.dumps(data))
                f.write('\n')
            print(f'-----extend {len(diversified_instructions)} instructions-----')

        if tot_instructions >= 900:
            break

    return diversified_instructions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2023, type=int)
    # two seed_tasks, running once each
    parser.add_argument("--seed_tasks", default='./seed_task/keyword/keyword.json', type=str)
    parser.add_argument("--source_path", default='./instructions/keyword/keyword.jsonl', type=str)
    parser.add_argument("--target_path", default='./instructions/keyword/keyword_diversified.jsonl', type=str)
    parser.add_argument("--mode", default='self-instruct', type=str)

    args = parser.parse_args()
    set_seed(args)

    if args.mode == 'self-instruct':
        prompt_all = "In the instructions of keyword controlled text generation, you need to include a series of keywords and a word that cannot be included. " \
        "The generated text is required to be just a sentence or a short paragraph, represented by a text, a sentence, etc." \
        "The keywords is represented by {keywords} in the instructions of the keyword controlled text generation task. " \
        "The word that cannot be included is represented by {nonkeywords} in the instructions of the keyword controlled text generation task. " \
        "\n\nPlease comp up with 10 diverse keyword controlled text generation instructions, be concise:\n"

        prompt = "In the instructions of keyword controlled text generation, you need to include a series of keywords. " \
        "The generated text is required to be just a sentence or a short paragraph, represented by a text, a sentence, etc." \
        "The keywords is represented by {keywords} in the instructions of the keyword controlled text generation task. " \
        "\n\nPlease comp up with 10 diverse keyword controlled text generation instructions, be concise, put it in []:\n"

        seed_tasks = json.load(open(args.seed_tasks, 'r'))
        # save instructions
        with open(args.source_path, 'a+') as f:
            for txt in seed_tasks:
                data = {'instruction': txt}
                f.write(json.dumps(data))
                f.write('\n')
        instructions = self_instruct(seed_tasks, prompt, 8)

    elif args.mode == 'diversify':
        seed_tasks = []
        with open(args.source_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                seed_tasks.append(data['instruction'])

        pre_nums=3
        prompt_non = "In the instructions of keyword controlled text generation, you need to include a series of keywords and a word that cannot be included. " \
        "The generated text is required to be just a sentence or a short paragraph, represented by a text, a sentence, etc." \
        "The keywords is represented by {{keywords}} in the instructions of the keyword controlled text generation task. " \
        "The word that cannot be included is represented by {{nonkeywords}} in the instructions of the keyword controlled text generation task. " \
        "Note that the rewritten text is placed in [], for example, A:[]." \
        "\n\nPlease rewrite the keyword controlled text generation instructions as required:\n" \
        "Q:'You should generate a sentence that satisfies such requirements: contains {{keywords}}; excludes {{nonkeywords}}' Rewrite this instruction, be concise\nA:[Compose a sentence with {{keywords}} only, without {{nonkeywords}}.]\n" \
        "Q:'Please compose a text comprising of {{keywords}} but excluding {{nonkeywords}}.' Rewrite this instruction, be informal\nA:[Write a message using {{keywords}} but don't include {{nonkeywords}}. Let's see what you come up with!]\n" \
        "Q:'Produce a text utilizing {{keywords}}, while making certain that {{nonkeywords}} is not present within the text.' Rewrite this instruction\nA:[Generate a text with the given keywords: {{keywords}} but ensure {{nonkeywords}} is not included in it.]\n" \
        "Q:'{instruction}' Rewrite this instruction{style}\nA:"

        prompt = "In the instructions of keyword controlled text generation, you need to include a series of keywords. " \
        "The generated text is required to be just a sentence or a short paragraph, represented by a text, a sentence, etc." \
        "The keywords is represented by {{keywords}} in the instructions of the keyword controlled text generation task. " \
        "Note that the rewritten text is placed in [], for example, A:[]." \
        "\n\nPlease rewrite the keyword controlled text generation instructions as required:\n" \
        "Q:'You should generate a sentence that satisfies such requirements: contains {{keywords}}' Rewrite this instruction, be concise\nA:[Compose a sentence with {{keywords}} only.]\n" \
        "Q:'Please compose a text comprising of {{keywords}}.' Rewrite this instruction, be informal\nA:[Write a message using {{keywords}}. Let's see what you come up with!]\n" \
        "Q:'Produce a text utilizing {{keywords}}.' Rewrite this instruction\nA:[Generate a text with the given keywords: {{keywords}}.]\n" \
        "Q:'{instruction}' Rewrite this instruction{style}\nA:"
        instructions = diversify(seed_tasks, prompt, pre_nums)

