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
sent_style = [
    'Formal', 'informal', 'concise', 'verbose', 'polite', 'causal', 'word conversion'
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


def get_topic_dict():
    topic_dict = {}
    for topic in topics:
        topic_list = topic.split('_&_')
        if len(topic_list) > 1:
            topic_all = ' or '.join(topic_list)
            topic_list.append(topic_all)
        topic_dict[topic] = topic_list
    return topic_dict


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
            tasks_pool.extend(matches)

            print(f'-----extend {len(tasks_pool)} instructions-----')
            # save instructions
            with open(args.source_path, 'a+') as f:
                for txt in matches:
                    data = {'instruction': txt}
                    f.write(json.dumps(data))
                    f.write('\n')

        if len(tasks_pool) >= 100:
            break
    return tasks_pool


def diversify(tasks_pool, prompt, pre_nums):
    '''
    Diversify seed tasks, use iCL
    '''
    diversified_instructions = {
        'arts_&_culture':[], 'business_&_entrepreneurs':[], 'celebrity_&_pop culture':[], 'daily life':[], 'family':[], 
        'fashion_&_style':[], 'film_&_tv_&_video':[], 'fitness_&_health':[], 'food_&_dining':[], 'gaming':[],
        'learning_&_educational':[], 'music':[], 'social concern':[], 'relationships':[],
        'science_&_technology':[], 'sports':[], 'travel_&_adventure':[], 'youth_&_student life':[]
    }
    all_instructions = []
    topic_dict = get_topic_dict()
    rouge = Rouge()

    for idx in range(3000):
        prompt_to_query = ''
        # choose a topic
        topic_selected = random.choice(topics)
        topic_to_query = random.choice(topic_dict[topic_selected])

        example = random.sample(tasks_pool + diversified_instructions[topic_selected], 1)[0]
        if '{topic}' in example:
            example = example.format(topic=topic_to_query)

        style = None
        method = random.random()
        if method < 0.2:
            style = random.choice(sent_style_extend)
        elif method < 0.7:
            style = random.choice(sent_style)

        if style is not None:
            if style == 'word conversion':
                style_query = ', using part-of-speech conversion of words expressing topic.'
            else:
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

            # filter
            rouge = Rouge()
            scores = []
            for i in diversified_instructions[topic_selected]:
                scores.append(rouge.get_scores(res, i, avg=True)['rouge-l']['f'])
            if len(scores) > 0 and max(scores) > 0.95:
                continue

            diversified_instructions[topic_selected].append(res)

            # save instructions
            with open(args.target_path, 'a+') as f:
                data = {'attribute': topic_selected, 'instruction': res}
                f.write(json.dumps(data))
                f.write('\n')

            print(f'-----{topic_selected} has extend {len(diversified_instructions[topic_selected])} instructions-----')
        
        if tot_instructions >= 1900:
            break

    return diversified_instructions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--seed_tasks", default='./seed_task/topic/topic.json', type=str)
    parser.add_argument("--source_path", default='./instructions/topic/topic.jsonl', type=str)
    parser.add_argument("--target_path", default='./instructions/topic/topic_diversified.jsonl', type=str)
    parser.add_argument("--mode", default='self-instruct', type=str)

    args = parser.parse_args()
    set_seed(args)

    if args.mode == 'self-instruct':
        prompt = "The generated text should be generally represented by generalized words such as text, something, etc. " \
        "Do not use words that indicate text categories such as story, poem, summary. Do not use words that limit the length of text such as article, passage, short paragraphs. " \
        "The topic is represented by {topic} in the instructions of the topic-controlled text generation task. " \
        "Note that {topic} is a noun. Note that the topic here is a generalized topic, such as science, news, sports, etc." \
        "\n\nPlease comp up with 10 diverse topic-controlled text generation instructions, be concise:\n"

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
        prompt = "The generated text should be generally represented by generalized words such as text, something, etc." \
        "Do not use words that indicate text categories such as story, poem, summary. Do not use words that limit the length of text such as article, passage, short paragraphs." \
        "Note that you cannot generate instructions about {{topic}} itself, such as 'discussing the importance of {{topic}}'. " \
        "Note that the topic here is a generalized topic, such as science, news, sports, etc. " \
        "Note that the rewritten text is placed in [], for example, A:[]." \
        "\n\nPlease rewrite the topic-controlled text generation instructions as required:\n" \
        "Q:'Please produce a creative output centered around the topic of business:' Rewrite this instruction, be concise\nA:[Create a business-centered creative output:]\n" \
        "Q:'Can you share text related to science with me?' Rewrite this instruction, using part-of-speech conversion of words expressing topic.\nA:[Could you please provide me with a scientific topic text?]\n" \
        "Q:'Please give me text that has a topic of sports:' Rewrite this instruction\nA:[I need a piece of writing that focuses on sports:]\n" \
        "Q:'{instruction}' Rewrite this instruction{style}\nA:"
        instructions = diversify(seed_tasks, prompt, pre_nums)
