import json
import time
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm
from utils.length_eval import number2word, get_mode, get_interval
from self_instruct_topic import get_topic_dict

sentiments = ['anger', 'disgust', 'fear', 'joy', 'negative sentiment', 'neutral sentiment', 'positive sentiment', 'sadness', 'surprise']
topics = [
    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop culture', 'daily life', 'family', 
    'fashion_&_style', 'film_&_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming',
    'learning_&_educational', 'music', 'social concern', 'relationships',
    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student life'
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def fill(args):
    tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    # save instructions
    with open(args.diversified_path, 'a+') as f:
        for i in range(len(tasks)):
            data = tasks[i]
            if args.mode == 'sentiment':
                sentiment = random.choice(sentiments)
                data['label'] = sentiment
                data['instruction'] = data['instruction'].format(sentiment=sentiment)
            elif args.mode == 'topic':
                topic_dict = get_topic_dict()
                topic_selected = random.choice(topics)
                data['label'] = topic_selected
                topic_to_query = random.choice(topic_dict[topic_selected])
                data['instruction'] = data['instruction'].format(topic=topic_to_query)
            f.write(json.dumps(data))
            f.write('\n')


def fill_length(args):
    tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    with open(args.diversified_path, 'a+') as f:
        for i in tqdm(range(len(tasks))):
            data = tasks[i]
            mode = data['mode']

            if '{number}' in data['instruction'] and ('{number1}' not in data['instruction'] and '{number2}' not in data['instruction']):
                # sentence can't be too long
                if 'sentence' in data['instruction']:
                    if random.random() < 0.5:
                        number = random.randint(1, 3) * 10
                    else:
                        number = random.randint(5, 30)
                else:
                    if random.random() < 0.5:
                        number = random.randint(1, 20) * 10
                    else:
                        number = random.randint(5, 200)
                data['label'] = get_interval(mode, number)
                # 25% number will be transfer to words
                if random.random() < 0.25:
                    number = number2word(number)
                data['instruction'] = data['instruction'].format(number=number)

            # Limit the difference between number1 and number2 to within 100
            elif ('{number1}' in data['instruction'] and '{number2}' in data['instruction']) and '{number}' not in data['instruction']:
                # sentence can't be too long
                if 'sentence' in data['instruction']:
                    if random.random() < 0.5:
                        number2 = random.randint(2, 3) * 10
                        number1 = random.randint(1, (number2//10)-1) * 10
                    else:
                        number2 = random.randint(20, 30)
                        number1 = random.randint(1, number2-10)
                else:
                    if random.random() < 0.5:
                        number2 = random.randint(2, 20) * 10
                        number1 = random.randint(1, (number2//10)-1) * 10
                    else:
                        number2 = random.randint(20, 200)
                        number1 = random.randint(1, number2-10)
                data['label'] = [number1, number2]
                if random.random() < 0.25:
                    number1 = number2word(number1)
                    number2 = number2word(number2)
                data['instruction'] = data['instruction'].format(number1=number1, number2=number2)
            else:
                continue

            if data['label'] is None:
                continue

            f.write(json.dumps(data))
            f.write('\n')


def fill_multi(args):
    seed_tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            seed_tasks.append(json.loads(line))

    # save instructions
    with open(args.save_path, 'a+') as f:
        for i in range(len(seed_tasks)):
            data = seed_tasks[i]
            sentiment = random.choice(sentiments)
            data['label2'] = sentiment
            topic_dict = get_topic_dict()
            topic_selected = random.choice(topics)
            data['label1'] = topic_selected
            topic_to_query = random.choice(topic_dict[topic_selected])
            data['instruction'] = data['instruction'].format(topic=topic_to_query, sentiment=sentiment)
            f.write(json.dumps(data))
            f.write('\n')

    # filter lite
    tasks = []
    with open(args.diversified_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))
    
    tasks = random.sample(tasks, 900)
    with open(args.save_path, 'a+') as f:
        for i in range(len(tasks)):
            data = tasks[i]
            f.write(json.dumps(data))
            f.write('\n')


def fill_keywords(args):
    # read keywords
    keywords = []
    with open('./seed_task/keyword/c2gen_augmentation.jsonl', 'r') as f:
        for line in f.readlines():
            keywords.append(json.loads(line))

    words = {3:0, 4:0, 5:0}
    tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    with open(args.save_path, 'a+') as f:
        for i in tqdm(range(len(tasks))):
            data = tasks[i]
            keyword = keywords[i]
            klist = []
            words[len(keyword['words'])] += 1
            for word in keyword['words']:
                if '{nonkeywords}' in data['instruction']:
                    # more complicate case
                    if word == keyword['word1']:
                        klist.append(f"{word} or {keyword['sub1']}")
                    else:
                        klist.append(word)
                else:
                    klist.append(word)

            pattern = random.random()
            if pattern < 1/3:
                keyword = ', '.join(klist[:-1]) + f', and {klist[-1]}'
            elif pattern < 2/3:
                keyword = ', '.join(klist)
            else:
                keyword = '[' + ', '.join(klist) + ']'

            if '{nonkeywords}' in data['instruction']:
                nonkeyword = keywords[i]['sub2']
                data['instruction'] = data['instruction'].format(keywords=keyword, nonkeywords=nonkeyword)
                data['anti_label'] = nonkeyword
            else:
                data['instruction'] = data['instruction'].format(keywords=keyword)
            data['label'] = klist
            f.write(json.dumps(data))
            f.write('\n')
    print(words)


def fill_detoxic(args):
    # read contexts
    contexts = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            contexts.append(data['prompt'])

    # read instructions
    instructions = json.load(open(args.seed_tasks, 'r'))

    for i in range(len(contexts)):
        context = contexts[i]
        for instruction in instructions:
            with open(args.save_path, 'a+') as f:
                instruction = instruction.format(context=context)
                data = {'instruction':instruction, 'context_id':i, 'context':context}
                f.write(json.dumps(data))
                f.write('\n')


def get_topic_attributes():
    topic_dict = {}
    for topic in topics:
        topic_list = topic.split('_&_')
        if len(topic_list) > 1:
            topic_all = ' or '.join(topic_list)
            topic_list.append(topic_all)
        topic_dict[topic] = topic_list
    print(topic_dict)


def filter_definite(args):
    tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    # save instructions
    with open(args.diversified_path, 'a+') as f:
        for i in range(len(tasks)):
            data = tasks[i]
            if '{number}' in data['instruction'] or ('{number1}' in data['instruction'] and '{number2}' in data['instruction']):
                f.write(json.dumps(data))
                f.write('\n')


def filter_lite(args):
    instructions = []
    with open(args.seed_tasks, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            instructions.append(data['instruction'])

    tasks = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    # save instructions
    with open(args.diversified_path, 'a+') as f:
        for i in tqdm(range(len(tasks))):
            data = tasks[i]
            if data['instruction'] in instructions:
                f.write(json.dumps(data))
                f.write('\n')


def find(args):
    instructions = []
    with open(args.expansion_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            instructions.append(data['instruction'])

    tasks = []
    with open(args.seed_tasks, 'r') as f:
        for line in f.readlines():
            tasks.append(json.loads(line))

    # save instructions
    with open(args.diversified_path, 'a+') as f:
        for i in tqdm(range(len(tasks))):
            data = tasks[i]
            if data['instruction'] not in instructions:
                f.write(json.dumps(data))
                f.write('\n')


def sample(args):
    model = 'ChatGPT'
    test_dir = {'zero_shot':'results_zero_shot', 'few_shot':'results_few_shot'}
    for mode in ['zero_shot', 'few_shot']:
        for task in ['sentiment', 'topic', 'multi', 'length', 'keyword', 'detoxic']:
            test_file = f'./{test_dir[mode]}/{model}/{task}/eval_lite.jsonl'
            
            instructions = []
            with open(test_file, 'r') as f:
                for line in f.readlines():
                    instructions.append(json.loads(line))

            instructions = random.sample(instructions, len(instructions)//10)
            output_file = f'./evaluate/{model}_{mode}_{task}.jsonl'
            with open(output_file, 'w') as f:
                for i in instructions:
                    i['human_label'] = ''
                    f.write(json.dumps(i))
                    f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--seed_tasks", default='./seed_task/length/length.json', type=str)
    parser.add_argument("--expansion_path", default='./instrutions/length/length.jsonl', type=str)
    parser.add_argument("--diversified_path", default='./instrutions/length/length_diversified.jsonl', type=str)
    parser.add_argument("--save_path", default='./instructions/length.jsonl', type=str)
    parser.add_argument("--mode", default='length', type=str)

    args = parser.parse_args()
    set_seed(args)

    if args.mode in ['sentiment', 'topic']:
        fill(args)
    elif args.mode == 'length':
        fill_length(args)
    elif args.mode == 'multi':
        fill_multi(args)
    elif args.mode == 'keyword':
        fill_keywords(args)
    elif args.mode == 'detoxic':
        fill_detoxic(args)
