import json
import time
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest

from utils.length_eval import number2word, get_mode, get_interval
from utils.eval_utils import *


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sentiments = ['anger', 'disgust', 'fear', 'joy', 'negative sentiment', 'neutral sentiment', 'positive sentiment', 'sadness', 'surprise']
topics = [
    'arts_&_culture', 'business_&_entrepreneurs', 'celebrity_&_pop culture', 'daily life', 'family', 
    'fashion_&_style', 'film_&_tv_&_video', 'fitness_&_health', 'food_&_dining', 'gaming',
    'learning_&_educational', 'music', 'social concern', 'relationships',
    'science_&_technology', 'sports', 'travel_&_adventure', 'youth_&_student life'
]
length_mode = ['equal to', 'at most', 'at least', 'between', 'around']
LLMs = ["gpt-3.5-turbo-1106"]


def get_data(path):
    tasks = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


def cal_max_avg_toxicity(tasks):
    toxicity = {}
    for i in range(len(tasks)):
        t = tasks[i]
        context_id = i // 20
        if context_id not in toxicity.keys():
            toxicity[context_id] = [t['toxicity']]
        else:
            toxicity[context_id].append(t['toxicity'])
    max_sum = 0
    toxic_num = 0
    for i in range(203):
        max_toxicity = max(toxicity[i])

        if max_toxicity < 0:
            max_toxicity = 1
        if max_toxicity >= 0.5:
            toxic_num += 1
        max_sum += max_toxicity

    avg_max = np.around(max_sum / 203, decimals=4)
    nontoxic_prob = np.around(toxic_num / 203, decimals=4)
    return avg_max, nontoxic_prob


def stat_all(args):
    results_df = pd.DataFrame(
        columns=['sentiment', 'topic', 'multi-aspect', 'length', 'keyword', 'detoxification', 'average'], 
        index=LLMs
    )

    for model in LLMs:
        avg_accuracy = 0

        ### statistic sentiment
        task = 'sentiment'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        # count
        correct = {s: 0 for s in sentiments}
        total = {s: 0 for s in sentiments}
        for i in tqdm(range(len(instructions)), desc=task):
            instruction = instructions[i]
            total[instruction['label']] += 1
            correct[instruction['label']] += int(instruction['success'])
        tot_correct = 0
        for s in sentiments:
            tot_correct += correct[s]
            correct[s] = np.around(correct[s]/total[s], decimals=4)
        avg_accuracy += np.around(tot_correct/len(instructions), decimals=4)
        # write to statistics file
        results_df.loc[model, task] = 100 * np.around(tot_correct/len(instructions), decimals=4)

        ### statistic topic
        task = 'topic'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        # count
        correct = {s: 0 for s in topics}
        total = {s: 0 for s in topics}
        for i in tqdm(range(len(instructions)), desc=task):
            instruction = instructions[i]
            total[instruction['label']] += 1
            correct[instruction['label']] += int(instruction['success'])
        tot_correct = 0
        for s in topics:
            tot_correct += correct[s]
            correct[s] = np.around(correct[s]/total[s], decimals=4)

        avg_accuracy += np.around(tot_correct/len(instructions), decimals=4)
        # write to statistics file
        results_df.loc[model, task] = 100 * np.around(tot_correct/len(instructions), decimals=4)

        ### statistic length
        task = 'length'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        length_lite = get_data('./data/instructions/length.jsonl')
        # count
        correct = {s: 0 for s in length_mode}
        total = {s: 0 for s in length_mode}
        for i in tqdm(range(len(instructions)), desc=task):
            instruction = instructions[i]
            mode = length_lite[i]['mode']
            total[mode] += 1
            correct[mode] += int(instruction['success'])
        tot_correct = 0
        for s in length_mode:
            tot_correct += correct[s]
            correct[s] = np.around(correct[s]/total[s], decimals=4)

        avg_accuracy += np.around(tot_correct/len(instructions), decimals=4)
        # write to statistics file
        results_df.loc[model, task] = 100 * np.around(tot_correct/len(instructions), decimals=4)

        ### statistic multi-aspect
        task = 'multi'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        # count
        tot_correct = 0
        correct_topic = {s: 0 for s in topics}
        total_topic = {s: 0 for s in topics}
        correct_senti = {s: 0 for s in sentiments}
        total_senti = {s: 0 for s in sentiments}
        for i in tqdm(range(len(instructions)), desc=task):
            instruction = instructions[i]
            total_topic[instruction['label1']] += 1
            total_senti[instruction['label2']] += 1
            correct_topic[instruction['label1']] += int(instruction['success1'])
            correct_senti[instruction['label2']] += int(instruction['success2'])
            tot_correct += int(instruction['success'])
        tot_topic_correct = 0
        for s in topics:
            tot_topic_correct += correct_topic[s]
            correct_topic[s] = np.around(correct_topic[s]/total_topic[s], decimals=4)
        tot_sentiment_correct = 0
        for s in sentiments:
            tot_sentiment_correct += correct_senti[s]
            correct_senti[s] = np.around(correct_senti[s]/total_senti[s], decimals=4)

        avg_accuracy += np.around(tot_correct/len(instructions), decimals=4)
        # write to statistics file
        results_df.loc[model, task] = 100 * np.around(tot_correct/len(instructions), decimals=4)

        ### statistic keyword
        task = 'keyword'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        # count
        correct_easy, correct_hard = 0, 0
        for i in tqdm(range(500), desc=task):
            correct_easy += int(instructions[i]['success'])
            correct_hard += int(instructions[i+500]['success'])

        avg_accuracy += np.around((correct_easy+correct_hard) / len(instructions), decimals=4)
        # write to statistics file
        results_df.loc[model, task] = 100 * np.around((correct_easy+correct_hard) / len(instructions), decimals=4)

        ### statistic detoxification
        task = 'detoxic'
        eval_data = f"./{args.save_fold}/{model}/{task}/eval_results.jsonl"
        instructions = get_data(eval_data)
        # count
        avg_max, nontoxic_prob = cal_max_avg_toxicity(instructions)
        # write to statistics file
        results_df.loc[model, task] = 100 * (1-nontoxic_prob)

        # average
        avg_accuracy += 1-nontoxic_prob
        results_df.loc[model, 'average'] = 100 * np.around(avg_accuracy / 6, decimals=4)

    results_df = results_df.dropna(axis=1)
    results_df = results_df.applymap(lambda x: '{:.4g}'.format(x))
    results_df = results_df.sort_values('average', ascending=False)
    results_df = results_df.reindex(columns=['sentiment', 'topic', 'multi', 'length', 'keyword', 'detoxic', 'average'])
    results_df.to_csv(f'./{args.save_fold}/statistic.csv')


def error_analysis_keyword(args):
    eval_data = f"./{args.save_fold}/{args.model}/keyword/eval_results.jsonl"
    instructions = get_data(eval_data)
    instructions = instructions[500:]
    total = 500
    false = 0
    error = [0,0,0,0,0]

    for data in instructions:
        data['text'] = extract_text(data['text']).strip()
        word_list = extract_words(data['text'])
        klist = []
        for word in data['label']:
            if ' or ' in word:
                word = word.split(' or ')
                klist.append(get_lemma(word[0]) + ' or ' + get_lemma(word[1]))
            else:
                klist.append(get_lemma(word))

        wrong = []
        if data["success"] == '0':
            false += 1
            if 'anti_label' in data.keys():
                # more complicate case
                for word in klist:
                    word = word.lower()
                    if ' or ' in word:
                        word = word.split(' or ')
                        if word[0] not in word_list and word[1] not in word_list:
                            wrong.append('select')
                            break
                    else:
                        if word not in word_list:
                            wrong.append('include')
                            break
                if get_lemma(data['anti_label']) in word_list:
                    wrong.append('exclude')
        
        if len(wrong) == 1 and wrong[0] == 'include':
            error[0] += 1
        elif len(wrong) == 1 and wrong[0] == 'select':
            error[1] += 1
        elif len(wrong) == 1 and wrong[0] == 'exclude':
            error[2] += 1
        elif len(wrong) == 2:
            error[3] += 1
        elif len(wrong) == 3:
            error[4] += 1

    print(error)
    print(f'not include: {np.around(error[0]/false, decimals=4)}')
    print(f'not select one: {np.around(error[1]/false, decimals=4)}')
    print(f'not exclude: {np.around(error[2]/false, decimals=4)}')
    print(f'2: {np.around(error[3]/false, decimals=4)}')
    print(f'3: {np.around(error[4]/false, decimals=4)}')


def error_analysis_length(args):
    eval_data = f"./{args.save_fold}/{args.model}/length/eval_results.jsonl"
    results = get_data(eval_data)
    instructions = get_data('./data/instructions/length.jsonl')

    delta_equal = []
    delta_about = []
    for i in range(len(results)):
        data = results[i]
        task = instructions[i]
        if task['mode'] == 'equal to':
            target = data['label'][0]
            data['text'] = extract_text(data['text']).strip()
            generate_l = len(data['text'].split())
            delta_equal.append(np.around((generate_l-target)/target, decimals=4))
        elif task['mode'] == 'around':
            target = (data['label'][0] + data['label'][1]) / 2
            data['text'] = extract_text(data['text']).strip()
            generate_l = len(data['text'].split())
            delta_about.append(np.around((generate_l-target)/target, decimals=4))

    # histogram
    fig, ax = plt.subplots(figsize=(8, 3.2))

    p1 = kstest(delta_equal, cdf="norm")
    p2 = kstest(delta_about, cdf="norm")
    ax.hist(delta_equal, bins=20, alpha=0.4, color='orange', label=f'Equal to')
    ax.hist(delta_about, bins=100, alpha=0.4, color='green', label=f'Around')
    ax.set_xlabel('Length error ratio', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=13)
    ax.set_xlim(-1, 1.5)

    plt.tight_layout()
    plt.savefig(f'./pictures/length/length_delta.pdf')


def error_analysis_multi(args):
    eval_data = f"./{args.save_fold}/{args.model}/multi/eval_results.jsonl"
    instructions = get_data(eval_data)

    error = [0,0,0]
    false = 0
    for data in instructions:
        if data["success"] == '0':
            false += 1
            if data["success1"] == '0' and data["success2"] == '1':
                error[0] += 1
            elif data["success1"] == '1' and data["success2"] == '0':
                error[1] += 1
            elif data["success1"] == '0' and data["success2"] == '0':
                error[2] += 1

    print(error)
    print(f'Only topic: {np.around(error[0]/false, decimals=4)}')
    print(f'Only sentiment: {np.around(error[1]/false, decimals=4)}')
    print(f'Both: {np.around(error[2]/false, decimals=4)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_fold", default='results_zero_shot', type=str, choices=['results_zero_shot', 'results_few_shot'])
    parser.add_argument("--mode", default='stat', type=str, choices=['stat', 'analysis'])

    args = parser.parse_args()

    # list the LLMs you want to evaluate
    LLMs = ["gpt-3.5-turbo-1106"]

    if args.mode == 'stat':
        stat_all(args)
    elif args.mode == 'analysis':
        error_analysis_keyword(args)
        error_analysis_length(args)
        error_analysis_multi(args)
