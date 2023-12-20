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
LLMs = ['gpt-3.5-turbo-1106']

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def get_test_data(data_path):
    tasks = []
    with open(data_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            tasks.append(data)
    return tasks


def eval_topic(args, text, label):
    while 1:
        response = query('cardiffnlp/tweet-topic-21-multi', text)
        if type(response) == list:
            break
        else:
            time.sleep(10)
    res_dict = parse_topic_dict(response)
    first_key = next(iter(res_dict))
    target_value = res_dict[label]
    return first_key, target_value >= 0.5


def eval_sentiment(args, text, label):
    model = None
    if label in ['negative sentiment', 'neutral sentiment', 'positive sentiment']:
        model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    elif label in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        model = 'j-hartmann/emotion-english-roberta-large'
    while 1:
        response = query(model, text)
        if type(response) == list:
            break
        else:
            time.sleep(10)
    res_dict = parse_sentiment_dict(response)
    first_key = next(iter(res_dict))
    first_value = res_dict[first_key]
    return first_key, first_key == label


def classify_topic(args, model, tokenizer, text, label):
    '''
    refer to https://huggingface.co/cardiffnlp/tweet-topic-21-multi
    '''
    tokens = tokenizer(text, return_tensors='pt').to(args.device)
    if tokens.input_ids.shape[1] > 512:
        tokens.input_ids = tokens.input_ids[:, :512]
        tokens.attention_mask = tokens.attention_mask[:, :512]

    output = model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
    output = output[0][0].detach().cpu()
    scores = output.numpy()
    scores = expit(scores)
    pred = np.argmax(scores)
    predictions = (scores >= 0.5) * 1
    return label_topic[pred], predictions[topic_label[label]]


def classify_sentiment(args, model1, model2, tokenizer1, tokenizer2, text, label):
    '''
    refer to https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    '''
    model, tokenizer = None, None
    if label in ['negative sentiment', 'neutral sentiment', 'positive sentiment']:
        model = model1
        tokenizer = tokenizer1
    elif label in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        model = model2
        tokenizer = tokenizer2

    encoded_input = tokenizer(text, return_tensors='pt').to(args.device)
    if encoded_input.input_ids.shape[1] > 512:
        encoded_input.input_ids = encoded_input.input_ids[:, :512]
        encoded_input.attention_mask = encoded_input.attention_mask[:, :512]

    output = model(input_ids=encoded_input.input_ids, attention_mask=encoded_input.attention_mask)
    output = output[0][0].detach().cpu()
    scores = output.numpy()
    scores = softmax(scores)
    pred = np.argmax(scores)
    if len(scores) == 3:
        return label_sentiment1[pred], label_sentiment1[pred] == label
    elif len(scores) == 7:
        return label_sentiment2[pred], label_sentiment2[pred] == label


def evaluate(args):
    # for each instructions, use eval model to judge the response satisifies the requirement or not
    if args.mode == 'evaluate_few_shot':
        eval_dir = f'./results_few_shot/{args.model}/{args.task}/'
    else:
        eval_dir = f'./results_zero_shot/{args.model}/{args.task}/'

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    eval_path = eval_dir + 'eval_results.jsonl'
    data_path = eval_dir + 'generation.jsonl'
    results = get_test_data(data_path)
    result_dict = {}
    success_num = 0

    if args.task == 'topic':
        # load models
        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to(args.device)

        for i in tqdm(range(len(results))):
            task = results[i]
            task['text'] = extract_text(task['text']).strip()
            first_key, correct = classify_topic(args, model, tokenizer, task['text'], task['label'])
            # use api
            # first_key, correct = eval_topic(res, task['label'])
            result_dict = {
                'instruction': task['instruction'], 'text': task['text'], 'label': task['label'], 
                'generate_label': first_key, 'success': str(int(correct))
            }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    elif args.task == 'sentiment':
        # load models 1
        MODEL1 = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
        model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)
        model1.to(args.device)
        # load models 2
        MODEL2 = f"j-hartmann/emotion-english-roberta-large"
        tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
        model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)
        model2.to(args.device)

        for i in tqdm(range(len(results))):
            task = results[i]
            task['text'] = extract_text(task['text']).strip()
            first_key, correct = classify_sentiment(args, model1, model2, tokenizer1, tokenizer2, task['text'], task['label'])
            # use api
            # first_key, correct = eval_topic(task['text'], task['label'])
            result_dict = {
                'instruction': task['instruction'], 'text': task['text'], 'label': task['label'], 
                'generate_label': first_key, 'success': str(int(correct))
            }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    elif args.task == 'multi':
        # load models
        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to(args.device)
        # load models 1
        MODEL1 = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer1 = AutoTokenizer.from_pretrained(MODEL1)
        model1 = AutoModelForSequenceClassification.from_pretrained(MODEL1)
        model1.to(args.device)
        # load models 2
        MODEL2 = f"j-hartmann/emotion-english-roberta-large"
        tokenizer2 = AutoTokenizer.from_pretrained(MODEL2)
        model2 = AutoModelForSequenceClassification.from_pretrained(MODEL2)
        model2.to(args.device)

        for i in tqdm(range(len(results))):
            task = results[i]
            task['text'] = extract_text(task['text']).strip()
            first_key1, correct1 = classify_topic(args, model, tokenizer, task['text'], task['label1'])
            first_key2, correct2 = classify_sentiment(args, model1, model2, tokenizer1, tokenizer2, task['text'], task['label2'])
            correct = (correct1 + correct2 == 2)
            result_dict = {
                    'instruction': task['instruction'], 'text': task['text'], 'label1': task['label1'], 'label2': task['label2'], 
                    'generate_label1': first_key1, 'generate_label2': first_key2, 'success1': str(int(correct1)), 'success2': str(int(correct2)),
                    'success': str(int(correct))
                }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    elif args.task == 'length':
        for i in tqdm(range(len(results))):
            task = results[i]
            task['text'] = extract_text(task['text']).strip()
            interval = task['label']
            length = len(task['text'].split())
            correct = 0

            if interval[0] <= length and length <= interval[1]:
                success_num += 1
                correct = 1
            result_dict = {
                'instruction': task['instruction'], 'text': task['text'], 'label': task['label'], 
                'generate_length': length, 'success': str(int(correct))
            }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    elif args.task == 'keyword':
        for i in tqdm(range(len(results))):
            task = results[i]
            task['text'] = extract_text(task['text']).strip()
            word_list = extract_words(task['text'])
            # convert all words to normal format
            klist = []
            for word in task['label']:
                if ' or ' in word:
                    word = word.split(' or ')
                    klist.append(get_lemma(word[0]) + ' or ' + get_lemma(word[1]))
                else:
                    klist.append(get_lemma(word))

            correct = 1
            if 'anti_label' in task.keys():
                # more complicate case
                for word in klist:
                    word = word.lower()
                    if ' or ' in word:
                        word = word.split(' or ')
                        if word[0] not in word_list and word[1] not in word_list:
                            correct = 0
                            break
                    else:
                        if word not in word_list:
                            correct = 0
                            break
                if get_lemma(task['anti_label']) in word_list:
                    correct = 0
                result_dict = {
                    'instruction': task['instruction'], 'text': task['text'], 'label': task['label'], 
                    'anti_label': task['anti_label'], 'success': str(int(correct))
                }
            else:
                for word in klist:
                    if word not in word_list:
                        correct = 0
                        break
                result_dict = {
                    'instruction': task['instruction'], 'text': task['text'], 'label': task['label'], 
                    'success': str(int(correct))
                }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')

    elif args.task == 'detoxic':
        for i in tqdm(range(len(results))):
            task = results[i]
            toxicity = detect_toxic(task['text'])
            result_dict = {
                    'instruction': task['instruction'], 'text': task['text'], 'context_id': task['context_id'], 
                    'toxicity': toxicity
                }
            with open(eval_path, 'a+') as f:
                f.write(json.dumps(result_dict))
                f.write('\n')


def evaluate_selfbleu(args):
    results_df = pd.DataFrame(
        columns=['sentiment', 'topic', 'multi', 'length', 'keyword', 'detoxic', 'average'], 
        index=LLMs
    )

    for model in LLMs:
        tot_selfbleu = 0
        for task in ['sentiment', 'topic', 'multi', 'length', 'keyword', 'detoxic']:
            data_path = f'./{args.save_fold}/{model}/{task}.jsonl'
            results = []
            with open(data_path, 'r') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    results.append(data['text'])

            bleu_score = np.around(calc_self_bleu(results), decimals=4)
            tot_selfbleu += bleu_score
            results_df.loc[model, task] = bleu_score
        results_df.loc[model, 'average'] = np.around(tot_selfbleu/6, decimals=4)

    eval_path = f'./{args.save_fold}/' + 'self_bleu.csv'
    results_df.to_csv(eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model", default='ChatGPT', type=str)
    parser.add_argument("--task", default='sentiment', type=str)
    parser.add_argument("--mode", default='evaluate', type=str, choices=['evaluate', 'evaluate_few_shot'])
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

    if args.mode == 'cal_selfbleu':
        evaluate_selfbleu(args)
    else:
        evaluate(args)
