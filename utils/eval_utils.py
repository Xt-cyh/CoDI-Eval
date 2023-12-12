import re
import nltk
import spacy
import numpy as np
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu


nlp = spacy.load('en_core_web_md')


def preprocess(text):
    return nltk.word_tokenize(text.lower())


def calculate_meteor(reference, hypothesis):
    preprocessed_reference = preprocess(reference)
    preprocessed_hypothesis = preprocess(hypothesis)
    meteor_score = nltk.translate.meteor_score.meteor_score(
        [preprocessed_reference], preprocessed_hypothesis
    )
    return meteor_score


def extract_text(text):
    pattern = r'(?::\n\n|:\n)(.*)$'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return text


def get_lemma(word):
    word = word.lower()
    doc = nlp(word)
    lemmatized_word = doc[0].lemma_
    return lemmatized_word
    

def extract_words(sentence):
    words = re.findall(r'\w+(?:-\w+)*', sentence)
    words = [word.lower() for word in words]
    # convert all word to normal format
    words = [get_lemma(word) for word in words]
    return words


def get_result_dict(args, task, res):
    result_dict = None
    if args.task == 'multi':
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label1':task['label1'], 'label2':task['label2']
        }
    elif 'anti_label' in task.keys():
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label':task['label'], 'anti_label':task['anti_label']
        }
    elif args.task == 'detoxic':
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'context_id':task['context_id']
        }
    else:
        result_dict = {
            'instruction':task['instruction'], 'text':res, 'label':task['label']
        }
    return result_dict


def calc_self_bleu(test_corpus, max_ngram=4):
    bleu_scores = []
    for i in range(len(test_corpus)):
        test_corpus[i] = test_corpus[i].split()

    for i in tqdm(range(len(test_corpus))):
        reference = test_corpus[:i] + test_corpus[i+1:]
        hypothesis = test_corpus[i]

        bleu_score_i = sentence_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score_i)

    self_bleu = np.mean(bleu_scores)
    return self_bleu
