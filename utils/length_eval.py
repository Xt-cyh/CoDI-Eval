import os
import re
import openai
import json
import random
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
from rouge import Rouge
import matplotlib.pyplot as plt
from utils.query_chatgpt import query_chatgpt

trans_1 = {
    0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven',
    8:'eight', 9:'nine', 10:'ten', 11:'eleven', 12:'twelve', 13:'thirteen', 14:'fourteen', 
    15:'fifteen', 16:'sixteen', 17:'seventeen', 18:'eighteen', 19:'nineteen'
}

trans_10 = {2:'twenty', 3:'thirty', 4:'forty', 5:'fifty', 6:'sixty', 7:'seventy', 8:'eighty', 9:'ninety'}


def number2word(num):
    # max number is 999
    if num < 20:
        return trans_1[num]
    elif num < 100 and num % 10 == 0:
        return trans_10[num // 10]
    elif num < 100:
        ones = num % 10
        tens = num // 10
        return f'{trans_10[tens]}-{trans_1[ones]}'
    elif num % 100 == 0:
        return f'{trans_1[num // 100]} hundreds'
    elif num < 1000:
        hundreds = num // 100
        num -= hundreds * 100
        num_left = number2word(num)
        return f'{trans_1[hundreds]} hundreds and {num_left}'


def get_mode(res):
    prompt = "Format the length control instructions below in order to identify the content of length control. " \
    "The mode should be the words in ['at most','at least','equal to','around','between'].\n" \
    "There are several modes of length control as follows:\n" \
    "1.instruction means no more than, output 'at most'\n" \
    "2.instruction means no less than, output 'at least'\n" \
    "5.instruction means equal to, output 'equal to'\n" \
    "6.instruction means around, output 'around'\n" \
    "7.instruction means between, output 'between'\n" \
    "Instruction:Please write a response in {{number}} words.\nMode:[equal to]\n" \
    "Instruction:Just help me to generate texts, at least {{number}} words:\nMode:[at least]\n" \
    "Instruction:Write something between {{number1}} to {{number2}} words\nMode:[between]\n" \
    "Instruction:No more than {{number}} words, write a sentence:\nMode:[at most]\n" \
    "Instruction:Can you create some texts, about {{number}} words.\nMode:[around]\n" \
    "Instruction:{instruction}\nMode:" \

    prompt = prompt.format(instruction=res)
    response = query_chatgpt(prompt, temperature=1)
    res = response["choices"][0]["message"]["content"]
    pattern = r"\[(.+?)\]"
    match = re.search(pattern, res)
    if match:
        res = match.group(1)
    else:
        res = None
    return res


def get_interval(mode, n):
    # Set 10000 to positive infinity
    interval = None
    if mode == 'at most':
        interval = [1, n]
    elif mode == 'at least':
        interval = [n, 10000]
    elif mode == 'equal to':
        interval = [n, n]
    elif mode == 'around':
        interval = [np.around(n*0.9), np.around(n*1.1)]
    return interval

