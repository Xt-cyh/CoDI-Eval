# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
import openai
import json
import random
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModel


openai.api_key='openai api key'


def query_chatgpt(
                prompt,
                system_content='You are an intelligent writing assistant.',
                max_tokens=2048,
                temperature=1,
                top_p=0.9,
                n_completions=1
    ):
    '''
    '''
    max_tries = 15
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n_completions,
            )
            success = True
        except Exception as e:
            print(f"Error encountered: {e}")
            print(f'key:{openai.api_key}')
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)
    return response


def query_chatglm(model, 
                tokenizer, 
                prompt,
                max_tokens=300,
                temperature=1,
                top_p=0.9,
                n_completions=1,):

    response, history = model.chat(
        tokenizer, prompt, history=[], top_p=top_p, temperature=1, max_new_tokens=max_tokens
    )
    return response


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    ### Get LLaMA2-chat prompt
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)