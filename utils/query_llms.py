import os
import json
import random
import time
import pandas as pd
import openai
import requests
from transformers import AutoTokenizer, AutoModel

import huggingface_hub
from http import HTTPStatus
from dashscope import Generation


# openai.api_key = 'openai api key'
# qwen_api_key = 'qwen api key'
openai.api_key = "sk-rdQk8mQXtI6r9DbX6f34Ff67D6Eb4c9a8864BeF4D16fB33a"
openai.api_base = 'https://api.ngapi.top/v1'

def query_chatgpt(model,
                prompt,
                system_content='You are an intelligent writing assistant.',
                max_tokens=2048,
                temperature=1,
                top_p=0.9,
                n_completions=1,
                presence_penalty=0,
                frequency_penalty=0
    ):
    max_tries = 30
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n_completions,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            response = response["choices"][0]["message"]["content"]
            success = True
        except Exception as e:
            print(f"Error encountered: {e}")
            print(f'key:{openai.api_key}')
            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(20)
    return response


def query_qwen(args, prompt, system_content):
    messages = [{'role': 'system', 'content': system_content},
                {'role': 'user', 'content': prompt}]
    gen = Generation()

    max_tries = 15
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            response = gen.call(
                'qwen-72b-chat',
                messages=messages,
                seed=args.seed,
                max_tokens=args.max_len,
                temperature=args.temperature,
                top_p=args.top_p,
                result_format='message'
            )
            if response.status_code == HTTPStatus.OK:
                response = response['output']['choices'][0]['message']['content'].strip()
                success = True
            else:
                print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
                success = False
        except Exception as e:
            print(f"Error encountered: {e}")
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


def get_llama2_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    ### Get LLaMA2-chat prompt
    texts = [f'[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    for user_input, response in chat_history:
        texts.append(f'{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ')
    texts.append(f'{message.strip()} [/INST]')
    return ''.join(texts)
