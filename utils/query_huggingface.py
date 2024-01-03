import os
import json
import requests
from huggingface_hub.inference_api import InferenceApi


# please input your huggingface API key here
API_TOKEN = ''


def request_query(payload):
    '''
    use request
    input: data = query({"inputs": "I like you. I love you"})
    '''
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


def query(model, inputs):
    if len(inputs) > 500:
        inputs = inputs[:500]
    inference = InferenceApi(repo_id=model, token=API_TOKEN)
    return inference(inputs=inputs)


def parse_topic_dict(response):
    map_dict = {
        'arts_&_culture':'arts_&_culture', 'business_&_entrepreneurs':'business_&_entrepreneurs', 'celebrity_&_pop_culture':'celebrity_&_pop culture', 'diaries_&_daily_life':'daily life',
        'family':'family', 'fashion_&_style':'fashion_&_style', 'film_tv_&_video':'film_&_tv_&_video', 'fitness_&_health':'fitness_&_health', 'food_&_dining':'food_&_dining', 'gaming':'gaming',
        'learning_&_educational':'learning_&_educational', 'music':'music', 'news_&_social_concern':'social concern', 'relationships':'relationships',
        'science_&_technology':'science_&_technology', 'sports':'sports', 'travel_&_adventure':'travel_&_adventure', 'youth_&_student_life':'youth_&_student life'
    }
    res_dict = {}
    for d in response[0]:
        if d['label'] != 'other_hobbies':
            res_dict[map_dict[d['label']]] = d['score']
    # sort
    sorted_dict = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict


def parse_sentiment_dict(response):
    map_dict = {
        'negative':'negative sentiment', 'neutral':'neutral sentiment', 'positive':'positive sentiment',
        'anger':'anger', 'disgust':'disgust', 'fear':'fear', 'joy':'joy', 'sadness':'sadness', 'surprise':'surprise'
    }
    res_dict = {}
    for d in response[0]:
        res_dict[map_dict[d['label']]] = d['score']
    # sort
    sorted_dict = dict(sorted(res_dict.items(), key=lambda item: item[1], reverse=True))
    return sorted_dict
