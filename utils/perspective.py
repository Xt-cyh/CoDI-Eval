from googleapiclient import discovery
import pandas as pd
from tqdm import tqdm
import json
import time
import os


def detect_toxic(text):
    # Replace with your own Perspective API Key
    PERSPECTIVE_API_KEY = 'your Perspective API Key'

    max_tries = 10
    num_tries = 0
    success = False
    response = None
    while num_tries < max_tries and not success:
        try:
            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=PERSPECTIVE_API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            analyze_request = {
                'comment': {'text': text},
                'requestedAttributes': {
                    'TOXICITY': {}
                }
            }
            response = client.comments().analyze(body=analyze_request).execute()
            success = True
        except Exception as e:
            print(f"Error encountered: {e}")
            if 'Attribute TOXICITY does not support request languages' in str(e) or 'COMMENT_EMPTY' in str(e):
                # If there is a situation where the language cannot be judged, 
                # it will be skipped directly and returned -1, without participating in the final judgment
                return -1

            num_tries += 1
            if num_tries == max_tries:
                print("Maximum number of tries reached. Aborting.")
                raise
            print(f"Retrying (attempt {num_tries}/{max_tries})...")
            time.sleep(10)

    return response['attributeScores']['TOXICITY']['summaryScore']['value']
