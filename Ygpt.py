from sympy import nan
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import pandas as pd
import telebot

import requests
import time

class SimpleSyncGPTClient:
    def __init__(self, api_key, url, model_uri,
                 temperature=0.7, max_tokens=1000,
                 max_retries=3, retry_delay=1.0):
        self.api_key = api_key
        self.url = url
        self.model_uri = model_uri
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def request(self, task, data, few_shot_samples=None):
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._make_request(task, data, few_shot_samples)
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        return None

    def _make_request(self, task, data, few_shot_samples):
        headers = {
            'Authorization': f'Api-Key {self.api_key}',
            'Content-Type': 'application/json',
        }

        messages = [{'role': 'system', 'text': task}]
        if few_shot_samples:
            for q, a in zip(few_shot_samples.get("data", []), few_shot_samples.get("answer", [])):
                messages.extend([
                    {'role': 'user', 'text': q},
                    {'role': 'assistant', 'text': a}
                ])
        messages.append({'role': 'user', 'text': data})

        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": self.max_tokens
            },
            "messages": messages
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        result = response.json()
        return result["result"]["alternatives"][0]["message"]["text"]



client = SimpleSyncGPTClient(
    api_key="",
    url="https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
    model_uri="gpt://b1gof4el2meubsf35lsd/yandexgpt-32k"
)




def get_drug(prompt ,query):
    response = client.request(
        task=prompt,
        data=query,
        few_shot_samples=None
    )

    return response

