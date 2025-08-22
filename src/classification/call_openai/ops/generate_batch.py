import json
from openai import OpenAI
import pandas as pd
from IPython.display import Image, display

class OpenAISummarizer:
    def __init__(self, api_key, dataset_path, output_path, system_prompt, model="gpt-4o-mini", temperature=0.1):
        self.client = OpenAI(api_key=api_key)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.model = model
        self.temperature = temperature
        self.system_prompt_linewise = system_prompt

    def get_categories(self, description):
        response = self.client.chat_completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={
                "type": "json_object"
            },
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt_linewise
                },
                {
                    "role": "user",
                    "content": description
                }
            ],
        )

        return response.choices[0].message.content

    def process_file(self):
        tasks = []
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                row = json.loads(line)
                description = row['content']
                title = row['cluster']
                rank = row['rank']
                task = {
                    "custom_id": f"task-cluster-{title}-rank-{rank}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "temperature": self.temperature,
                        "response_format": {
                            "type": "json_object"
                        },
                        "messages": [
                            {
                                "role": "system",
                                "content": self.system_prompt_linewise
                            },
                            {
                                "role": "user",
                                "content": description
                            }
                        ],
                    }
                }
                tasks.append(task)

        with open(self.output_path, 'w', encoding='utf-8') as file:
            for obj in tasks:
                file.write(json.dumps(obj, ensure_ascii=False) + '\n')

