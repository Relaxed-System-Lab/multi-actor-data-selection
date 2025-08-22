import json
import re

class JsonlProcessor:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file

    def extract_info(self, line):
        data = json.loads(line)
        custom_id = data.get('custom_id', '')
        response = data.get('response', {})
        choices = response.get('body', {}).get('choices', [])

        cluster = None
        rank = None
        summary = None
        type_ = None
        theme = None

        match = re.search(r'task-cluster-(\d+)-rank-(\d+)', custom_id)
        if match:
            cluster = match.group(1)
            rank = match.group(2)

        if choices:
            content = choices[0].get('message', {}).get('content', '')
            if content:
                content_json = json.loads(content)
                # summary = content_json.get('summary')
                # type_ = content_json.get('type')
                # theme = content_json.get('label')
                topic = content_json.get('topic')
                selected_topic = content_json.get('selected topic')
                explanation = content_json.get('explanation')


        return {
            'cluster': cluster,
            'rank': rank,
            'topic': topic,
            'selected_topic': selected_topic,
            # 'type': type_,
            'explanation': explanation
        }

    def process_jsonl(self):
        with open(self.input_file, 'r', encoding='utf-8') as infile, open(self.output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                info = self.extract_info(line)
                json.dump(info, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == '__main__':
    start_stage = 200
    end_stage = 299
    version = 1
    for i in range(start_stage, end_stage, 100):
        start = i
        end = i + 99
        input_file_path =f'/path/to/folder/v{version}_batch_{start}_{end}.jsonl'
        output_file_path = f'/path/to/folder/v{version}_batch_{start}_{end}.jsonl'
        processor = JsonlProcessor(input_file_path, output_file_path)
        processor.process_jsonl()