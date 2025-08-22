import os
import json
import heapq

def scan_files(base_folder):
    jsonl_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.strip()

base_folder = '/path/to/folder'
jsonl_files = scan_files(base_folder)

top_n_heap = []


for jsonl_file in jsonl_files:
    for line in read_jsonl_file(jsonl_file):
        data = json.loads(line)
        score_1 = data.get('score_1', 0)
        if len(top_n_heap) < 8500000:
            heapq.heappush(top_n_heap, (score_1, line))
        else:
            if score_1 > top_n_heap[0][0]:
                heapq.heapreplace(top_n_heap, (score_1, line))


threshold = top_n_heap[0][0]

for jsonl_file in jsonl_files:
    output_file = jsonl_file.replace('.jsonl', '_processed.jsonl')
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for line in read_jsonl_file(jsonl_file):
            data = json.loads(line)
            score_1 = data.get('score_1', 0)
            if score_1 >= threshold:
                data['selection'] = 1
            else:
                data['selection'] = 0
            out_file.write(json.dumps(data) + '\n')

print("数据处理完成！")
