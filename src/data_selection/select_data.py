import os
import json
from concurrent.futures import ThreadPoolExecutor

def scan_meta_files(base_folder):
    jsonl_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file):
            yield line_num, line.strip()

def process_meta_file(meta_file, subfolder):
    selection_lines = {}
    selection_lines[meta_file] = []
    for line_num, line in read_jsonl_file(meta_file):
        data = json.loads(line)
        if data.get('selection', 0) == 1:
            selection_lines[meta_file].append(line_num)
    return subfolder, selection_lines

def extract_selected_lines(subfolder, files, source_folder, output_folder):
    for file, lines in files.items():
        source_file_path = os.path.join(source_folder, subfolder, os.path.basename(file))
        output_file_path = os.path.join(output_folder, subfolder + '_' + os.path.basename(file))
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(source_file_path, 'r', encoding='utf-8') as source_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line_num, line in enumerate(source_file):
                if line_num in lines:
                    output_file.write(line)

def main():
    meta_folder = ''
    source_folder = ''
    output_folder = ''
    meta_files = scan_meta_files(meta_folder)

    selection_lines = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for meta_file in meta_files:
            subfolder = os.path.basename(os.path.dirname(meta_file))
            futures.append(executor.submit(process_meta_file, meta_file, subfolder))

        for future in futures:
            subfolder, result = future.result()
            if subfolder not in selection_lines:
                selection_lines[subfolder] = {}
            selection_lines[subfolder].update(result)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for subfolder, files in selection_lines.items():
            futures.append(executor.submit(extract_selected_lines, subfolder, files, source_folder, output_folder))

        for future in futures:
            future.result()

    print("processed!")

if __name__ == "__main__":
    main()
