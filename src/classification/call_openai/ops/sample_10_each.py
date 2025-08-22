import os
import json
import logging
from multiprocessing import Pool, cpu_count
import gc
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClusterProcessor:
    def __init__(self, base_folder_path, start_stage, end_stage, output_jsonl_path):
        self.base_folder_path = base_folder_path
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.output_jsonl_path = output_jsonl_path

    def get_sampled_lines_by_rank(self, folder_path, cluster_index):
        all_data = []
        
        logging.info(f'Processing folder: {folder_path}')
        
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        data = json.loads(line)
                        all_data.append(data)
        
        # Sort data by rank
        all_data = sorted(all_data, key=lambda x: x.get('rank', float('inf')))
        
        # Calculate the interval
        total_lines = len(all_data)
        interval = math.ceil(total_lines / 10)
        
        # Sample based on the rank interval
        sampled_data = [all_data[i] for i in range(0, total_lines, interval)]
        
        # Extract 'content' and 'rank', remove double quotes from 'content', and add 'cluster'
        sampled_content = [{'cluster': cluster_index, 'rank': entry.get('rank'), 'content': entry['content']} for entry in sampled_data if 'content' in entry]
        
        # Free up memory
        del all_data
        gc.collect()
        
        return sampled_content

    def process_single_cluster(self, args):
        base_folder_path, i = args
        folder_name = f'clustering-{i}'
        folder_path = os.path.join(base_folder_path, folder_name)
        
        if os.path.isdir(folder_path):
            logging.info(f'Starting processing for cluster: {i}')
            try:
                sampled_content = self.get_sampled_lines_by_rank(folder_path, i)
                logging.info(f'Completed processing for cluster: {i}')
                return sampled_content
            except Exception as e:
                logging.error(f'Error processing cluster {i}: {e}')
                return None
        
        logging.warning(f'Folder does not exist: {folder_path}')
        return None

    def process_clusters_parallel(self):
        pool = Pool(cpu_count())
        args = [(self.base_folder_path, i) for i in range(self.start_stage, self.end_stage + 1)]
        
        # Process in parallel
        logging.info('Starting parallel processing')
        rows = pool.map(self.process_single_cluster, args)
        logging.info('Parallel processing completed')
        
        # Filter out None results
        rows = [row for row in rows if row is not None]
        
        # Flatten the list of lists
        flat_rows = [item for sublist in rows for item in sublist]
        
        # Save rows to JSONL file
        logging.info(f'Saving results to JSONL file: {self.output_jsonl_path}')
        with open(self.output_jsonl_path, 'w', encoding='utf-8') as jsonlfile:
            for row in flat_rows:
                jsonlfile.write(json.dumps(row) + '\n')
        
        pool.close()
        pool.join()
        logging.info('Processing complete')

