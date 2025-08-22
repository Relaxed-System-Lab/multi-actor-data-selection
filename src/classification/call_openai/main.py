import logging
import argparse
from ops.sample_10_each import *
from ops.generate_batch import *
from ops.batch_process import *
from ops.process_output import *

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('process.log'), logging.StreamHandler()])
    
    # Command-line argument parsing
    # parser = argparse.ArgumentParser(description="Process cluster data and generate summaries.")
    # parser.add_argument('--start', type=int, required=True, help="Start stage")
    # parser.add_argument('--end', type=int, required=True, help="End stage")
    # args = parser.parse_args()
    
    # Parameters
    start_stage = 0
    end_stage = 9999
    version = 1

    api_key = ""
    system_prompt2 = '''
    You are an annotator tasked with analyzing the category distribution of web data. 
    For each provided full-text segment, select one topic from the following 13 provided topics that can best represents the primary topic of the text:
    General reference, Culture and the arts, Geography and places, Health and fitness, History and events, Human activities, Mathematics and logic, Natural and physical sciences, People and self, Philosophy and thinking, Religion and belief systems, Society and social sciences, Technology and applied sciences
    Please respond with a JSON object in the following format:
    {
    "topic": string, //summarize the topic of the provided document
    "selected topic": string //selected topic from the 13 topics:General reference, Culture and the arts, Geography and places, Health and fitness, History and events, Human activities, Mathematics and logic, Natural and physical sciences, People and self, Philosophy and thinking, Religion and belief systems, Society and social sciences, Technology and applied sciences
    "explanation": string //explain why select the topic in this way
    }
    '''

    for i in range(start_stage, end_stage, 100):
        start = i
        end = i + 99


        # Define base folder path and output JSONL file path
        base_folder_path = '/mnt/hwfile/opendatalab/baitianyi/cc_minhash0.7_cluster/en/'
        output_jsonl_path = f'/mnt/petrelfs/baitianyi/dup/openai_v2/cluster/output_{start}_{end}.jsonl'

        # Log the start of the process
        # logging.info('Starting cluster processing')
        # logging.info(f'Base folder path: {base_folder_path}')
        # logging.info(f'Output JSONL path: {output_jsonl_path}')
        
        # # Create an instance of the ClusterProcessor class
        # try:
        #     processor = ClusterProcessor(base_folder_path, start, end, output_jsonl_path)
        #     # Process clusters and save to JSONL
        #     processor.process_clusters_parallel()
        #     logging.info('Cluster processing completed successfully')
        # except Exception as e:
        #     logging.error(f'Error during cluster processing: {e}')
            
        # 转成batch process格式
        output_path = f"/mnt/petrelfs/baitianyi/dup/openai_v2/generate_batch/v{version}_batch_{start}_{end}.jsonl"

        logging.info('Starting OpenAISummarizer processing')
        logging.info(f'API key: {api_key}')
        logging.info(f'Summarizer output path: {output_path}')
        
        try:
            summarizer = OpenAISummarizer(api_key, output_jsonl_path, output_path, system_prompt2)
            summarizer.process_file()
            logging.info('Summarizer processing completed successfully')
        except Exception as e:
            logging.error(f'Error during summarizer processing: {e}')

        # batch process
        output_file_path = f"/mnt/petrelfs/baitianyi/dup/openai_v2/call_api/v{version}_batch_{start}_{end}.jsonl"
        
        logging.info('Starting OpenAIBatchProcessor processing')
        logging.info(f'Batch processor output file path: {output_file_path}')
        
        try:
            batch_processor = OpenAIBatchProcessor(api_key, input_file_path=output_path, output_file_path=output_file_path)
            batch_processor.process_batch()
            logging.info('Batch processing completed successfully')
        except Exception as e:
            logging.error(f'Error during batch processing: {e}')

        final_output = f'/mnt/petrelfs/baitianyi/dup/openai_v2/final_output/new_batch_{start}_{end}.jsonl'

        logging.info('Starting JsonlProcessor processing')
        logging.info(f'Jsonl processor final output file path: {final_output}')

        try:
            processor = JsonlProcessor(output_file_path, final_output)
            processor.process_jsonl()
            logging.info('JSONL processing completed successfully')
        except Exception as e:
            logging.error(f'Error during JSONL processing: {e}')

