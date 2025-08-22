import json
from openai import OpenAI
import time
import logging



class OpenAIBatchProcessor:
    def __init__(self, api_key, input_file_path, output_file_path):
        self.client = OpenAI(api_key=api_key)
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path

    def upload_file(self):
        with open(self.input_file_path, "rb") as file:
            batch_file = self.client.files.create(file=file, purpose="batch")
        print("Input file created:", batch_file)
        return batch_file

    def create_batch_job(self, batch_file_id):
        batch_job = self.client.batches.create(
            input_file_id=batch_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        print("Batch job created:", batch_job)
        return batch_job

    def wait_for_batch_completion(self, batch_job_id):
        while True:
            batch_job = self.client.batches.retrieve(batch_job_id)
            print("Batch job status:", batch_job.status)
            if batch_job.status in ["completed", "failed"]:
                break
            time.sleep(30)  
        return batch_job

    def save_batch_results(self, batch_job):
        if batch_job.status == "completed":
            result_file_id = batch_job.output_file_id
            result = self.client.files.content(result_file_id).content

            with open(self.output_file_path, 'wb') as file:
                file.write(result)
            print("Batch job results saved to:", self.output_file_path)
        else:
            print("Batch job failed or did not complete successfully:", batch_job.errors)

    def process_batch(self):
        batch_file = self.upload_file()
        batch_job = self.create_batch_job(batch_file.id)
        completed_batch_job = self.wait_for_batch_completion(batch_job.id)
        self.save_batch_results(completed_batch_job)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler('process.log'), logging.StreamHandler()])
    start_stage = 0
    end_stage = 500

    api_key = "your-api-key"
    input_path = f"/path/to/folder/batch_{start_stage}_{end_stage}.jsonl"
    output_path = f"/path/to/folder/batch_{start_stage}_{end_stage}.jsonl"
    
    logging.info('Starting OpenAIBatchProcessor processing')
    logging.info(f'Batch processor file path: {input_path}')
    
    try:
        batch_processor = OpenAIBatchProcessor(api_key=api_key, input_file_path=input_path, output_file_path=output_path)
        batch_processor.process_batch()
        logging.info('Batch processing completed successfully')
    except Exception as e:
        logging.error(f'Error during batch processing: {e}')
