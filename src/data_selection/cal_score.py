import os
import json
import numpy as np
import logging
from update_agent import WeightUpdater

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def decay_function(count, lambda_param=0.1):
    return 1 / (1 + lambda_param * count)

def compute_final_score(alpha, beta, gamma, quality_value, domain_value, topic_value, count):
    score = (alpha * quality_value + beta * domain_value + gamma * topic_value) * decay_function(count)
    logging.debug(f'Computed score: {score} with alpha={alpha}, beta={beta}, gamma={gamma}, '
                  f'quality_value={quality_value}, domain_value={domain_value}, topic_value={topic_value}, count={count}')
    return score

def update_scores(file_path, quality_weights, domain_weights, topic_weights, collaborative_weights, iteration):
    logging.info(f'Updating scores for file: {file_path}')
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logging.error(f'Error reading file {file_path}: {e}')
        return
    
    new_lines = []
    for line in lines:
        try:
            data = json.loads(line.strip())
            quality_value = next((v for k, v in quality_weights.items() if float(data["attr-fineweb-edu"]) >= float(k.split('_')[1]) and float(data["attr-fineweb-edu"]) < float(k.split('_')[2])), 0)
            domain_value = domain_weights[data["domain"]]
            topic_value = topic_weights[f"topic_{data['attr-cc_en_topic']}"]
            score = compute_final_score(collaborative_weights[0], collaborative_weights[1], collaborative_weights[2],
                                        quality_value, domain_value, topic_value, data["epoch"])
            
            score_key = f"score_{iteration}"
            data[score_key] = score
            
            new_lines.append(json.dumps(data))
        except KeyError as e:
            logging.warning(f'Missing key in data: {e} - {line}')
        except Exception as e:
            logging.error(f'Error processing line: {e} - {line}')

    try:
        with open(file_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
        logging.info(f'Successfully updated file: {file_path}')
    except Exception as e:
        logging.error(f'Error writing file {file_path}: {e}')

def main(agent_weight_path, collaborative_weight_path, data_folder_path,iteration):
    logging.info('Starting the main process...')

    try:
        with open(agent_weight_path, 'r') as f:
            agent_weights = json.load(f)
        logging.info(f'Loaded agent weights from {agent_weight_path}')
    except Exception as e:
        logging.error(f'Error loading agent weights: {e}')
        return

    try:
        with open(collaborative_weight_path, 'r') as f:
            collaborative_weights = json.load(f)
        logging.info(f'Loaded collaborative weights from {collaborative_weight_path}')
    except Exception as e:
        logging.error(f'Error loading collaborative weights: {e}')
        return

    quality_weights = agent_weights.get('quality_weights', {})
    domain_weights = agent_weights.get('domain_weights', {})
    topic_weights = agent_weights.get('topic_weights', {})

    data_folder = data_folder_path
    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(subfolder_path, filename)
                    update_scores(file_path, quality_weights, domain_weights, topic_weights, collaborative_weights, iteration)

if __name__ == "__main__":
    agent_weight_path = '/path/to/file'
    collaborative_weight_path = '/path/to/file'
    data_folder_path = '/path/to/folder'
    iteration=1
    main(agent_weight_path, collaborative_weight_path, data_folder_path,iteration)
