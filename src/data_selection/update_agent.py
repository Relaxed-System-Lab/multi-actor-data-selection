import os
import json
import numpy as np
import logging

from sklearn.preprocessing import scale
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeightUpdater:
    def __init__(self, topic_weights, domain_weights, quality_weights, collaborative_weights, eta=0.1, base_path=""):
        self.topic_weights = topic_weights
        self.domain_weights = domain_weights
        self.quality_weights = quality_weights
        self.collaborative_weights = collaborative_weights
        self.eta = eta
        self.base_path = base_path
        logging.info("Initialized WeightUpdater with initial weights and base path.")

    def load_influences(self, file_path):
        logging.info(f"Loading influences from {file_path}")
        influences = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'influence' in data and isinstance(data['influence'], list):
                    for value in data['influence']:
                        influences.append(value)
                else:
                    logging.warning(f"Missing or malformed 'influence' field in line: {line.strip()}")
        logging.info(f"Loaded {len(influences)} influence values from {file_path}")
        return influences

    def normalize_weights(self, weights_dict):
        total = sum(weights_dict.values())
        for key in weights_dict:
            weights_dict[key] /= total
        logging.debug(f"Normalized weights: {weights_dict}")
        return weights_dict

    def update_weight_single_step(self, weight, loss, mean_value, std_value):
        losses = (loss - mean_value) / std_value
        scale_score = 0.5
        scaled_loss = np.exp(loss*scale_score)
        updated_weight = (1 - self.eta) * weight + self.eta * scaled_loss
        logging.info(f"Updated weight from {weight} to {updated_weight} using scaled loss {scaled_loss}")
        return updated_weight

    def update_weights_per_loss(self, weights_dict, losses_dict):
        for key in weights_dict.keys():
            mean_loss, std_loss = self.calculate_mean_and_std(losses_dict[key])
            for loss in losses_dict[key]:
                weights_dict[key] = self.update_weight_single_step(weights_dict[key], loss, mean_loss, std_loss)
        weights_dict = self.normalize_weights(weights_dict)
        logging.info(f"Updated weights for {list(weights_dict.keys())}")
        return weights_dict

    def compute_normalized_S(self, weights_dict, losses_dict):
        S_values = {}
        for key in weights_dict.keys():
            for loss in losses_dict[key]:
                scaling_factor = 0.6 
                scaled_loss = np.exp(scaling_factor * loss)/100 - 1 
                weighted_loss_sum = sum([weights_dict[key] * scaled_loss ])
            S_values[key] = weighted_loss_sum / len(losses_dict[key])
        logging.info(f"Computed normalized S values: {S_values}")
        return S_values


    def update_collaborative_weights(self, S_values, beta=1):
        average_S = np.mean(list(S_values.values()))
        for i in range(len(self.collaborative_weights)):
            diff = list(S_values.values())[i] - average_S
            accelerated_diff = np.sign(diff) * (abs(diff) ** beta)  # 加速调整
            self.collaborative_weights[i] += accelerated_diff
        
        total_weight = sum(self.collaborative_weights)
        if total_weight != 0:
            self.collaborative_weights = [weight / total_weight for weight in self.collaborative_weights]
        
        logging.info(f"Updated and normalized collaborative weights: {self.collaborative_weights}")
        return self.collaborative_weights


    def update_weights(self):
        topics = list(range(13))
        topic_losses = {}
        for topic in topics:
            file_path = os.path.join(self.base_path, f"topic/topic_{topic}.jsonl")
            losses = self.load_influences(file_path)
            topic_losses[f"topic_{topic}"] = losses
       
        self.topic_weights = self.update_weights_per_loss(self.topic_weights, topic_losses)

        domains = ['arxiv', 'book', 'c4', 'commoncrawl', 'github', 'stackexchange', 'wikipedia']
        domain_losses = {}
        for domain in domains:
            file_path = os.path.join(self.base_path, f"domain/{domain}.jsonl")
            losses = self.load_influences(file_path)
            domain_losses[domain] = losses

        self.domain_weights = self.update_weights_per_loss(self.domain_weights, domain_losses)

        quality_intervals = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        quality_losses = {}
        for low, high in quality_intervals:
            file_path = os.path.join(self.base_path, f"edu/edu_bin_{low}_{high}.jsonl")
            losses = self.load_influences(file_path)
            quality_losses[f"quality_{low}_{high}"] = losses

        self.quality_weights = self.update_weights_per_loss(self.quality_weights, quality_losses)

        topic_S = self.compute_normalized_S(self.topic_weights, topic_losses)
        domain_S = self.compute_normalized_S(self.domain_weights, domain_losses)
        quality_S = self.compute_normalized_S(self.quality_weights, quality_losses)
        S_values = {**quality_S, **domain_S, **topic_S}
        self.collaborative_weights = self.update_collaborative_weights(S_values)

        return {
            'topic_weights': self.topic_weights,
            'domain_weights': self.domain_weights,
            'quality_weights': self.quality_weights,
            'collaborative_weights': self.collaborative_weights
        }

    def save_weights(self, iteration):
        with open(f'agent_weights_iteration_{iteration}.json', 'w') as f:
            json.dump({
                'topic_weights': self.topic_weights,
                'domain_weights': self.domain_weights,
                'quality_weights': self.quality_weights
            }, f)
        with open(f'collaborative_weights_iteration_{iteration}.json', 'w') as f:
            json.dump(self.collaborative_weights, f)
        logging.info(f"Saved weights for iteration {iteration}")

    @staticmethod
    def initialize_weights():
        initial_topic_weights = {f"topic_{i}": 10 for i in range(13)}
        initial_domain_weights = {domain: 10 for domain in ['arxiv', 'book', 'c4', 'commoncrawl', 'github', 'stackexchange', 'wikipedia']}
        initial_quality_weights = {f"quality_{low}_{high}": 10 for low, high in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]}
        initial_collaborative_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3] 
        logging.info("Initialized default weights.")
        return initial_topic_weights, initial_domain_weights, initial_quality_weights, initial_collaborative_weights

    @staticmethod
    def load_previous_weights(agent_weight_path, collaborative_weight_path):
        logging.info(f"Loading previous weights from {agent_weight_path} and {collaborative_weight_path}")
        try:
            with open(agent_weight_path, 'r') as f:
                agent_weights = json.load(f)
            
            with open(collaborative_weight_path, 'r') as f:
                collaborative_weights = json.load(f)

            topic_weights = agent_weights['topic_weights']
            domain_weights = agent_weights['domain_weights']
            quality_weights = agent_weights['quality_weights']
            
            logging.info("Loaded previous weights successfully.")
            return topic_weights, domain_weights, quality_weights, collaborative_weights
        except FileNotFoundError as e:
            logging.error(f"Error loading weights: {e}")
            raise

    @staticmethod
    def calculate_mean_and_std(values):
        mean_value = np.mean(values)
        std_value = np.std(values)
        logging.info(f"Calculated mean: {mean_value}, std: {std_value}")
        return mean_value, std_value

if __name__ == "__main__":
    logging.info("Program started.")
    
    base_path = '/path/to/folder'
    
    topic_weights, domain_weights, quality_weights, collaborative_weights = WeightUpdater.initialize_weights()
    iteration = 1
    weight_updater = WeightUpdater(topic_weights, domain_weights, quality_weights, collaborative_weights, base_path=base_path)

    updated_weights = weight_updater.update_weights()

    weight_updater.save_weights(iteration=iteration)
    logging.info(f"weight updated for {iteration}")
