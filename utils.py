import numpy as np
import torch
import os
import logging as log

# TODO: dont put here stuff for training, just utillities. 

def save_predictions(predictions, predict_output_dir):
    """
    Save the predictions to a file.
    Args:
        predictions: A tensor containing the predictions.
        predict_output_dir: The directory to save the predictions to.
    """
    if not os.path.exists(predict_output_dir):
        os.makedirs(predict_output_dir)
    with open(os.path.join(predict_output_dir, 'predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')

def get_device():
    """
    Get the device (GPU or CPU) for computation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    return device

def get_person_correlation(predictions, intensities):
    """Calculates the Pearson correlation between the predictions and intensities."""
    return np.corrcoef(predictions, intensities)[0, 1]


def print_args(args):
    """
    Prints all the arguments as a dictionary.

    Args:
        args: Namespace object containing configuration and hyperparameters.
    """
    args_dict = vars(args)
    print("Arguments used for this run:")
    print(args_dict)


def transform_into_valid_intensity_range(rna_score, rbp_known_intensity_range=(0.0, 8.401)):
    """
    Scales predicted RNA scores into the known range of RBP intensities.

    :param rna_score: A NumPy array of floats representing the RNA scores to be transformed.
    :param rbp_known_intensity_range: A tuple (min_rbp, max_rbp) representing the known range of RBP intensities.
    :return: Scaled RNA score(s) within the known RBP intensity range as a NumPy array.
    """
    min_rbp, max_rbp = rbp_known_intensity_range
    
    # Calculate the min and max of the RNA scores
    min_score = np.min(rna_score)
    max_score = np.max(rna_score)
    
    if min_score == max_score:
        # If all values are the same, map everything to the midpoint of the range
        return np.full_like(rna_score, (min_rbp + max_rbp) / 2.0)
    
    # Apply min-max scaling
    scaled_scores = ((rna_score - min_score) / (max_score - min_score)) * (max_rbp - min_rbp) + min_rbp
    
    return scaled_scores


def find_min_max_values_in_rbp_intensity_files(directory='RNAcompete_intensities'):
    min_value = float('inf')
    max_value = float('-inf')
    
    for i in range(1, 39):
        file_path = os.path.join(directory, f"RBP{i}.txt")
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        value = float(line.strip())
                        min_value = min(min_value, value)
                        max_value = max(max_value, value)
                    except ValueError:
                        continue
        else:
            print(f"File not found: {file_path}")
    
    return min_value, max_value


if __name__ == '__main__':
    print(find_min_max_values_in_rbp_intensity_files())