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
