"""
'utils.py' file implements the utility functions used in the assignment.
"""
import copy
import inspect
import random

import torch
import torch.nn as nn
import logging as log
import tqdm
import sys
import os

from torch.utils.data import DataLoader
from typing import NamedTuple, List, Any, Callable
from pathlib import Path

# Constants
SEED = 1
GLOBAL_RANDOM_GENERATOR = torch.Generator()



class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class TorchTrainer():
    """
    A class for training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, optimizer, scheduler=None, device=''):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param optimizer: The optimizer to train with.
        :param scheduler: The learning rate scheduler.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_loss = float('inf')
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_val: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_val: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        best_loss = -1
        best_acc = -1
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = \
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/val_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            train_epoch_results = self.train_epoch(dl_train=dl_train)
            train_loss.extend(train_epoch_results.losses)
            train_acc.append(train_epoch_results.accuracy)

            val_epoch_results = self.test_epoch(dl_test=dl_val)
            val_loss.extend(val_epoch_results.losses)
            val_acc.append(val_epoch_results.accuracy)
            avg_loss = sum(val_epoch_results.losses) / len(val_epoch_results.losses)

            if self.scheduler:
                # self.scheduler.step(avg_loss)
                self.scheduler.step()
                self._print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}', verbose)

            # Early Stopping
            if avg_loss < best_loss or best_loss == -1:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping:
                print("Early Stopping\n")
                break

            if val_epoch_results.accuracy > best_acc:
                best_acc = val_epoch_results.accuracy

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_epoch_results, val_epoch_results, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, val_loss, val_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test/validation set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        X, _, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model.forward(X)
        loss = self.model.loss(outputs, y)  # output = loss(input, target)
        loss.backward()
        self.optimizer.step()
        num_correct = (outputs.argmax(dim=1) == y).sum().item()
        loss = loss.item()

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, _, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            loss = self.model.loss(outputs, y)
            num_correct = (outputs.argmax(dim=1) == y).sum().item()
            loss = loss.item()
        return BatchResult(loss, num_correct)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False

def get_device():
    """
    @brief: Get the device (GPU or CPU) for computation.
    @return: Device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    return device

def set_seed(seed):
    """
    @brief: Set the seed for reproducibility.
    @param seed: Seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    GLOBAL_RANDOM_GENERATOR.manual_seed(seed)
    # torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, filename):
    """
    @brief: Save the model to a file.
    @param model: The model to save.
    @param filename: The file to save the model to.
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(model.state_dict(), filename)


def save_predictions(predictions, predict_output_dir):
    """
    @brief: Save the predictions to a file.
    Args:
        predictions: A tensor containing the predictions.
        predict_output_dir: The directory to save the predictions to.
    """
    if not os.path.exists(predict_output_dir):
        os.makedirs(predict_output_dir)
    with open(os.path.join(predict_output_dir, 'predictions.txt'), 'w') as f:
        for pred in predictions:
            f.write(f'{pred}\n')