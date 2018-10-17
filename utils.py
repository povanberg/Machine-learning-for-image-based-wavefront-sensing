import logging
import json
import os
import matplotlib.pyplot as plt

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Params():
    """
    Class that loads hyperparameters from a json file.

        params = Params(json_path)
        print(params.learning_rate)
        params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):

        if not os.path.exists(json_path):
            with open(json_path, 'w') as f:
                data = {}
                json.dump(data, f, indent=4)

        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def hasKey(self, json_path, key_name):
        bool_key = False
        with open(json_path) as f:
            params = json.load(f)
            if key_name in params:
                bool_key = True

        return bool_key

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def plot_learningcurve(metrics, save=True, name='lrcurve.pdf'):

    plt.figure()
    plt.plot(metrics['train_loss'], label='Training loss', color='blue')
    plt.plot(metrics['val_loss'], label='Validation loss', color='red')
    plt.legend()
    plt.grid()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, metrics['n_epoch'])
    if save:
        plt.savefig(name)
    plt.show()


def get_metrics(model_dir=''):

    metrics_path = os.path.join(model_dir, 'metrics.json')

    with open(metrics_path) as f:
        metrics = json.load(f)
        return metrics

    return None