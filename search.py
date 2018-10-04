import utils
import os
import sys
import argparse
import logging
from subprocess import Popen, check_call
from synthesize import get_metrics


# This python script performs hyperparameters search
# - Search range is specified in params.json

def launch_training_job(parent_dir, data_dir, params, processes):
    """
        Launch training of the model with a set of hyperparameters in parent_dir
    """

    # Create a new folder in parent_dir with unique_name "job_name"
    job_name = "lr" + str(params.learning_rate) + "_b" + str(params.batch_size) + "_e" + str(params.num_epochs)


    version_dir = os.path.join(parent_dir, params.model_version)
    if not os.path.exists(version_dir):
        os.makedirs(version_dir)

    model_dir = os.path.join(version_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config (subprocesses)
    logging.info("Training launch: {}".format(job_name))
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                  data_dir=data_dir)
    #proc = Popen(cmd, shell=True)
    proc = check_call(cmd, shell=True)
    processes.append(proc)

# Args parsing
PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', default='models/', help='Directory containing models')
parser.add_argument('--config_dir', default='config/', help='Directory containing params.json')
parser.add_argument('--data_dir', default='psfs/', help="Directory containing the dataset")
parser.add_argument('--logs_dir', default='logs/', help="Directory containing the logs")

# Logger
args = parser.parse_args()
log_path = os.path.join(args.logs_dir, 'logs.log')
utils.set_logger(log_path)

# Load config parameters
json_path = os.path.join(args.config_dir, 'params.json')
assert os.path.isfile(json_path), logging.error("No json configuration file found at {}".format(json_path))
params = utils.Params(json_path)
batch_sizes = params.batch_size
learning_rates = params.learning_rate
num_epochs = params.num_epochs

# Hyperparameter search

processes = []

for batch_size in batch_sizes:
    params.batch_size = batch_size

    for lr  in learning_rates:
        params.learning_rate = lr

        for n_epochs in num_epochs:
            params.num_epochs = n_epochs

            launch_training_job(args.models_dir, args.data_dir, params, processes)


exit_codes = 0 #[p.wait() for p in processes]

model_path = os.path.join(args.models_dir, params.model_version)
get_metrics(model_path)

logging.info('Processes ended with exit code: {}'.format(exit_codes))
logging.info('All trainings succesfully finished.')