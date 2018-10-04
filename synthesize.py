import os
import json

def aggregate_metrics(parent_dir, metrics):
    """
    Aggregate the metrics of all experiments in folder `parent_dir`.
    Assumes that `parent_dir` contains multiple experiments, with their results stored in
    `parent_dir/subdir/metrics.json`

    Args:
        parent_dir: (string) path to directory containing experiments results
        metrics: (dict) subdir -> {'accuracy': ..., ...}
    """
    # Get the metrics for the folder if it has results from an experiment
    metrics_file = os.path.join(parent_dir, 'metrics.json')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics[parent_dir] = json.load(f)

    # Check every subdirectory of parent_dir
    for subdir in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, subdir)):
            continue
        else:
            aggregate_metrics(os.path.join(parent_dir, subdir), metrics)


def get_metrics(parent_dir):

    metrics = dict()
    aggregate_metrics(parent_dir, metrics)

    results = {'best': 'None', 'accuracy': -1, 'experiments': []}
    for key, m in metrics.items():
        if m['accuracy'] < results['accuracy'] or results['accuracy'] < 0:
            results['best'] = key
            results['accuracy'] = m['accuracy']
        results['experiments'].append({'name': key, 'accuracy': m['accuracy']})

    results_file = os.path.join(parent_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)