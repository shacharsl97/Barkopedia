import subprocess
import time
import os
import re
import itertools
import json
from multiprocessing import Process

def get_free_gpu(threshold_mb=20000):
    """Return the index of a free GPU with memory usage below threshold_mb, or None if none found."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        for idx, line in enumerate(result.stdout.strip().split("\n")):
            used, total = map(int, line.split(','))
            if used < threshold_mb:
                return idx
    except Exception as e:
        print(f"Error checking GPUs: {e}")
    return None

# Example grid search params
param_grid = {
    'learning_rate': [2.5e-5, 1.5e-5],
    'weight_decay': [0.001, 0.01],
    'hidden_dropout_prob': [0.25, 0.35],
    'attention_probs_dropout_prob': [0.25,0.35],
    'clean_audio': [True],
    'loss': ['soft_cross'],
    'num_train_epochs': [6],
}

def param_to_str(k, v):
    if isinstance(v, float):
        return f"{k}-{v:.0e}"  # scientific notation for floats
    return f"{k}-{v}"

def run_training(params, gpu_num, result_dir):
    # Sort keys for consistent config_id
    config_id = "_".join([param_to_str(k, params[k]) for k in sorted(params.keys())])
    args = " ".join([f"--{k} {v}" for k, v in params.items()])
    metrics_file = os.path.join(result_dir, f"metrics_{config_id}.json")
    cmd = f"python train_barkopedia.py {args} --metrics_out {metrics_file} --gpu_num {gpu_num}"
    print(f"Launching: {cmd} on GPU {gpu_num}")
    os.system(cmd)

result_dir = "grid_search_results"
os.makedirs(result_dir, exist_ok=True)

processes = []
for values in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))
    if params['clean_audio'] == False:
        del params['clean_audio']  # Remove if False
    else:
        params['clean_audio'] = ''
    while True:
        gpu = get_free_gpu()
        if gpu is not None:
            p = Process(target=run_training, args=(params, gpu, result_dir))
            p.start()
            processes.append((p, params))
            time.sleep(30)
            break
        else:
            print("No free GPU found, waiting...")
            time.sleep(60)
# Wait for all jobs to finish
for p, _ in processes:
    p.join()
print("Grid search complete.")

# Parse results and summarize
results = []
for _, params in processes:
    config_id = "_".join([param_to_str(k, params[k]) for k in sorted(params.keys())])
    metrics_file = os.path.join(result_dir, f"metrics_{config_id}.json")
    if not os.path.exists(metrics_file):
        continue
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        # Try to load log_history if present
        log_history = metrics.get('log_history')
        best_epoch = None
        best_eval_acc = None
        last_epoch = None
        last_eval_acc = None
        if log_history:
            eval_accs = [(entry.get('epoch'), entry.get('eval_accuracy')) for entry in log_history if 'eval_accuracy' in entry]
            eval_accs = [(e, a) for e, a in eval_accs if e is not None and a is not None]
            if eval_accs:
                best_epoch, best_eval_acc = max(eval_accs, key=lambda x: x[1])
                last_epoch, last_eval_acc = eval_accs[-1]
        else:
            # Fallback: use eval_acc from metrics
            best_epoch = last_epoch = None
            best_eval_acc = last_eval_acc = metrics.get('eval_acc')
    except Exception:
        continue
    results.append({
        'params': params,
        'train_loss': metrics.get('train_loss'),
        'eval_loss': metrics.get('eval_loss'),
        'train_acc': metrics.get('train_acc'),
        'eval_acc': metrics.get('eval_acc'),
        'train_f1': metrics.get('train_f1'),
        'eval_f1': metrics.get('eval_f1'),
        'best_epoch': best_epoch,
        'best_eval_acc': best_eval_acc,
        'last_epoch': last_epoch,
        'last_eval_acc': last_eval_acc,
    })
# Sort by best eval accuracy descending
results = sorted(results, key=lambda x: (x['best_eval_acc'] if x['best_eval_acc'] is not None else -1), reverse=True)
# Write summary
with open(os.path.join(result_dir, "grid_search_summary.txt"), "w") as f:
    for r in results:
        f.write(f"params: {r['params']} | train_loss: {r['train_loss']} | eval_loss: {r['eval_loss']} | train_acc: {r['train_acc']} | eval_acc: {r['eval_acc']} | train_f1: {r['train_f1']} | eval_f1: {r['eval_f1']} | best_epoch: {r['best_epoch']} | best_eval_acc: {r['best_eval_acc']} | last_epoch: {r['last_epoch']} | last_eval_acc: {r['last_eval_acc']}\n")
print(f"Summary written to {os.path.join(result_dir, 'grid_search_summary.txt')}")
