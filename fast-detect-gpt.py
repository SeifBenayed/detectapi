# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
from model import load_tokenizer, load_model

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def get_roc_metrics(real_preds, sample_preds):
    """
    Compute ROC curve and ROC area for detection task
    """
    from sklearn.metrics import roc_curve, auc
    
    y_true = np.array([0] * len(real_preds) + [1] * len(sample_preds))
    y_score = np.array(real_preds + sample_preds)
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return fpr.tolist(), tpr.tolist(), roc_auc

def get_precision_recall_metrics(real_preds, sample_preds):
    """
    Compute Precision-Recall curve and PR area
    """
    from sklearn.metrics import precision_recall_curve, auc
    
    y_true = np.array([0] * len(real_preds) + [1] * len(sample_preds))
    y_score = np.array(real_preds + sample_preds)
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    return precision.tolist(), recall.tolist(), pr_auc

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.reference_model_name != args.scoring_model_name:
            self.reference_tokenizer = load_tokenizer(args.reference_model_name, args.cache_dir)
            self.reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
            self.reference_model.eval()
        # pre-calculated parameters by fitting a LogisticRegression on detection results
        # gpt-j-6B_gpt-neo-2.7B: k: 1.87, b: -2.19, acc: 0.82
        # gpt-neo-2.7B_gpt-neo-2.7B: k: 1.97, b: -1.47, acc: 0.83
        # falcon-7b_falcon-7b-instruct: k: 2.42, b: -2.83, acc: 0.90
        # deepseek-v3-7b_deepseek-v3-7b-chat: k: 2.20, b: -2.50, acc: 0.87 (estimation - calibration requise)
        # deepseek-r1-9b_deepseek-r1-9b-chat: k: 2.30, b: -2.65, acc: 0.89 (estimation - calibration requise)
        linear_params = {
            'gpt-j-6B_gpt-neo-2.7B': (1.87, -2.19),
            'gpt-neo-2.7B_gpt-neo-2.7B': (1.97, -1.47),
            'falcon-7b_falcon-7b-instruct': (2.42, -2.83),
            'deepseek-v3-7b_deepseek-v3-7b-chat': (2.20, -2.50),
            'deepseek-r1-9b_deepseek-r1-9b-chat': (2.30, -2.65),
        }
        key = f'{args.reference_model_name}_{args.scoring_model_name}'
        
        if key in linear_params:
            self.linear_k, self.linear_b = linear_params[key]
        else:
            # Valeurs par défaut si les paramètres ne sont pas calibrés
            print(f"Warning: No calibrated parameters for {key}, using default values")
            self.linear_k, self.linear_b = 2.0, -2.0

    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.reference_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)

    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        prob = sigmoid(self.linear_k * crit + self.linear_b)
        return prob, crit, ntoken


# run interactive local inference
def run(args):
    detector = FastDetectGPT(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        lines = []
        while True:
            line = input()
            if len(line) == 0:
                break
            lines.append(line)
        text = "\n".join(lines)
        if len(text) == 0:
            break
        # estimate the probability of machine generated text
        prob, crit, ntokens = detector.compute_prob(text)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
        print()

def load_data(dataset_file):
    """
    Load data from a dataset file (can be JSON or other formats)
    """
    import json
    import os
    
    # Attempt to determine file type from extension
    _, ext = os.path.splitext(dataset_file)
    
    if ext.lower() == '.json':
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Default fallback - try to load as JSON
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            raise ValueError(f"Unsupported dataset file format: {dataset_file}")
    
    return data

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir, args.dataset)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
        
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    
    # evaluate criterion
    if getattr(args, 'discrepancy_analytic', True):
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits_ref, logits_score, labels)
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mode interactif ou évaluation
    parser.add_argument('--mode', type=str, choices=['interactive', 'evaluate'], default='interactive',
                        help="Mode: 'interactive' pour l'inférence locale, 'evaluate' pour l'évaluation sur un dataset")
    
    # Arguments pour l'interactif
    parser.add_argument('--reference_model_name', type=str, default="deepseek-v3-7b",
                        help="Modèle de référence (deepseek-v3-7b, gpt-neo-2.7B, etc.)")
    parser.add_argument('--scoring_model_name', type=str, default="deepseek-v3-7b-chat",
                        help="Modèle de scoring (deepseek-v3-7b-chat, gpt-neo-2.7B, etc.)")
    
    # Arguments pour l'évaluation
    parser.add_argument('--output_file', type=str, default="./results/eval_results",
                        help="Préfixe du fichier de sortie pour l'évaluation")
    parser.add_argument('--dataset', type=str, default="xsum",
                        help="Nom du dataset")
    parser.add_argument('--dataset_file', type=str, default="./data/dataset.json",
                        help="Chemin vers le fichier de dataset pour l'évaluation")
    parser.add_argument('--discrepancy_analytic', action='store_true',
                        help="Utiliser la version analytique (plus rapide) du calcul de divergence")
    
    # Arguments communs
    parser.add_argument('--seed', type=int, default=0,
                        help="Graine aléatoire")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositif à utiliser (cuda, cpu)")
    parser.add_argument('--cache_dir', type=str, default="../cache",
                        help="Répertoire de cache pour les modèles")
    
    args = parser.parse_args()
    
    if args.mode == 'interactive':
        run(args)
    else:
        experiment(args)
