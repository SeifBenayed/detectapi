#!/usr/bin/env python3
"""
Script pour évaluer les modèles DeepSeek dans le cadre de la détection de texte généré par IA
"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm

def run_evaluation_pipeline(args):
    """
    Exécute l'évaluation pour plusieurs paires de modèles DeepSeek
    """
    # Configurations des modèles à évaluer
    model_pairs = []
    
    # Paires DeepSeek V3
    if args.eval_deepseek_v3:
        model_pairs.extend([
            ("deepseek-v3-7b", "deepseek-v3-7b-chat", "DeepSeek V3 7B"),
        ])
        if args.include_large_models:
            model_pairs.append(("deepseek-v3-34b", "deepseek-v3-34b-chat", "DeepSeek V3 34B"))
            
    # Paires DeepSeek R1
    if args.eval_deepseek_r1:
        model_pairs.extend([
            ("deepseek-r1-lite", "deepseek-r1-lite-chat", "DeepSeek R1 Lite"),
            ("deepseek-r1-9b", "deepseek-r1-9b-chat", "DeepSeek R1 9B"),
        ])
    
    # Paires de référence (modèles standards)
    if args.eval_reference_models:
        model_pairs.extend([
            ("gpt-neo-2.7B", "gpt-neo-2.7B", "GPT-Neo 2.7B"),
            ("gpt-j-6B", "gpt-neo-2.7B", "GPT-J 6B → GPT-Neo 2.7B"),
            ("falcon-7b", "falcon-7b-instruct", "Falcon 7B"),
        ])
    
    # Si aucune paire n'est spécifiée, évaluer uniquement la paire DeepSeek V3 7B
    if not model_pairs:
        model_pairs.append(("deepseek-v3-7b", "deepseek-v3-7b-chat", "DeepSeek V3 7B"))
    
    # Configuration par défaut
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Exécuter l'évaluation pour chaque paire
    results_summary = []
    for ref_model, score_model, description in model_pairs:
        print(f"\n{'='*80}")
        print(f"Évaluation de {description}: {ref_model} → {score_model}")
        print(f"{'='*80}")
        
        # Construire le chemin de sortie
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{ref_model}_{score_model}")
        
        # Construire la commande
        cmd_args = argparse.Namespace(
            output_file=output_file,
            dataset=args.dataset,
            dataset_file=args.dataset_file,
            reference_model_name=ref_model,
            scoring_model_name=score_model,
            discrepancy_analytic=True,  # Utiliser la version analytique (plus rapide)
            seed=args.seed,
            device=args.device,
            cache_dir=args.cache_dir
        )
        
        try:
            # Importer et exécuter la fonction d'expérimentation
            from fast_detect_gpt import experiment
            result = experiment(cmd_args)
            
            # Charger les résultats et extraire les métriques
            results_file = f'{output_file}.sampling_discrepancy_analytic.json'
            with open(results_file, 'r') as fin:
                results = json.load(fin)
                
            roc_auc = results['metrics']['roc_auc']
            pr_auc = results['pr_metrics']['pr_auc']
            
            # Ajouter au résumé
            results_summary.append({
                'reference_model': ref_model,
                'scoring_model': score_model,
                'description': description,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'results_file': results_file
            })
            
            print(f"\nRésultats pour {description}:")
            print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
            
        except Exception as e:
            print(f"\nErreur lors de l'évaluation de {description}: {e}")
            import traceback
            traceback.print_exc()
    
    # Afficher le résumé des résultats
    if results_summary:
        print("\n" + "="*80)
        print("RÉSUMÉ DES RÉSULTATS")
        print("="*80)
        print(f"{'Modèles':<30} {'ROC AUC':<10} {'PR AUC':<10}")
        print("-"*50)
        
        for result in sorted(results_summary, key=lambda x: x['pr_auc'], reverse=True):
            print(f"{result['description']:<30} {result['roc_auc']:.4f}    {result['pr_auc']:.4f}")
        
        # Enregistrer le résumé dans un fichier
        summary_file = os.path.join(args.output_dir, f"{args.dataset}_summary.json")
        with open(summary_file, 'w') as fout:
            json.dump(results_summary, fout, indent=2)
            print(f"\nRésumé enregistré dans {summary_file}")
    
    return results_summary

def create_test_dataset(args):
    """
    Crée un dataset de test synthétique pour l'évaluation
    """
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Création d'un dataset de test synthétique...")
    
    # Nombre d'échantillons
    n_samples = args.test_samples
    
    # Charger un modèle pour générer des textes
    if args.device == 'cpu':
        print("Attention: La génération sur CPU peut être lente...")
        model_name = "distilgpt2"  # modèle plus petit pour CPU
    else:
        model_name = "gpt2"
    
    print(f"Utilisation de {model_name} pour la génération de textes...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(args.device)
    
    # Textes d'amorce
    prompts = [
        "L'intelligence artificielle est",
        "Les avantages de la technologie moderne comprennent",
        "Les scientifiques ont découvert récemment que",
        "Dans le monde de l'économie, il est important de",
        "Lors d'un voyage en montagne, il faut toujours",
        "La cuisine française se caractérise par",
        "Les défis environnementaux du 21ème siècle incluent",
        "L'histoire de l'art a été marquée par",
        "Dans le domaine de l'éducation, les méthodes modernes",
        "La littérature contemporaine explore souvent des thèmes"
    ]
    
    # Créer le dataset
    dataset = {"original": [], "sampled": []}
    
    for i in tqdm(range(n_samples), desc="Génération d'échantillons"):
        # Sélectionner un prompt aléatoire
        prompt = random.choice(prompts)
        
        # Générer un texte "humain" (court, pour simulation)
        human_text = prompt + " " + random.choice([
            "un domaine en pleine expansion qui suscite à la fois fascination et inquiétude.",
            "la possibilité de communiquer instantanément avec des personnes du monde entier.",
            "certains comportements humains sont fortement influencés par des facteurs génétiques.",
            "comprendre les mécanismes de marché et d'anticiper les tendances économiques.",
            "emporter suffisamment d'eau et de nourriture pour éviter les situations dangereuses.",
            "l'utilisation d'ingrédients de qualité et des techniques précises de préparation.",
            "le réchauffement climatique et la pollution des océans par les microplastiques.",
            "différents mouvements artistiques qui reflètent l'évolution de la société.",
            "privilégient l'autonomie et le développement de l'esprit critique des élèves.",
            "comme l'identité, la mémoire et les relations humaines dans un monde numérique."
        ])
        
        # Générer un texte par IA
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2
        )
        ai_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Ajouter au dataset
        dataset["original"].append(human_text)
        dataset["sampled"].append(ai_text)
    
    # Enregistrer le dataset
    os.makedirs(os.path.dirname(args.test_dataset_path), exist_ok=True)
    with open(args.test_dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset de test créé avec {n_samples} échantillons et enregistré dans {args.test_dataset_path}")
    
    return args.test_dataset_path

def main():
    parser = argparse.ArgumentParser(description="Évaluer les modèles DeepSeek pour la détection de texte IA")
    
    # Options du dataset
    dataset_group = parser.add_argument_group('Options du dataset')
    dataset_group.add_argument('--dataset', type=str, default="test",
                        help="Nom du dataset")
    dataset_group.add_argument('--dataset_file', type=str,
                        help="Chemin vers le fichier de dataset")
    dataset_group.add_argument('--create_test_dataset', action='store_true',
                        help="Créer un dataset de test synthétique")
    dataset_group.add_argument('--test_dataset_path', type=str, default="./data/test_dataset.json",
                        help="Chemin pour le dataset de test")
    dataset_group.add_argument('--test_samples', type=int, default=100,
                        help="Nombre d'échantillons dans le dataset de test")
                        
    # Options des modèles
    models_group = parser.add_argument_group('Options des modèles')
    models_group.add_argument('--eval_deepseek_v3', action='store_true',
                        help="Évaluer les modèles DeepSeek V3")
    models_group.add_argument('--eval_deepseek_r1', action='store_true',
                        help="Évaluer les modèles DeepSeek R1")
    models_group.add_argument('--eval_reference_models', action='store_true',
                        help="Évaluer les modèles de référence standards (GPT-Neo, GPT-J, Falcon)")
    models_group.add_argument('--include_large_models', action='store_true',
                        help="Inclure les modèles les plus grands (34B+)")
    
    # Options générales
    parser.add_argument('--output_dir', type=str, default="./results/deepseek",
                        help="Répertoire pour les résultats")
    parser.add_argument('--seed', type=int, default=42,
                        help="Graine aléatoire")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositif à utiliser (cuda, cpu)")
    parser.add_argument('--cache_dir', type=str, default="../cache",
                        help="Répertoire de cache pour les modèles")
    
    args = parser.parse_args()
    
    # Créer un dataset de test si demandé
    if args.create_test_dataset:
        args.dataset_file = create_test_dataset(args)
    
    # Vérifier que le dataset est spécifié
    if not args.dataset_file:
        parser.error("Vous devez spécifier --dataset_file ou utiliser --create_test_dataset")
    
    # Exécuter l'évaluation
    run_evaluation_pipeline(args)

if __name__ == "__main__":
    main()
