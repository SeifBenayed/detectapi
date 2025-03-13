#!/usr/bin/env python3
"""
Script pour visualiser les résultats d'évaluation des différents modèles
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc

def plot_distribution(results_file, output_dir):
    """
    Trace la distribution des scores pour les textes réels et générés
    """
    # Charger les résultats
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extraire les scores
    real_scores = results['predictions']['real']
    fake_scores = results['predictions']['samples']
    
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    # Histogramme des distributions
    bins = np.linspace(min(min(real_scores), min(fake_scores)), 
                       max(max(real_scores), max(fake_scores)), 50)
    
    plt.hist(real_scores, bins=bins, alpha=0.7, label='Texte humain', color='green')
    plt.hist(fake_scores, bins=bins, alpha=0.7, label='Texte IA', color='red')
    
    # Statistiques
    real_mean = np.mean(real_scores)
    fake_mean = np.mean(fake_scores)
    
    plt.axvline(real_mean, color='darkgreen', linestyle='dashed', linewidth=2, 
                label=f'Moyenne humain: {real_mean:.2f}')
    plt.axvline(fake_mean, color='darkred', linestyle='dashed', linewidth=2, 
                label=f'Moyenne IA: {fake_mean:.2f}')
    
    # Configurer le graphique
    plt.title(f'Distribution des scores de détection\n{os.path.basename(results_file)}', fontsize=14)
    plt.xlabel('Score de détection', fontsize=12)
    plt.ylabel('Nombre d\'échantillons', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Enregistrer
    output_file = os.path.join(output_dir, f"{os.path.basename(results_file).replace('.json', '')}_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution enregistrée dans {output_file}")

def plot_roc_curves(results_files, output_dir, labels=None):
    """
    Trace les courbes ROC pour plusieurs résultats
    """
    plt.figure(figsize=(10, 8))
    
    for i, results_file in enumerate(results_files):
        # Charger les résultats
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extraire les données ROC
        fpr = results['metrics']['fpr']
        tpr = results['metrics']['tpr']
        roc_auc = results['metrics']['roc_auc']
        
        # Déterminer le label
        if labels and i < len(labels):
            label = f"{labels[i]} (AUC = {roc_auc:.3f})"
        else:
            label = f"{os.path.basename(results_file)} (AUC = {roc_auc:.3f})"
        
        # Tracer la courbe
        plt.plot(fpr, tpr, lw=2, label=label)
    
    # Ligne de référence
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (AUC = 0.5)')
    
    # Configurer le graphique
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Taux de vrais positifs', fontsize=12)
    plt.title('Courbes ROC comparatives', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Enregistrer
    output_file = os.path.join(output_dir, "comparative_roc_curves.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Courbes ROC comparatives enregistrées dans {output_file}")

def plot_pr_curves(results_files, output_dir, labels=None):
    """
    Trace les courbes précision-rappel pour plusieurs résultats
    """
    plt.figure(figsize=(10, 8))
    
    for i, results_file in enumerate(results_files):
        # Charger les résultats
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extraire les données de précision-rappel
        precision = results['pr_metrics']['precision']
        recall = results['pr_metrics']['recall']
        pr_auc = results['pr_metrics']['pr_auc']
        
        # Déterminer le label
        if labels and i < len(labels):
            label = f"{labels[i]} (AUC = {pr_auc:.3f})"
        else:
            label = f"{os.path.basename(results_file)} (AUC = {pr_auc:.3f})"
        
        # Tracer la courbe
        plt.plot(recall, precision, lw=2, label=label)
    
    # Configurer le graphique
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Rappel', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Courbes précision-rappel comparatives', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    
    # Enregistrer
    output_file = os.path.join(output_dir, "comparative_pr_curves.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Courbes précision-rappel comparatives enregistrées dans {output_file}")

def plot_metrics_comparison(summary_file, output_dir):
    """
    Crée un graphique de comparaison des métriques entre modèles
    """
    # Charger le résumé
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Trier par PR AUC
    summary = sorted(summary, key=lambda x: x['pr_auc'])
    
    # Extraire les données
    models = [item['description'] for item in summary]
    roc_auc = [item['roc_auc'] for item in summary]
    pr_auc = [item['pr_auc'] for item in summary]
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    # Barres pour ROC AUC et PR AUC
    rects1 = ax.bar(x - width/2, roc_auc, width, label='ROC AUC', color='royalblue')
    rects2 = ax.bar(x + width/2, pr_auc, width, label='PR AUC', color='orangered')
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Configurer le graphique
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des performances de détection')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Enregistrer
    output_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparaison des modèles enregistrée dans {output_file}")

def analyze_example_texts(args):
    """
    Analyse quelques exemples de textes avec différents modèles pour comparer les scores
    """
    import torch
    from fast_detect_gpt import FastDetectGPT
    
    print("Analyse d'exemples de textes avec différents modèles...")
    
    # Liste des configurations de modèles à comparer
    model_configs = []
    
    # Détecter les paires disponibles à partir du résumé
    if args.summary_file and os.path.exists(args.summary_file):
        with open(args.summary_file, 'r') as f:
            summary = json.load(f)
            
        for item in summary:
            model_configs.append((
                item['reference_model'],
                item['scoring_model'],
                item['description']
            ))
    # Sinon, utiliser des configurations par défaut
    else:
        model_configs = [
            ("deepseek-v3-7b", "deepseek-v3-7b-chat", "DeepSeek V3 7B"),
            ("gpt-neo-2.7B", "gpt-neo-2.7B", "GPT-Neo 2.7B"),
        ]
    
    # Exemples de textes
    example_texts = [
        {
            "title": "Texte humain court",
            "text": "L'intelligence artificielle est fascinante mais elle soulève aussi des questions éthiques importantes. J'ai récemment lu un article qui abordait ce sujet et je me demande comment nous allons résoudre ces problèmes à l'avenir."
        },
        {
            "title": "Texte IA court",
            "text": "L'intelligence artificielle représente une avancée technologique majeure qui transforme de nombreux secteurs. Elle offre des possibilités significatives d'automatisation, d'analyse de données et d'assistance dans diverses tâches. Cependant, il est essentiel de considérer les implications éthiques et sociales de son déploiement."
        },
        {
            "title": "Texte humain académique",
            "text": "Dans mon dernier article de recherche, j'ai exploré les interactions entre politique monétaire et inégalités socio-économiques. Les résultats préliminaires suggèrent un lien complexe qui mérite d'être approfondi. Je pense qu'il faudrait mener des études complémentaires avec des méthodologies mixtes."
        },
        {
            "title": "Texte IA académique",
            "text": "Cette étude examine l'impact des politiques monétaires sur les inégalités socio-économiques à travers une analyse quantitative de données macroéconomiques. Les résultats révèlent une corrélation significative entre l'assouplissement monétaire et l'augmentation des inégalités patrimoniales, tandis que les effets sur les inégalités de revenus sont plus nuancés et dépendent de facteurs contextuels spécifiques."
        }
    ]
    
    # Créer le répertoire de sortie
    output_file = os.path.join(args.output_dir, "example_analysis_results.csv")
    
    # Analyser les exemples avec chaque configuration
    results = []
    print(f"Analyse de {len(example_texts)} exemples avec {len(model_configs)} configurations de modèles...")
    
    for ref_model, score_model, description in model_configs:
        try:
            # Créer l'instance du détecteur
            detector_args = argparse.Namespace(
                reference_model_name=ref_model,
                scoring_model_name=score_model,
                device=args.device,
                cache_dir=args.cache_dir
            )
            
            print(f"\nUtilisation de {description} ({ref_model} → {score_model})...")
            detector = FastDetectGPT(detector_args)
            
            # Analyser chaque exemple
            for example in example_texts:
                title = example["title"]
                text = example["text"]
                
                print(f"  Analyse de '{title}'...")
                prob, crit, ntokens = detector.compute_prob(text)
                
                results.append({
                    "model": description,
                    "example": title,
                    "probability": prob * 100,
                    "criterion": crit,
                    "tokens": ntokens
                })
                
                print(f"    Probabilité IA: {prob * 100:.1f}%, Critère: {crit:.4f}, Tokens: {ntokens}")
            
        except Exception as e:
            print(f"Erreur avec la configuration {description}: {e}")
            
    # Enregistrer les résultats
    if results:
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["model", "example", "probability", "criterion", "tokens"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\nRésultats de l'analyse enregistrés dans {output_file}")
        
        # Créer un graphique de comparaison
        create_example_comparison_plot(results, args.output_dir)
    
def create_example_comparison_plot(results, output_dir):
    """
    Crée un graphique comparant les probabilités IA pour différents exemples et modèles
    """
    # Extraire les modèles et exemples uniques
    models = sorted(set(item["model"] for item in results))
    examples = sorted(set(item["example"] for item in results))
    
    # Préparer les données
    data = {}
    for model in models:
        data[model] = {}
        for example in examples:
            for item in results:
                if item["model"] == model and item["example"] == example:
                    data[model][example] = item["probability"]
                    break
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Positions sur l'axe X
    x = np.arange(len(examples))
    width = 0.8 / len(models)
    
    # Tracer les barres pour chaque modèle
    for i, model in enumerate(models):
        values = [data[model].get(example, 0) for example in examples]
        positions = x + (i - len(models)/2 + 0.5) * width
        ax.bar(positions, values, width, label=model)
    
    # Configurer le graphique
    ax.set_ylabel('Probabilité d\'être généré par IA (%)', fontsize=12)
    ax.set_title('Comparaison des probabilités IA par modèle et exemple', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(examples, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Ligne de référence à 50%
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    
    # Limites de l'axe Y
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    # Enregistrer
    output_file = os.path.join(output_dir, "example_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique de comparaison des exemples enregistré dans {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualiser les résultats d'évaluation")
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Répertoire contenant les fichiers de résultats")
    parser.add_argument('--output_dir', type=str, default="./visualizations",
                        help="Répertoire pour les visualisations")
    parser.add_argument('--summary_file', type=str,
                        help="Fichier de résumé (optionnel)")
    parser.add_argument('--analyze_examples', action='store_true',
                        help="Analyser des exemples de textes avec différents modèles")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositif à utiliser pour l'analyse d'exemples")
    parser.add_argument('--cache_dir', type=str, default="../cache",
                        help="Répertoire de cache pour les modèles")
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie si nécessaire
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Chercher les fichiers de résultats
    results_files = []
    for file in os.listdir(args.results_dir):
        if file.endswith('.sampling_discrepancy_analytic.json'):
            results_files.append(os.path.join(args.results_dir, file))
    
    if not results_files:
        print(f"Aucun fichier de résultats trouvé dans {args.results_dir}")
    else:
        # Tracer la distribution pour chaque fichier
        for results_file in results_files:
            plot_distribution(results_file, args.output_dir)
        
        # Tracer les courbes comparatives
        try:
            labels = [os.path.basename(f).split('_')[1] + '_' + os.path.basename(f).split('_')[2] for f in results_files]
            plot_roc_curves(results_files, args.output_dir, labels)
            plot_pr_curves(results_files, args.output_dir, labels)
        except Exception as e:
            print(f"Erreur lors du tracé des courbes comparatives: {e}")
        
        # Tracer la comparaison des modèles si un fichier de résumé est fourni
        if args.summary_file and os.path.exists(args.summary_file):
            try:
                plot_metrics_comparison(args.summary_file, args.output_dir)
            except Exception as e:
                print(f"Erreur lors du tracé de la comparaison des modèles: {e}")
    
    # Analyser des exemples de textes si demandé
    if args.analyze_examples:
        try:
            import torch
            analyze_example_texts(args)
        except Exception as e:
            print(f"Erreur lors de l'analyse des exemples: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Visualisations enregistrées dans {args.output_dir}")

if __name__ == "__main__":
    main()
