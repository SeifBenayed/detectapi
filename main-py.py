#!/usr/bin/env python3
"""
Script principal pour FastDetectGPT avec interface utilisateur améliorée
"""

import argparse
import os
import time
import torch
import numpy as np
from fast_detect_gpt import FastDetectGPT, sigmoid

def analyze_text(detector, text):
    """Analyse un texte et affiche les résultats"""
    import time
    
    print("\nAnalyse en cours...")
    start_time = time.time()
    
    prob, crit, ntokens = detector.compute_prob(text)
    
    duration = time.time() - start_time
    
    print("\nRésultats de l'analyse:")
    print(f"- Nombre de tokens: {ntokens}")
    print(f"- Score du critère: {crit:.4f}")
    print(f"- Probabilité d'être généré par IA: {prob * 100:.1f}%")
    print(f"- Temps d'analyse: {duration:.2f} secondes")
    
    # Interprétation des résultats
    print("\nInterprétation:")
    if prob < 0.3:
        print("✅ Ce texte est très probablement écrit par un humain.")
    elif prob < 0.7:
        print("⚠️ Résultat incertain - le texte présente des caractéristiques mixtes.")
    else:
        print("🤖 Ce texte est très probablement généré par une IA.")
    
    # Conseils pour améliorer la détection
    if ntokens < 100:
        print("\nNote: Le texte est court, ce qui peut affecter la fiabilité de la détection.")
        print("Pour de meilleurs résultats, utilisez des textes plus longs (>200 tokens).")

def run_improved(args):
    """Interface utilisateur améliorée pour FastDetectGPT"""
    detector = FastDetectGPT(args)
    print('FastDetectGPT - Détecteur de texte généré par IA')
    print('-----------------------------------------------')
    print(f'Modèle de référence: {args.reference_model_name}')
    print(f'Modèle de scoring: {args.scoring_model_name}')
    print('-----------------------------------------------')
    
    while True:
        print("\nChoisissez une option:")
        print("1. Analyser un texte saisi manuellement")
        print("2. Analyser un fichier texte")
        print("3. Analyser plusieurs fichiers (mode batch)")
        print("4. Quitter")
        
        choice = input("\nVotre choix (1-4): ")
        
        if choice == '1':
            print("\nEntrez votre texte (appuyez sur Entrée deux fois pour commencer l'analyse):")
            lines = []
            while True:
                line = input()
                if len(line) == 0:
                    break
                lines.append(line)
            text = "\n".join(lines)
            
            if len(text) == 0:
                print("Aucun texte saisi.")
                continue
                
            analyze_text(detector, text)
            
        elif choice == '2':
            file_path = input("\nEntrez le chemin du fichier: ")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                print(f"\nAnalyse du fichier: {file_path} ({len(text)} caractères)")
                analyze_text(detector, text)
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier: {e}")
                
        elif choice == '3':
            dir_path = input("\nEntrez le chemin du dossier contenant les fichiers texte: ")
            file_pattern = input("Entrez le motif de fichier (ex: *.txt): ")
            output_path = input("Entrez le chemin du fichier de résultats (ex: results.csv): ")
            
            try:
                import glob
                import os
                import csv
                
                files = glob.glob(os.path.join(dir_path, file_pattern))
                print(f"\nAnalyse de {len(files)} fichiers...")
                
                with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Fichier', 'Probabilité IA (%)', 'Score', 'Tokens'])
                    
                    for file_path in files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                            
                            prob, crit, ntokens = detector.compute_prob(text)
                            writer.writerow([file_path, f"{prob * 100:.1f}", f"{crit:.4f}", ntokens])
                            print(f"Analysé: {file_path} - Probabilité IA: {prob * 100:.1f}%")
                            
                        except Exception as e:
                            print(f"Erreur lors de l'analyse de {file_path}: {e}")
                            writer.writerow([file_path, "ERREUR", "ERREUR", "ERREUR"])
                
                print(f"\nRésultats enregistrés dans {output_path}")
                
            except Exception as e:
                print(f"Erreur lors du traitement par lots: {e}")
                
        elif choice == '4':
            print("\nMerci d'avoir utilisé FastDetectGPT!")
            break
            
        else:
            print("\nOption non valide. Veuillez choisir entre 1 et 4.")

def calibrate_model_parameters(args, dataset_path):
    """
    Calibre les paramètres de régression logistique pour une paire de modèles
    
    Args:
        args: Arguments contenant reference_model_name et scoring_model_name
        dataset_path: Chemin vers un dataset contenant des textes humains et IA
    
    Returns:
        tuple: (k, b) paramètres de la régression logistique
    """
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    
    # Initialiser le détecteur avec les modèles spécifiés
    detector = FastDetectGPT(args)
    
    print(f"Calibration de la paire de modèles: {args.reference_model_name}_{args.scoring_model_name}")
    
    # Charger le dataset (format attendu: colonne 'text' et colonne 'label' avec 0=humain, 1=IA)
    try:
        data = pd.read_csv(dataset_path)
        print(f"Dataset chargé: {len(data)} échantillons")
    except:
        print(f"Erreur: Impossible de charger {dataset_path}")
        return None, None
    
    # Vérifier les colonnes nécessaires
    required_cols = ['text', 'label']
    if not all(col in data.columns for col in required_cols):
        print(f"Erreur: Le dataset doit contenir les colonnes {required_cols}")
        return None, None
    
    # Calculer le critère pour chaque texte
    print("Calcul des critères pour les échantillons...")
    crits = []
    for text in tqdm(data['text']):
        try:
            crit, _ = detector.compute_crit(text)
            crits.append(crit)
        except Exception as e:
            print(f"Erreur lors du calcul du critère: {e}")
            crits.append(np.nan)
    
    # Supprimer les valeurs NaN
    valid_indices = ~np.isnan(crits)
    crits_clean = np.array([c for c, v in zip(crits, valid_indices) if v])
    labels_clean = np.array([l for l, v in zip(data['label'], valid_indices) if v])
    
    print(f"Utilisation de {len(crits_clean)}/{len(crits)} échantillons valides pour la calibration")
    
    # Entraîner la régression logistique
    X = crits_clean.reshape(-1, 1)
    y = labels_clean
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Extraire les paramètres (k=coefficient, b=intercept)
    k = model.coef_[0][0]
    b = model.intercept_[0]
    
    # Évaluer la précision
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    
    print(f"Résultats de la calibration:")
    print(f"k: {k:.2f}, b: {b:.2f}, précision: {accuracy:.2f}")
    
    return k, b

def main():
    parser = argparse.ArgumentParser(description="FastDetectGPT - Détecteur de texte généré par IA")
    
    # Options principales
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch', 'calibrate'], default='interactive',
                        help="Mode d'exécution (interactive, batch, calibrate)")
    parser.add_argument('--reference_model_name', type=str, default="deepseek-v3-7b",
                        help="Nom du modèle de référence")
    parser.add_argument('--scoring_model_name', type=str, default="deepseek-v3-7b-chat",
                        help="Nom du modèle de scoring")
    
    # Options de périphérique et performance
    device_group = parser.add_argument_group('Options de périphérique')
    device_group.add_argument('--device', type=str, default="auto",
                             help="Périphérique à utiliser (cuda, cpu, mps, auto)")
    device_group.add_argument('--quantize', action='store_true',
                             help="Activer la quantification 8-bit (économise de la mémoire)")
    device_group.add_argument('--cache_dir', type=str, default="../cache",
                             help="Répertoire de cache pour les modèles")
    
    # Options pour le mode batch
    batch_group = parser.add_argument_group('Options pour le mode batch')
    batch_group.add_argument('--input_dir', type=str,
                            help="Répertoire contenant les fichiers à analyser")
    batch_group.add_argument('--file_pattern', type=str, default="*.txt",
                            help="Motif des fichiers à analyser")
    batch_group.add_argument('--output_file', type=str, default="results.csv",
                            help="Fichier de sortie pour les résultats batch")
    
    # Options pour le mode calibration
    calib_group = parser.add_argument_group('Options pour la calibration')
    calib_group.add_argument('--dataset_path', type=str,
                            help="Chemin vers le dataset de calibration")
    
    args = parser.parse_args()
    
    # Détection automatique du périphérique
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
            print("CUDA détecté - utilisation du GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
            print("Apple Silicon détecté - utilisation de MPS")
        else:
            args.device = 'cpu'
            print("Aucun accélérateur détecté - utilisation du CPU")
    
    # Exécution selon le mode
    if args.mode == 'interactive':
        run_improved(args)
        
    elif args.mode == 'batch':
        if not args.input_dir:
            print("Erreur: --input_dir est requis en mode batch")
            return
            
        import glob
        import os
        import csv
        
        # Vérifier le répertoire d'entrée
        if not os.path.isdir(args.input_dir):
            print(f"Erreur: {args.input_dir} n'est pas un répertoire valide")
            return
            
        # Initialiser le détecteur
        detector = FastDetectGPT(args)
        
        # Trouver les fichiers
        files = glob.glob(os.path.join(args.input_dir, args.file_pattern))
        if not files:
            print(f"Aucun fichier correspondant à {args.file_pattern} trouvé dans {args.input_dir}")
            return
            
        print(f"Analyse de {len(files)} fichiers...")
        
        # Créer le répertoire de sortie si nécessaire
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Analyser les fichiers
        with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fichier', 'Probabilité IA (%)', 'Score', 'Tokens'])
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    prob, crit, ntokens = detector.compute_prob(text)
                    writer.writerow([file_path, f"{prob * 100:.1f}", f"{crit:.4f}", ntokens])
                    print(f"Analysé: {file_path} - Probabilité IA: {prob * 100:.1f}%")
                    
                except Exception as e:
                    print(f"Erreur lors de l'analyse de {file_path}: {e}")
                    writer.writerow([file_path, "ERREUR", "ERREUR", "ERREUR"])
            
        print(f"Résultats enregistrés dans {args.output_file}")
        
    elif args.mode == 'calibrate':
        if not args.dataset_path:
            print("Erreur: --dataset_path est requis en mode calibrate")
            return
            
        k, b = calibrate_model_parameters(args, args.dataset_path)
        if k is not None and b is not None:
            print("\nPour utiliser ces paramètres calibrés, ajoutez la ligne suivante dans fast_detect_gpt.py:")
            print(f"'{args.reference_model_name}_{args.scoring_model_name}': ({k:.2f}, {b:.2f}),")

if __name__ == "__main__":
    main()
