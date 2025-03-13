#!/bin/bash
# Script pour exécuter FastDetectGPT avec différentes options

# Créer les répertoires nécessaires
mkdir -p data
mkdir -p results
mkdir -p visualizations
mkdir -p cache

# Définir le répertoire de cache pour les modèles
export CACHE_DIR="./cache"

# Fonction d'aide
show_help() {
    echo "Script pour exécuter FastDetectGPT avec différentes options"
    echo ""
    echo "Usage: ./run.sh <command> [options]"
    echo ""
    echo "Commandes disponibles:"
    echo "  detect      - Mode interactif pour détecter si un texte est généré par IA"
    echo "  batch       - Analyser plusieurs fichiers texte"
    echo "  calibrate   - Calibrer les paramètres pour une paire de modèles"
    echo "  evaluate    - Évaluer des paires de modèles sur un dataset"
    echo "  visualize   - Visualiser les résultats d'évaluation"
    echo "  test        - Créer un dataset de test et l'évaluer"
    echo "  help        - Afficher cette aide"
    echo ""
    echo "Exemple: ./run.sh detect --reference_model_name deepseek-v3-7b --scoring_model_name deepseek-v3-7b-chat"
}

# Vérifier les arguments
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

# Récupérer la commande
COMMAND=$1
shift

# Exécuter la commande appropriée
case "$COMMAND" in
    detect)
        # Mode interactif
        python main.py --mode interactive --cache_dir "$CACHE_DIR" "$@"
        ;;
    batch)
        # Mode batch
        python main.py --mode batch --cache_dir "$CACHE_DIR" "$@"
        ;;
    calibrate)
        # Mode calibration
        python main.py --mode calibrate --cache_dir "$CACHE_DIR" "$@"
        ;;
    evaluate)
        # Évaluation
        python evaluation.py --cache_dir "$CACHE_DIR" --output_dir "./results" "$@"
        ;;
    visualize)
        # Visualisation
        python visualization.py --results_dir "./results" --output_dir "./visualizations" "$@"
        ;;
    test)
        # Créer un dataset de test et l'évaluer
        python evaluation.py --create_test_dataset --test_dataset_path "./data/test_dataset.json" --cache_dir "$CACHE_DIR" --output_dir "./results" "$@"
        python visualization.py --results_dir "./results" --output_dir "./visualizations" --summary_file "./results/test_summary.json"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Commande inconnue: $COMMAND"
        show_help
        exit 1
        ;;
esac
