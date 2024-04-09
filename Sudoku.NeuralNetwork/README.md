Bienvenue sur le dépôt du TP Sudoku sur les réseaux de neurones

## Présentation des solvers

Vous trouverez ici la documentation qui accompagne tout les solvers.


## Linear

Chaque groupe est invité à créer un [Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) de ce dépôt principal muni d'un compte sur Github, et d'indiquer dans le fil de suivi de projet du groupe sur le forum son adresse. 

Vous pourrez ensuite travailler de façon collaborative sur ce fork  en  [attribuant les permissions d'éditions](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository) aux autres membres du groupe, en clonant votre fork sur vos machines, par le biais de validations (commits), de push pour remonter les validations sur le server, et de pulls/tirages sur les machines locales des utilisateurs du groupe habilités sur le fork. 

## FeedForward

Accéder au dossier Feed Forward
Intaller les packages nécessaires: 

```bash
pip install -r requirement.txt
```

# Dataset

Le modèle est entraîné sur un ensemble de données comprenant 17 millions de grilles de Sudoku provenant de diverses sources ([Fiche de données du jeu de données](https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset)). Le jeu de données comprend des configurations de puzzles, des solutions, des niveaux de difficulté et des sources.

# Models

2 models sont disponibles

1. **Feed Forward Neural Network (FFN):** dans `ffn.py`
2. **Convolutional Neural Network (CNN):** dans `cnn.py`

# Training


Pour lancer le training, il suffit de lancer la commande suivante acec les argument que vous voulez :

```bash
usage: main.py [-h] [--train TRAIN [TRAIN ...]] [--valid VALID [VALID ...]] [--model-load MODEL_LOAD] [--model-save MODEL_SAVE] [--model-type MODEL_TYPE][--num-delete NUM_DELETE] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--resume]
```

# Solve

Pour lancer une simulation de résolution de grille de sudoku, il suffit de :

Pour CNN:
```bash
python solve_cnn.py
```
Pour FFN:
```bash
python solve_ffn.py
```


## Convolutional

