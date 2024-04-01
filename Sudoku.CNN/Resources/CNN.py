# may require update keras
#!pip install keras --upgrade
import os

import numpy as np
from keras import models, utils
import pandas as pd
from huggingface_hub import hf_hub_download
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential

class SudokuSolver:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        return models.load_model(model_path)

    def __call__(self, puzzles):
        puzzles = puzzles.copy()
        # We want to ensure the array has the shape (1, 9, 9)
        target_shape = (1, 9, 9)
        # Check if the current shape matches the target shape
        if puzzles.shape != target_shape:
            # Reshape the array to the target shape
            puzzles = puzzles.reshape(target_shape)
        
         # Begin processing 
        for _ in range((puzzles == 0).sum((1, 2)).max()):
            model_preds = self.model.predict(
                utils.to_categorical(puzzles, num_classes=10), verbose=0
            )
            preds = np.zeros((puzzles.shape[0], 81, 9))
            for i in range(9):
                for j in range(9):
                    preds[:, i * 9 + j] = model_preds[f"position_{i+1}_{j+1}"]
            probs = preds.max(2)
            values = preds.argmax(2) + 1
            zeros = (puzzles == 0).reshape((puzzles.shape[0], 81))
            for grid, prob, value, zero in zip(puzzles, probs, values, zeros):
                if any(zero):
                    where = np.where(zero)[0]
                    confidence_position = where[prob[zero].argmax()]
                    confidence_value = value[confidence_position]
                    grid.flat[confidence_position] = confidence_value
        return puzzles


def empiric_testing(): # was previously if __name__ == "__main__":

    # For testing purposes
    # Replace puzzles by your puzzle
    #Define FFN solver
    solver = SudokuSolver(
        hf_hub_download(
            repo_id="Ritvik19/SuDoKu-Net",
            filename="ffn.keras",
            revision="b57f9a0538e28249c92733cb025c87d07831baa1",
        )
    )
    # Char-Grid layout format 
    puzzle = """
    0 0 4 3 0 0 2 0 9
    0 0 5 0 0 9 0 0 1
    0 7 0 0 6 0 0 4 3
    0 0 6 0 0 2 0 8 7
    1 9 0 0 0 7 4 0 0
    0 5 0 0 8 3 0 0 0
    6 0 0 0 0 0 1 0 5
    0 0 3 5 0 8 6 9 0
    0 4 2 9 1 0 3 0 0
    """.strip()
    puzzle = np.array([int(digit) for digit in puzzle.split()]).reshape(1, 9, 9)
    solution = solver(puzzle)[0]
    print("Sudoku au format chaine de caractères \"0 0 4 3 0 0 2 0 9 ...\"")
    print(pd.DataFrame(solution))
    print(check_sudoku(solution))

    # 2D matrix format
    puzzle2 = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ]
    puzzle2_array = np.array(puzzle2).reshape(1,9,9)
    solution2 = solver(puzzle2_array)[0]
    print("Sudoku au format matriciel  [ [5, 3, 0, 0, 7, 0, 0, 0, 0], ...")
    print(pd.DataFrame(solution2))
    print(check_sudoku(solution2))

    # Full string format
    unsolved_sudoku = "467100805912835607085647192296351470708920351531408926073064510624519783159783064"
    # Convert into numpy array
    unsolved_sudoku = np.array([int(digit) for digit in unsolved_sudoku]).reshape(9, 9)
    solved_sudoku = solver(unsolved_sudoku)[0]
    print("Sudoku au format suite de caractères 467100805912835607...")
    print(pd.DataFrame(solved_sudoku))
    print(check_sudoku(solved_sudoku))


    
def check_sudoku(grid): #From https://scipython.com/book2/chapter-6-numpy/examples/checking-a-sudoku-grid-for-validity/
    """ Return True if grid is a valid Sudoku square, otherwise False. """
    for i in range(9):
        # j, k index top left hand corner of each 3x3 tile
        j, k = (i // 3) * 3, (i % 3) * 3
        if len(set(grid[i,:])) != 9 or len(set(grid[:,i])) != 9 or len(set(grid[j:j+3, k:k+3].ravel())) != 9:
            return False
    return True



def solve_sudoku(grid, row=0, col=0):
    solver = SudokuSolver(
        hf_hub_download(
            repo_id="Ritvik19/SuDoKu-Net",
            filename="ffn.keras",
            revision="b57f9a0538e28249c92733cb025c87d07831baa1",
        )
    )
    solved = solver(grid)[0]
    #print(solved)
    return check_sudoku(solved)

def CNNSolver():
    return SudokuSolver(
        hf_hub_download(
            repo_id="Ritvik19/SuDoKu-Net",
            filename="ffn.keras",
            revision="b57f9a0538e28249c92733cb025c87d07831baa1",
        )
    )


# Définir `instance` uniquement si non déjà défini par PythonNET
if 'instance' not in locals():
    instance = np.array([
        [0,0,0,0,9,4,0,3,0],
        [0,0,0,5,1,0,0,0,7],
        [0,8,9,0,0,0,0,4,0],
        [0,0,0,0,0,0,2,0,8],
        [0,6,0,2,0,1,0,5,0],
        [1,0,2,0,0,0,0,0,0],
        [0,7,0,0,0,0,5,2,0],
        [9,0,0,0,6,5,0,0,0],
        [0,4,0,9,7,0,0,0,0]
    ], dtype=int)

start = default_timer()
# Exécuter la résolution de Sudoku
if solve_sudoku(instance):
    # print("Sudoku résolu par backtracking avec succès.")
    result = instance  # `result` sera utilisé pour récupérer la grille résolue depuis C#
else:
    print("Aucune solution trouvée.")
execution = default_timer() - start
print("Le temps de résolution est de : ", execution * 1000, " ms")