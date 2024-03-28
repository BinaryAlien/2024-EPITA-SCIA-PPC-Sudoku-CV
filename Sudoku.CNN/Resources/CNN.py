import numpy as np
from timeit import default_timer

import pandas as pd
from sklearn.model_selection import train_test_split


def get_model():

    model = keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(81*9))
    model.add(Reshape((-1, 9)))
    model.add(Activation('softmax'))
    
    return model

def train_model(model):
    data = pd.read_csv("../data.csv")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = get_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model


'''def solve_sudoku(grid, row=0, col=0):
    # Trouver la prochaine cellule vide
    for i in range(row, 9):
        for j in range(col if i == row else 0, 9):
            if grid[i, j] == 0:
                for num in range(1, 10):
                    if is_valid(grid, i, j, num):
                        grid[i, j] = num
                        if solve_sudoku(grid, i, j + 1):
                            return True
                        grid[i, j] = 0
                return False
    return True'''

    
def solve_sudoku(grid, row=0, col=0):
    model = get_model()
    trained_model = train_model(model)

    return trained_model.predict(grid)




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