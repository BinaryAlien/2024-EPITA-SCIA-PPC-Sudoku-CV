import copy
import keras
import numpy as np
from module import get_model
from data import get_data


# x_train, x_test, y_train, y_test = get_data('sudoku.csv')

# model = get_model()

# adam = keras.optimizers.Adam(learning_rate=.001)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

# model.fit(x_train, y_train, batch_size=32, epochs=2)

# model.save("ez.keras")


def norm(a):
    return (a/9)-.5

def denorm(a):
    return (a+.5)*9

def inference_sudoku(sample):
    
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    
    feat = copy.copy(sample)
    
    while(1):
    
        out = model.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
            
        prob_new = prob*mask
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    
    return pred

def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
    print(correct/feats.shape[0])


game = ["85...24..72......9..4.........1.7..23.5...9...4...........8..7..17..........36.4.", 
"..53.....8......2..7..1.5..4....53...1..7...6..32...8..6.5....9..4....3......97..",
"12..4......5.69.1...9...5.........7.7...52.9..3......2.9.6...5.4..9..8.1..3...9.4",
"...57..3.1......2.7...234......8...4..7..4...49....6.5.42...3.....7..9....18.....",
"7..1523........92....3.....1....47.8.......6............9...5.6.4.9.7...8....6.1.",
"1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..4......7..7...3..",
"1...34.8....8..5....4.6..21.18......3..1.2..6......81.52..7.9....6..9....9.64...2",
"...92......68.3...19..7...623..4.1....1...7....8.3..297...8..91...5.72......64...",
".6.5.4.3.1...9...8.........9...5...6.4.6.2.7.7...4...5.........4...8...1.5.2.3.4.",
"7.....4...2..7..8...3..8.799..5..3...6..2..9...1.97..6...3..9...3..4..6...9..1.35",
"....7..2.8.......6.1.2.5...9.54....8.........3....85.1...3.2.8.4.......9.7..6...."]


def solve_sudoku(game):
    
    game = game.replace('.', '0')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


def count_errors(grid):
    total_errors = 0
    
    # Fonction pour vérifier les erreurs dans une ligne, colonne ou sous-grille
    def check_errors(section):
        nonlocal total_errors
        counts = [0] * 10
        for num in section:
            counts[num] += 1
        total_errors += sum(max(count - 1, 0) for count in counts)
    
    # Vérifier les erreurs dans les lignes
    for row in grid:
        check_errors(row)
    
    # Vérifier les erreurs dans les colonnes
    for col in range(9):
        column = [grid[row][col] for row in range(9)]
        check_errors(column)
    
    # Vérifier les erreurs dans les sous-grilles de 3x3
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
            check_errors(square)
    
    return total_errors



model = keras.saving.load_model('ez.keras')
aled = []
for g in game:
    aled.append(count_errors(solve_sudoku(g)))
print(aled)
# test_accuracy(x_test[:10], y_test[:10])