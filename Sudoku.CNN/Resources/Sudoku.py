import copy
import keras
import numpy as np
import clr 
clr.AddReference('Sudoku.CNN')
from Sudoku.CNN import CNNSolver
from io import StringIO

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

        print(mask.sum())
        if(mask.sum()==0):
            break

        prob_new = prob*mask
        if np.all(prob_new == 0):
            print("Warning: No confident prediction to fill any cell. Check model or input.")
            break  # Exit the loop if stuck


        #print(mask)
        ind = np.argmax(prob_new)
        #print("IND:", ind)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
        #print("FEAT:", feat)
    return pred

def test_accuracy(feats, labels):

    correct = 0

    for i,feat in enumerate(feats):

        pred = inference_sudoku(feat)

        true = labels[i].reshape((9,9))+1

        if(abs(true - pred).sum()==0):
            correct += 1

    print(correct/feats.shape[0])


#test_accuracy(x_test[:100], y_test[:100])

def solve_sudoku(game):

    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game



csv_content = CNNSolver.GetSudokuCsvContent()

model = get_model()


if False:
    try:
        model = keras.models.load_model('..\\..\\..\\..\\Sudoku.CNN\\Resources\\model\\model.keras')
    except Exception as e:
        print("no model")
else:
    x_train, x_test, y_train, y_test = get_data(StringIO(csv_content))
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
    model.fit(x_train, y_train, batch_size=32, epochs=2)
    if True:
        model.save('..\\..\\..\\..\\Sudoku.CNN\\Resources\\model\\model.keras')


np_instance = np.array(instance)
instance_string = np_instance.flatten()
instance_string = ''.join(map(str, instance_string))
instance_string = instance_string.replace('\n', '')
instance_string = instance_string.replace(' ', '')


game = solve_sudoku(instance_string)

#print('solved puzzle:\n')
#print(game)


#model.save("C:\\Users\\rokra\\RiderProjects\\2024-EPITA-SCIA-PPC-Sudoku-CV\\Sudoku.CNN\\Resources\\model.keras")

#np.sum(game, axis=1)

result = np.array(list(map(int, game.flatten()))).reshape((9, 9))


