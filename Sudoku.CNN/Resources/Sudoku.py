import copy
import keras
import numpy as np
#from model import get_model
#from scripts.data_processes import get_data

import clr 
clr.AddReference('Sudoku.CNN')

from Sudoku.CNN import CNNSolver

from io import StringIO

csv_content = CNNSolver.GetSudokuCsvContent()

x_train, x_test, y_train, y_test = get_data(StringIO(csv_content))


#import os

#print(os.getcwd())

#x_train, x_test, y_train, y_test = get_data(Resouces.datas)

#print(x_train)


model = get_model()

adam = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)

model.fit(x_train, y_train, batch_size=32, epochs=2)

#model.save('model/sudoku.model')
#model = keras.models.load_model('model/sudoku.model')


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


def solve_sudoku(game):

    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    return game


result = np.zeros((9,9))

