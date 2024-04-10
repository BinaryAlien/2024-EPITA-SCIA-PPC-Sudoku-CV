import copy
import keras
import numpy as np
import clr 
clr.AddReference('Sudoku.CNN')
from Sudoku.CNN import CNNSolver
from io import StringIO
from huggingface_hub import hf_hub_download

def norm(a):
    return (a/9)-.5



def denorm(a):
    return (a+.5)*9


def inference_sudoku(model, sample):

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

        #print(mask.sum())
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

def test_accuracy(model, feats, labels):

    correct = 0

    for i,feat in enumerate(feats):

        pred = complete_sudoku_one_shot(model, feat)

        true = labels[i].reshape((9,9))+1

        if(abs(true - pred).sum()==0):
            correct += 1

    print(correct/feats.shape[0])


def test_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=-1)
    true_labels = np.argmax(y_test.reshape((-1, 9, 9, 9)), axis=-1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy


def complete_sudoku_with_dynamic_filling(model, game, base_threshold=0.9, max_iterations=100):
    game = game.replace('\n', '').replace(' ', '')
    original_game = np.array([int(j) for j in game]).reshape((9, 9))
    game_normalized = norm(np.array([int(j) for j in game]).reshape((9, 9, 1)))
    game_normalized = game_normalized.reshape((1, 9, 9, 1))

    for _ in range(max_iterations):
        if not np.any(original_game == 0):  # Break if no empty cells are left
            break

        prediction = model.predict(game_normalized)
        prediction = prediction.squeeze()

        max_prob_this_iteration = -1
        cell_to_update = None
        updates_made = False

        for i in range(9):
            for j in range(9):
                if original_game[i, j] != 0:
                    continue  # Skip already filled cells

                cell_predictions = prediction[i, j]
                max_prob = np.max(cell_predictions)
                digit = np.argmax(cell_predictions) + 1  # Adjusting for zero-indexing

                if max_prob >= base_threshold:
                    original_game[i, j] = digit  # Update the cell
                    game_normalized[0, i, j, 0] = (digit / 9) - 0.5  # Keep game_normalized in sync
                    updates_made = True
                elif max_prob > max_prob_this_iteration:
                    max_prob_this_iteration = max_prob
                    cell_to_update = (i, j, digit)

        # If no cells were updated and a cell with the highest probability was found
        if not updates_made and cell_to_update is not None:
            i, j, digit = cell_to_update
            original_game[i, j] = digit
            game_normalized[0, i, j, 0] = (digit / 9) - 0.5

    return original_game


def complete_sudoku_with_dynamic_filling_non_norm(model, game, base_threshold=0.9, max_iterations=100):
    game = game.replace('\n', '').replace(' ', '')
    original_game = np.array([int(j) for j in game]).reshape((9, 9))
    game_normalized = np.array([int(j) for j in game]).reshape((9, 9, 1))
    game_normalized = game_normalized.reshape((1, 9, 9, 1))

    for _ in range(max_iterations):
        if not np.any(original_game == 0):  # Break if no empty cells are left
            break

        prediction = model.predict(game_normalized)
        prediction = prediction.squeeze()

        max_prob_this_iteration = -1
        cell_to_update = None
        updates_made = False

        for i in range(9):
            for j in range(9):
                if original_game[i, j] != 0:
                    continue  # Skip already filled cells

                cell_predictions = prediction[i, j]
                max_prob = np.max(cell_predictions)
                digit = np.argmax(cell_predictions) + 1  # Adjusting for zero-indexing

                if max_prob >= base_threshold:
                    original_game[i, j] = digit  # Update the cell
                    game_normalized[0, i, j, 0] = digit  # Keep game_normalized in sync
                    updates_made = True
                elif max_prob > max_prob_this_iteration:
                    max_prob_this_iteration = max_prob
                    cell_to_update = (i, j, digit)

        # If no cells were updated and a cell with the highest probability was found
        if not updates_made and cell_to_update is not None:
            i, j, digit = cell_to_update
            original_game[i, j] = digit
            game_normalized[0, i, j, 0] = digit

    return original_game



def complete_sudoku_one_shot(model, game):
    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = game.reshape((1, 9, 9, 1))

    prediction = model.predict(game)

    prediction = prediction.squeeze()

    completed_sudoku = np.argmax(prediction, axis=-1).reshape((9, 9)) + 1

    return completed_sudoku



def solve_sudoku(model, game):

    game = game.replace('\n', '')
    game = game.replace(' ', '')
    game = np.array([int(j) for j in game]).reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(model, game)
    return game


def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 4 == 0:
        return lr * 0.1
    return lr






def one_shot(loadingmodel, savingmodel):
    model = get_complete_model()
    if loadingmodel:
        try:
            #model = keras.models.load_model('..\\..\\..\\..\\Sudoku.CNN\\Resources\\model\\model_oneshot.keras')
            model = keras.models.load_model( hf_hub_download(
                repo_id="PPCgroup/SudokuSolver", # le nom du repo
                filename="cnn.h5", # le fichier
                revision="2385bb511ae4927c3f18d4adeffaddf7fe5ceeda", # la version donc soit commit soit branch
            ))
        except Exception as e:
            print("no model")
    else:
        csv_content = CNNSolver.GetSudokuCsvContent()
        x_train, x_test, y_train, y_test = get_data_one_shot_fill(StringIO(csv_content))
        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
        model.fit(x_train, y_train, batch_size=32, epochs=10, callbacks=[callback])
        if savingmodel:
            model.save('..\\..\\..\\..\\Sudoku.CNN\\Resources\\model\\model_oneshot.keras')
    np_instance = np.array(instance)
    instance_string = np_instance.flatten()
    instance_string = ''.join(map(str, instance_string))
    instance_string = instance_string.replace('\n', '')
    instance_string = instance_string.replace(' ', '')
    game = complete_sudoku_with_dynamic_filling_non_norm(model, instance_string)
    return np.array(list(map(int, game.flatten()))).reshape((9, 9))

def multiple(loadingmodel, savingmodel):
    model = get_model()
    if loadingmodel:
        try:
            model = keras.models.load_model( hf_hub_download(
                repo_id="PPCgroup/SudokuSolver", # le nom du repo
                filename="model.keras", # le fichier
                revision="d2e3df7498ee6ffe1b868f8c4424697d3bb14f11", # la version donc soit commit soit branch
            ))
        except Exception as e:
            print("no model")
    else:
        csv_content = CNNSolver.GetSudokuCsvContent()
        x_train, x_test, y_train, y_test = get_data(StringIO(csv_content))
        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
        model.fit(x_train, y_train, batch_size=32, epochs=2)
        if savingmodel:
            model.save('..\\..\\..\\..\\Sudoku.CNN\\Resources\\model\\model.keras')
    np_instance = np.array(instance)
    instance_string = np_instance.flatten()
    instance_string = ''.join(map(str, instance_string))
    instance_string = instance_string.replace('\n', '')
    instance_string = instance_string.replace(' ', '')
    game = solve_sudoku(model, instance_string)
    return np.array(list(map(int, game.flatten()))).reshape((9, 9))

sudokusolvingoneshot = False
result = None

if sudokusolvingoneshot:
    result = one_shot(loadingmodel=True, savingmodel=False)
else:
    result = multiple(loadingmodel=True, savingmodel=False)
