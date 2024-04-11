import copy
import keras
import numpy as np
import os



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


path = r"..\..\..\..\Sudoku.NeuralNetwork\Convolutional\ConvolutionalModel.keras"
if (not os.path.isfile(path)):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="Bl4nc/ppc_model", filename="ConvolutionalModel.keras", revision="70be9179cc5c2efa84ce6e29e8ded6ab31eae0fd", local_dir=".")
model = keras.saving.load_model(path)
instance = norm(instance)
result = inference_sudoku(instance).astype(np.int32)