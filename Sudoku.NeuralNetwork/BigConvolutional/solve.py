import numpy as np
import torch


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        """It only support square kernels and stride=1, dilation=1, groups=1."""
        super(Conv2dSame, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )
    def forward(self, x):
        return self.net(x)
class SudokuCNN(torch.nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(Conv2dSame(1,512,3), #1
                                         Conv2dSame(512,512,3),#2
                                         Conv2dSame(512,512,3),#3
                                         Conv2dSame(512,512,3),#4
                                         Conv2dSame(512,512,3),#5
                                         Conv2dSame(512,512,3),#6
                                         Conv2dSame(512,512,3),#7
                                         Conv2dSame(512,512,3),#8
                                         Conv2dSame(512,512,3),#9
                                         Conv2dSame(512,512,3),#10
                                         Conv2dSame(512,512,3),#11
                                         Conv2dSame(512,512,3),#12
                                         Conv2dSame(512,512,3),#13
                                         Conv2dSame(512,512,3),#14
                                         Conv2dSame(512,512,3))#15
        self.last_conv = torch.nn.Conv2d(512, 9, 1)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.last_conv(x)
        return x

def norm(x):
    return (x/9)-.5

def denorm(x):
    return (x+0.5)*9

def inference_sudoku(sample, model):

    softmax = torch.nn.Softmax(dim=0)
    feat = sample
    feat_denormed = denorm(feat)

    while(1):

        with torch.no_grad():
            out = softmax(model(feat).squeeze(0)).numpy()
            #out = out.cpu().

            pred = np.argmax(out, axis=0).reshape((9,9))+1
            prob = np.around(np.max(out, axis = 0).reshape((9,9)), 2)
            
            mask = (feat_denormed.numpy()==0).squeeze(0).squeeze(0)

            if(mask.sum()==0):
                break

            prob_new = prob*mask

            ind = np.argmax(prob_new)
            x, y = (ind//9), (ind%9)

            val = pred[x][y]
            feat_denormed[0,0,x,y] = val
            feat = norm(feat_denormed)
    return np.around(feat_denormed.squeeze(0).squeeze(0).numpy())

def solve_sudoku(model, game):
    game = torch.tensor([int(j) for j in game.flatten()]).reshape((1,1,9,9))
    game = norm(game)
    game = inference_sudoku(game, model)
    return game


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

model = SudokuCNN()
model.load_state_dict(torch.load(r"..\..\..\..\Sudoku.NeuralNetwork\BigConvolutional\model_v1_final.pt", map_location='cpu'))
model.eval()

result = solve_sudoku(model, instance)
