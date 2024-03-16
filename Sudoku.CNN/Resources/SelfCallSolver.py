import clr
clr.AddReference("Sudoku.Shared")
clr.AddReference("Sudoku.CNN")
from Sudoku.CNN import CNNSolver
netSolver = CNNSolver()
solvedSudoku = netSolver.Solve(sudoku)