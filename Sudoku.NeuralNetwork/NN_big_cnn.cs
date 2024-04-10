using Python.Runtime;
using Sudoku.Shared;

namespace Sudoku.NeuralNetwork;

public class NN_big_cnn : PythonSolverBase
{
    
    public override SudokuGrid Solve(SudokuGrid s)
    {
        using (PyModule scope = Py.CreateScope())
        {

            // Injectez le script de conversion
            AddNumpyConverterScript(scope);

            // Convertissez le tableau .NET en tableau NumPy
            var pyCells = AsNumpyArray(s.Cells, scope);

            // create a Python variable "instance"
            scope.Set("instance", pyCells);

            // run the Python script
            string code = Resources1.solve_big_cnn_py;
            scope.Exec(code);

            PyObject result = scope.Get("result");

            // Convertissez le résultat NumPy en tableau .NET
            var managedResult = AsManagedArray(scope, result);

            // Console.WriteLine(result);

            return new SudokuGrid() { Cells = managedResult };
        }

        return s;
    }

    protected override void InitializePythonComponents()
    {
        InstallPipModule("numpy");
        InstallPipModule("torch");
        InstallPipModule("huggingface_hub");
        
        base.InitializePythonComponents();
    }
}