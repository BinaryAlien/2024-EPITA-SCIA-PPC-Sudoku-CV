using System;
using System.Collections.Generic;
using System.Linq;
using System.Resources;
using System.Text;
using System.Threading.Tasks;
using Python.Runtime;
using Sudoku.Shared;
using Python.Runtime;

namespace Sudoku.CNN
{
    public abstract class CNNSolver : PythonSolverBase
    {
        public CNNSolver()
        {
            _method = GetMethod();
        }
        private readonly bool _method;

        protected abstract bool GetMethod();

        public override Shared.SudokuGrid Solve(Shared.SudokuGrid s)
        {
            using (PyModule scope = Py.CreateScope())
            {

                // Injectez le script de conversion
                AddNumpyConverterScript(scope);

                // Convertissez le tableau .NET en tableau NumPy
                var pyCells = AsNumpyArray(s.Cells, scope);

                // create a Python variable "instance"
                scope.Set("instance", pyCells);
                scope.Set("sudokusolvingoneshot", _method);

                // run the Python script
				
                var codes = new List<string> { Resources.data_processes, Resources.model, Resources.Sudoku_py };

                codes.ForEach(code => scope.Exec(code));

                PyObject result = scope.Get("result");

                // Convertissez le résultat NumPy en tableau .NET
                var managedResult = AsManagedArray(scope, result);

                return new SudokuGrid() { Cells = managedResult };
            }
        }
        
        public static string GetSudokuCsvContent()
        {
            // Assuming you have a resource named "datas" that contains your CSV data
            return Resources.sudoku;
        }


        protected override void InitializePythonComponents()
        {
            //declare your pip packages here
            InstallPipModule("numpy");
            InstallPipModule("scikit-learn");
            InstallPipModule("pandas");
            InstallPipModule("keras");
            InstallPipModule("tensorflow");
            InstallPipModule("huggingface_hub");
            //InstallPipModule("copy");
            base.InitializePythonComponents();
        }
    }
}
