using Sudoku.Shared;

namespace Sudoku.GraphColoring.Solvers;

using Vertex = int;
using Color = int;

public class GraphGreedySolver : ISudokuSolver
{
    public SudokuGrid Solve(SudokuGrid grid)
    {
        var graph = new SudokuGraph(grid);
        Vertex? source = graph.First(SudokuGraph.Blank);
        if (!source.HasValue)
            return grid;
        GraphGreedySolver.SolveRecursive(graph, source.Value);
        return graph.ToGrid();
    }

    private static bool SolveRecursive(SudokuGraph graph, Vertex source)
    {
        foreach (Color color in graph.AvailableColors(source))
        {
            graph[source] = color;
            Vertex? next = graph.First(SudokuGraph.Blank);
            if (!next.HasValue || GraphGreedySolver.SolveRecursive(graph, next.Value))
                return true;
        }
        graph[source] = SudokuGraph.Blank;
        return false;
    }
}
