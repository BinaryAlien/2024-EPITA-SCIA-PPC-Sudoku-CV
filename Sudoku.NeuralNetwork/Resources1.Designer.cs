﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Sudoku.NeuralNetwork {
    using System;
    
    
    /// <summary>
    ///   A strongly-typed resource class, for looking up localized strings, etc.
    /// </summary>
    // This class was auto-generated by the StronglyTypedResourceBuilder
    // class via a tool like ResGen or Visual Studio.
    // To add or remove a member, edit your .ResX file then rerun ResGen
    // with the /str option, or rebuild your VS project.
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "4.0.0.0")]
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class Resources1 {
        
        private static global::System.Resources.ResourceManager resourceMan;
        
        private static global::System.Globalization.CultureInfo resourceCulture;
        
        [global::System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal Resources1() {
        }
        
        /// <summary>
        ///   Returns the cached ResourceManager instance used by this class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Resources.ResourceManager ResourceManager {
            get {
                if (object.ReferenceEquals(resourceMan, null)) {
                    global::System.Resources.ResourceManager temp = new global::System.Resources.ResourceManager("Sudoku.NeuralNetwork.Resources1", typeof(Resources1).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }
        
        /// <summary>
        ///   Overrides the current thread's CurrentUICulture property for all
        ///   resource lookups using this strongly typed resource class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to import os
        ///
        ///import numpy as np
        ///from keras import models, utils
        ///
        ///
        ///class SudokuSolver:
        ///    def __init__(self, model_path):
        ///        self.model = self.load_model(model_path)
        ///
        ///    def load_model(self, model_path):
        ///        return models.load_model(model_path)
        ///
        ///    def __call__(self, puzzles):
        ///        puzzles = puzzles.copy()
        ///        for _ in range((puzzles == 0).sum((1, 2)).max()):
        ///            model_preds = self.model.predict(
        ///                utils.to_categorical(puzzles, num_classes=10), verbose= [rest of string was truncated]&quot;;.
        /// </summary>
        internal static string solve_ffn_py {
            get {
                return ResourceManager.GetString("solve_ffn.py", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to import torch
        ///import torch.nn as nn
        ///import numpy as np
        ///
        ///
        ///def create_constraint_mask():
        ///    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)
        ///    # row constraints
        ///    for a in range(81):
        ///        r = 9 * (a // 9)
        ///        for b in range(9):
        ///            constraint_mask[a, 0, r + b] = 1
        ///
        ///    # column constraints
        ///    for a in range(81):
        ///        c = a % 9
        ///        for b in range(9):
        ///            constraint_mask[a, 1, c + 9 * b] = 1
        ///
        ///    # box constraints
        ///    for a in range(81):
        ///      [rest of string was truncated]&quot;;.
        /// </summary>
        internal static string solve_linear_dropout_py {
            get {
                return ResourceManager.GetString("solve_linear_dropout.py", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to import torch
        ///import torch.nn as nn
        ///import numpy as np
        ///
        ///
        ///def create_constraint_mask():
        ///    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)
        ///    # row constraints
        ///    for a in range(81):
        ///        r = 9 * (a // 9)
        ///        for b in range(9):
        ///            constraint_mask[a, 0, r + b] = 1
        ///
        ///    # column constraints
        ///    for a in range(81):
        ///        c = a % 9
        ///        for b in range(9):
        ///            constraint_mask[a, 1, c + 9 * b] = 1
        ///
        ///    # box constraints
        ///    for a in range(81):
        ///      [rest of string was truncated]&quot;;.
        /// </summary>
        internal static string solve_linear_py {
            get {
                return ResourceManager.GetString("solve_linear.py", resourceCulture);
            }
        }
    }
}
