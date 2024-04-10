Notre implémentation du CNN pour la résolution de Sudoku comporte plusieurs modèles,
-Le premier modèle, un modèle simple qui remplit le sudoku case par case qui fait peu d'erreurs, mais qui est lent.
-Le second qui rempli les cases en fonction de la probabilité du modèle et qui réitère sur le sudoku qui est rapide mais fait plus d'erreurs.
Celui qui est lancé est le premier et il faut changer la variable sudokusolvingoneshot et la mettre à True pour lancer le second.
Les modèles sont load depuis huggingface donc il faut une connexion internet.