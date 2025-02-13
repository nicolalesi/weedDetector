 Folder: MetricheModelli 
-------------------------------------------------
All'interno di questa folder è possibile trovare le metriche ottenute dal test dei modelli, possono essere diverse per lo stesso modello in base agli iper parametri usati,
ad esempio "KNNMetriche_euclidean.txt" sono le metriche derivanti dal modello KNN addestrato utilizzando la distanza euclidea.
Sono utili per il confronto tra modelli.

 Folder: codiceModelli 
 ------------------------------------
All'interno di questa folder sono presenti gli script Python che vanno ad addestrare i diversi modelli, tutti con l'obiettivo di stabilire se una pianta è infestante o meno:
- pianteInfestantiCNN.PY   --> Addestra il modello convoluzionale
- pianteInfestantiKNN.PY   --> Addestra il modello K nearest neighbour
- pianteInfestantiRFM.PY   --> Addestra il modello random forest
- pianteInfestantiSVM.PY   --> Addestra il modello support vector machine

 Folder: opDataset 
 ------------------------------------
In questa folder sono presenti script utili all'analisi del dataset e relative immaigini, ad esempio lo script "isBalanced.py", è utile per capire
se le categorie all'interno del dataset sono bilanciate.

Folder: modelliGenerati 
----------------------------------------
Al running del codice viene generata questa cartella che conterrà tutti i modelli generati dal codice

Folder : loadModelEx
---------------------------------- 
Un esempio di applicativo visivo in cui è possibile caricare il modello che si vuole utilizzare come riferimento e l'immagine che si vuole stabilire essere
oppure no una pianta infestante.
- loadModel.py supporta i modelli con estensione .h5 (CNN)
- loadModelPKL.py supporta i modelli con estensione .pkl (KNN, SVM ...) che non accettano in input delle semplici foto, ma è necessario estrarre gli HOG per la predizione
