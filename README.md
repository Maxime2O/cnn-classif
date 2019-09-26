Partie 1 

But : Extraire les imagettes de produits des images de rayons

python Part1.py -i inputJson -d imgDir -s snippetDir

inputJson : json contenant les données annotées
imdDir : dossier dans lequel mettre les images
snippetDir : dossier dans lequel mettre les imagettes


------------------------------------
Partie 2 

But : Entrainer un réseau convolutionnel à classifier les produits (i.e. reconnaitre le produit dans l'imagette)

1. Créer les bases de données pour apprentissage et test

python Part2_CreateDb.py -i inputJson -t trainDb -v validDb -e testDb -s snippetDir

inputJson : json contenant les données annotées
trainDb : fichier contenant les annotations des imagettes pour l'entraînement
validDb : fichier contenant les annotations des imagettes pour l'entraînement (base de validation)
testDb : fichier contenant les annotations des imagettes pour le test
snippetDir : dossier contenant les imagettes


2. Entraîner les réseaux

python Part2_train.py -d dumpDir -t trainDb -v validDb

dumpDir : dossier dans lequel écrire les fichiers ".json" des architectures des réseaux ainsi que les fichiers ".h5" contenant les poids appris pendant l'entraînement
trainDb : fichier contenant les annotations des imagettes pour l'entraînement
validDb : fichier contenant les annotations des imagettes pour l'entraînement (base de validation)


3. Tester et évaluer les réseaux

python Part2_test.py -i modelsDir -o resultFile -t testDb

dossier dans lequel se trouvent les fichiers ".json" des architectures des réseaux ainsi que les fichiers ".h5" contenant les poids appris pendant l'entraînement
resultFile : fichier contenant les matrices de confusion pour chaque réseau testé
testDb : fichier contenant les annotations des imagettes pour le test

 
 ------------------------------------
 Partie 3
 
 But : Détecter les produits dans une image et les reconnaitre, cela peut être une méthode non basée sur les réseaux convolutionnels
 
 Méthode mise en place : Oriented FAST and Rotated BRIEF (ORB)
 
 python Part3 -j inputJson -d refDir -i imgDir -r resFile
 
 inputJson : json contenant les données annotées
 refDir : dossier contenant les images référence de chaque classe
 imgDir : dossier contenant les images à traiter
 resFile : fichier contenant les résultats de l'évaluation (précision/rappel pour chaque classe)
 
 Notes : 
 refDir a été créé "à la main" en trouvant des snippets qui peuvent servir de référence pour leur classe
 

 