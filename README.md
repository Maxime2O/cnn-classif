Partie 1 

But : Extraire les imagettes de produits des images de rayons

python Part1.py -i inputJson -d imgDir -s snippetDir

inputJson : json contenant les donn�es annot�es
imdDir : dossier dans lequel mettre les images
snippetDir : dossier dans lequel mettre les imagettes


------------------------------------
Partie 2 

But : Entrainer un r�seau convolutionnel � classifier les produits (i.e. reconnaitre le produit dans l'imagette)

1. Cr�er les bases de donn�es pour apprentissage et test

python Part2_CreateDb.py -i inputJson -t trainDb -v validDb -e testDb -s snippetDir

inputJson : json contenant les donn�es annot�es
trainDb : fichier contenant les annotations des imagettes pour l'entra�nement
validDb : fichier contenant les annotations des imagettes pour l'entra�nement (base de validation)
testDb : fichier contenant les annotations des imagettes pour le test
snippetDir : dossier contenant les imagettes


2. Entra�ner les r�seaux

python Part2_train.py -d dumpDir -t trainDb -v validDb

dumpDir : dossier dans lequel �crire les fichiers ".json" des architectures des r�seaux ainsi que les fichiers ".h5" contenant les poids appris pendant l'entra�nement
trainDb : fichier contenant les annotations des imagettes pour l'entra�nement
validDb : fichier contenant les annotations des imagettes pour l'entra�nement (base de validation)


3. Tester et �valuer les r�seaux

python Part2_test.py -i modelsDir -o resultFile -t testDb

dossier dans lequel se trouvent les fichiers ".json" des architectures des r�seaux ainsi que les fichiers ".h5" contenant les poids appris pendant l'entra�nement
resultFile : fichier contenant les matrices de confusion pour chaque r�seau test�
testDb : fichier contenant les annotations des imagettes pour le test

 
 ------------------------------------
 Partie 3
 
 But : D�tecter les produits dans une image et les reconnaitre, cela peut �tre une m�thode non bas�e sur les r�seaux convolutionnels
 
 M�thode mise en place : Oriented FAST and Rotated BRIEF (ORB)
 
 python Part3 -j inputJson -d refDir -i imgDir -r resFile
 
 inputJson : json contenant les donn�es annot�es
 refDir : dossier contenant les images r�f�rence de chaque classe
 imgDir : dossier contenant les images � traiter
 resFile : fichier contenant les r�sultats de l'�valuation (pr�cision/rappel pour chaque classe)
 
 Notes : 
 refDir a �t� cr�� "� la main" en trouvant des snippets qui peuvent servir de r�f�rence pour leur classe
 

 