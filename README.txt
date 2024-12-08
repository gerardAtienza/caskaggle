El projecte té com a objectiu determinar si el govern dels Estats Units hauria de concedir préstecs a petites i mitjanes empreses (pimes) mitjançant l'ús de models de machine learning per predir la de devolució dels préstecs. El dataset ha sigut extret del excell de kaggles històrics i el seu link és https://www.kaggle.com/mirbektoktogaraev/should-this-loan-be-approved-or-denied

El codi es divideix en tres fitxers principals.

El fitxer main.py és el punt d'entrada del programa. Utilitza el mòdul argparse per gestionar els arguments de la línia de comandes i permet escollir entre diferents accions: entrenar (train), ajustar (tune), validar creuadament (cross_validate) o provar (test) un model. A més, permet seleccionar entre diversos models de machine learning: Decision Tree, Random Forest, Gradient Boosting i XGBoost. El codi carrega i pre-processa les dades utilitzant funcions del fitxer preprocess.py, entrena el model seleccionat, ajusta els hiperparàmetres si cal, i guarda els models entrenats.

El fitxer preprocess.py conté funcions per caregar i pre-processar les dades. Llegeix el conjunt de dades de la SBA, neteja les dades eliminant columnes irrellevants (com IDs o noms de les empreses) i amb alta correlació (per evitar data leakage), transforma les variables categòriques en variables numèriques mitjançant one-hot encoding, i normalitza les columnes numèriques. També gestiona els valors nuls i prepara les dades per ser dividides en conjunts de training i testing.

El fitxer models.py defineix la classe Model, que encapsula la lògica per ajustar, entrenar i avaluar els models de machine learning. Inclou mètodes per ajustar els hiperparàmetres amb GridSearchCV, entrenar el model amb les millors configuracions trobades, fer prediccions, validar creuadament amb diversos valors de k, provar el model amb dades de testing, i calcular mètriques d'avaluació com la precisió, el F1 score, el recall i la accuracy. A més, permet guardar i carregar models entrenats utilitzant pickle.

Per usar les funcionalitats del programa, s'ha d'executar a la terminal el progrma escrivint 'python main.py', seguit de l'acció i el model (els diferents casos son consultables al argparser del main). Finalment, una execució podria seguir la forma: 'python main.py test xgboost'.

