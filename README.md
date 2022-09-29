# TD1 #
### - MAGNIEZ - BLUM - ###


Pour ce TD, nous avons choisi d'évaluer l'influence de choix d'architectures et d'hyper-paramètres réels sur les performances dans un problème de régression. Les fonctions approximées sont des combinaisons linéaires de cos, de phase et de periodes diverses. L'échantillonage de valeurs fera partie de l'étude des performances, où l'on fera varier le bruit et la distribution.

Sauf précision, lorsqu'un paramètre changera les autres resteront constants. On utilisera un réseau feed-forward fully-connected tout au long des experiences. Les neurones seront de type f(b + Σw*a), et f est identique dans tout les neurones d'une même couche.

## I - Activations classiques

Dans cette partie, les fonctions d'activation f des neurones seront des fonctions "classiques": tanh, sigmoïde, relu.

La fonction approximée dans cette partie est G : R² -> R, G(x,y) = cos(x) + sin(y). A cause des valeurs que prend G, sigmoide et tanh sont plutot appropriées. Ce n'est pas le cas en général pour des régressions. Les mesures pour l'entrainement ne seront pas bruitées, et uniformes dans [-3,3]². 

Voici les résultats pour différentes tailles de couches cachées (une seule couche cachée).


# image #



