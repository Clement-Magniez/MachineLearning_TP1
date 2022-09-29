 <p style="text-align:left;">
  <font size = "+3"> <b>TD 1</b> </font> 
  <span style="float:right;"> 
   <font size = "+3"> MAGNIEZ - BLUM </font> 
  </span> 
</p> 




Pour ce TD, nous avons choisi d'évaluer l'influence de choix d'architectures et d'hyper-paramètres sur les performances dans un problème de régression. Les fonctions approximées sont des combinaisons linéaires de cosinus, de phase et de periodes diverses. L'échantillonage de valeurs fera partie de l'étude des performances, où l'on fera varier le bruit et la distribution.

Sauf précision, lorsqu'un paramètre changera les autres resteront constants. On utilisera un réseau feed-forward fully-connected tout au long des experiences. Les neurones seront de type <img src="images/033ad66e7dc7bc0d794635f016a1e5f2.svg?invert_in_darkmode" align=middle width=86.17380089999997pt height=24.65753399999998pt/>, et <img src="images/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode" align=middle width=9.81741584999999pt height=22.831056599999986pt/> est identique dans tout les neurones d'une même couche.

## I - Activations classiques

Dans cette partie, les fonctions d'activation f des neurones seront des fonctions "classiques": TanH, sigmoïde, ReLu.

La fonction approximée dans cette partie est :
<p align="center"><img src="images/7137ee964e0f256a9846a0bf351f4a3e.svg?invert_in_darkmode" align=middle width=202.6199241pt height=42.969917099999996pt/></p>

A cause des valeurs que prend G, sigmoide et TanH sont plutot appropriées. Ce n'est pas le cas en général pour des régressions. Les mesures pour l'entrainement ne seront pas bruitées, et uniformes dans <img src="images/9fa2a427913be55198bfe146c54995eb.svg?invert_in_darkmode" align=middle width=52.21473014999999pt height=26.76175259999998pt/>. 

Voici les résultats pour différentes tailles de couches cachées (une seule couche cachée). On constate que ReLU et TanH ont des résultats similaires, relativement médiocres. Peut-être que plusieurs couches cachées donneraient des meilleurs résultats, mais c'est beaucoup de paramètres pour une tâche d'apparence assez simple. Sigmoïde a des résultats bien inferieurs, ce qui est compréhensible en regardant les équations de la rétropropagation. 

Valeurs pour <img src="images/478d6c4eff797ea6356f6d894888d0ed.svg?invert_in_darkmode" align=middle width=112.12316279999997pt height=24.65753399999998pt/>, pour <img src="images/8436d02a042a1eec745015a5801fc1a0.svg?invert_in_darkmode" align=middle width=39.53182859999999pt height=21.18721440000001pt/> et y entre <img src="images/175ccc5874192ac2826db5f07bc0afba.svg?invert_in_darkmode" align=middle width=21.00464354999999pt height=21.18721440000001pt/> et <img src="images/5dc642f297e291cfdde8982599601d7e.svg?invert_in_darkmode" align=middle width=8.219209349999991pt height=21.18721440000001pt/> (comme dans le corpus d'entrainement):

# image #

Dynamiques de convergence:

![Figure 1](./images/Figure_1.png)
![Figure 2](./images/Figure_2.png)

# image #

## II - Activations sinusoïdales

Dans cette partie, on choisit la fonction d'activation sinus pour la couche de sortie. Elle n'est pas disponible par défault dans pytorch, il faut donc l'implémenter en sous-classant nn.Module. Sin est cependant présent dans la librairie (torch.sin), donc il n'est pas nécessaire de réimplémenter la propagation du gradient à travers l'activation.

```
class CustomSine(nn.Module):
    def forward(self, z):
        return torch.sin(z)
```

Voici les résultats obtenus, avec l'architecture suivante:
<p align="center"><img src="images/5fdeafeae14e06ddb014840a1fc4e54e.svg?invert_in_darkmode" align=middle width=308.01073214999997pt height=49.315569599999996pt/></p>
  
Le réseau peut converger parfaitement à partir de X (nb de neurones cachés) = 2, mais le résultat dépendra de l'initialisation. Pour X=5, les résultats sont robustes. On pourrait utiliser un learning rate plus élevé pour une convergence plus rapide, mais c'était sans importance ici.

X=2, convergence:
![Figure 3](./images/Figure_3_conv.png)
X=2, exmple de mauvaise convergence:
![Figure 3](./images/Figure_3_div.png)
X=5
![Figure 3](./images/Figure_3.png)

## Conclusion 

On aurait pu mettre l' activation sinus en entrée, ou choisir une architecture différente. L'important ici est que selon le problème, avoir directement des activations appropriées peut apporter énormément aux performances d'un réseau. Par exemple, un réseau feedforward utilisé pour une tâche de reinforcement learning incluant des déplacements dans l'espace gagne beaucoup à implémenter des fonctions type cosinus, racine carrée, multiplication, division (ou passage à l'inverse). L'implémentation est cependant plus complexe, puisque l'entrée des neurones n'est plus nécessairement le produit matrice vecteur des poids par les activations.





