---
title: "Auditer une prévision de variance sur futures : quand un décalage change l'expérience"
description: "Un pipeline contrat par contrat pour ES, CL et GC, puis l'audit temporel qui distingue une prévision de volatilité plausible d'un backtest irréalisable."
date: 2026-07-13
image: images/cover-futures-roll.png
categories: ["Quantitative Research", "Futures", "Risk Management"]
---

# Auditer une prévision de variance sur futures : quand un décalage change l'expérience

La question de départ semblait raisonnable : la forme de la courbe des futures aide-t-elle à prévoir la variance des deux prochaines séances ? Le projet réunissait les bons éléments : contrats continus, variables de structure par terme, benchmark de persistance, régression ridge et tests walk-forward sur l'E-mini S&P 500 (ES), le pétrole brut (CL) et l'or (GC).

J'ai ensuite suivi la cible ligne par ligne.

Cet audit a révélé trois problèmes de calendrier. La cible de variance future était décalée d'une séance, le benchmark de persistance utilisait une cible avant qu'elle ne soit observable, et la dernière cible d'entraînement de chaque fold débordait sur la période de test. J'ai corrigé les trois dans le pipeline de production et ajouté de petits tests, calculés à la main, pour verrouiller l'horloge de prévision.

J'ai conservé le reste du pipeline et relancé la comparaison hors échantillon depuis les contrats bruts. Le résultat négatif est justement ce qui mérite d'être retenu : une fois le calendrier réparé, aucun des deux modèles n'explique la variance future de façon fiable.

## Une série de futures a des coutures

Un contrat future expire. Une année de prix du « pétrole brut » est donc une succession de contrats individuels, et non l'historique continu d'un titre permanent. Leur raccordement produit une **série continue de futures** : un historique synthétique qui choisit un contrat actif à chaque date et ajuste les observations anciennes lors du changement de contrat.

Par défaut, le projet suit un calendrier de roll explicite. À la date de roll, il quitte l'ancien contrat front pour le nouveau. Avec l'ajustement par ratio, il divise le nouveau cours de clôture par le précédent, puis multiplie tous les prix open, high, low et close antérieurs par ce facteur. Le calcul remonte depuis le roll le plus récent. Il préserve ainsi les rendements proportionnels au sein de chaque segment historique tout en effaçant la rupture de niveau au raccord.

La série ajustée ne sert ici qu'au calcul des rendements quotidiens. Les variables de structure par terme proviennent toujours des prix bruts simultanés des contrats. Cette séparation est essentielle : un niveau rétroajusté convient à un historique de rendements, mais il ne décrit pas la courbe cotée ce jour-là.

Le code conserve aussi les prix non ajustés, l'identité du contrat actif et les dates de roll. Ce ne sont pas des sorties accessoires. Sans elles, il devient presque impossible d'expliquer un rendement suspect près d'une échéance.

## Transformer la courbe en variables

À chaque date, le pipeline classe les contrats par échéance et retient les trois plus proches. Notons $F_{1,t}$ le cours du contrat front à la date $t$, $F_{2,t}$ celui du deuxième contrat, et $d_t$ l'écart positif, en jours calendaires, entre leurs dates de fin configurées. Le rendement de roll annualisé $y_t$ vaut

$$
y_t = \left(\frac{F_{1,t}-F_{2,t}}{F_{2,t}}\right)\frac{365}{d_t}.
$$

Le code enregistre également le spread brut $F_{2,t}-F_{1,t}$ et ajuste une droite aux prix des contrats, normalisés par le prix front. La pente résume la forme des trois premières échéances.

Lorsque $F_{2,t}>F_{1,t}$, la courbe est en **contango** : la livraison lointaine coûte plus cher que la livraison proche, et cette définition du rendement de roll donne une valeur négative. Lorsque $F_{2,t}<F_{1,t}$, la courbe est en **backwardation**, avec un rendement de roll positif. Une bande annualisée de $\pm0.5\%$ classe les petites valeurs dans un régime plat. Les trois régimes deviennent des variables indicatrices dans le modèle linéaire.

Le modèle reçoit aussi le rendement logarithmique et la variance de la séance précédente. Il ne reçoit pas la variance quotidienne courante. Cette précision compte dès que la cible est datée correctement.

## Définir l'horloge avant la formule

Supposons que la prévision soit calculée après la clôture de la date $t$. Notons $P_t$ le cours continu ajusté et définissons le rendement logarithmique quotidien $r_t$ par

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right).
$$

Notons $H$ l'horizon de prévision en séances et $A$ le facteur d'annualisation, en séances par an. Le projet utilise $H=2$ et $A=252$. La variance réalisée future annualisée recherchée est

$$
RV^{ann}_{t,t+H}=\frac{A}{H}\sum_{i=1}^{H}r_{t+i}^{2}.
$$

L'indice est décisif : le premier rendement au carré doit être $r_{t+1}^2$, et non $r_t^2$.

L'implémentation initiale avançait la série avant d'appliquer une somme glissante :

```python
forward_sum = (
    squared_returns.shift(-1)
    .rolling(window=horizon_days, min_periods=horizon_days)
    .sum()
)
```

Or, une fenêtre glissante pandas regarde vers le passé. Avec $H=2$, la valeur attachée à la date $t$ devient $r_t^2+r_{t+1}^2$. Sur une petite série $[a,b,c,d]$ de rendements au carré, `shift(-1)` produit $[b,c,d,NaN]$, puis la fenêtre de deux lignes donne $[NaN,b+c,c+d,NaN]$. La cible de la deuxième ligne contient donc le rendement de cette même ligne.

La cible de production construit maintenant chaque avance explicitement :

```python
future_squared_returns = [
    squared_returns.shift(-lead)
    for lead in range(1, horizon_days + 1)
]
forward_sum = pd.concat(future_squared_returns, axis=1).sum(
    axis=1,
    min_count=horizon_days,
)
```

La ligne datée $t$ contient désormais exactement $r_{t+1}^2+r_{t+2}^2$.

## Un split walk-forward peut encore regarder vers l'avenir

Des cibles correctes ne suffisent pas : il faut aussi respecter leur date de **disponibilité**. Une cible à deux séances attachée à la date $s$ n'est entièrement connue qu'après la clôture de $s+2$. Si un fold de test commence à la date $T$, une cible d'entraînement n'est observable à cet instant que si $s+H\leq T$, soit $s\leq T-H$.

Un split walk-forward ordinaire termine l'entraînement à $T-1$. Avec $H=2$, sa dernière cible utilise les rendements de $T$ et $T+1$; le second n'existe pas encore au moment de prévoir à $T$. L'évaluateur termine désormais l'entraînement à $T-H$. Il omet ainsi $H-1=1$ ligne nominale entre l'échantillon ajusté et le bloc de test. Cette omission est une **purge** : elle empêche les cibles d'apprentissage d'employer une information indisponible à l'origine de la prévision.

```python
train_end = test_start - label_horizon_days
train = ordered.iloc[train_start : train_end + 1]
test = ordered.iloc[test_start : test_end + 1]
```

Le benchmark de persistance initial souffre du même défaut. Décaler une cible future d'une ligne ne la rend pas observable. Pour obtenir un benchmark réalisable, définissons la variance passée connue $K_t$ à partir du rendement courant et des $H-1$ rendements précédents :

$$
K_t=\frac{A}{H}\sum_{i=0}^{H-1}r_{t-i}^{2}.
$$

Après la clôture de $t$, tous les termes de $K_t$ sont connus. La prévision de persistance corrigée utilise $K_t$ comme estimation de $RV^{ann}_{t,t+H}$. Les variables de ridge restent décalées d'une séance : cette fenêtre courante alimente le benchmark, pas ridge de façon cachée.

La régression ridge travaille sur des variables standardisées. Notons $n$ le nombre d'observations d'entraînement, $y_i$ la cible de variance future de l'observation $i$, $\mathbf{x}_i$ son vecteur de variables standardisées, $b$ la constante non pénalisée, $\boldsymbol{\beta}$ le vecteur de coefficients, et $\lambda$ l'intensité de la pénalisation. Les paramètres minimisent

$$
\sum_{i=1}^{n}\left(y_i-b-\mathbf{x}_i^\top\boldsymbol{\beta}\right)^2
+\lambda\lVert\boldsymbol{\beta}\rVert_2^2.
$$

Ici, $\lambda=1$. La moyenne et l'écart-type de chaque variable sont estimés sur le fold d'entraînement, puis appliqués au bloc de test suivant. Cette partie du pipeline respecte bien la séparation temporelle.

## Ce qui reste après la correction

L'analyse corrigée en production utilise les données locales de 2024, le roll calendaire, l'ajustement par ratio, une frontière initiale nominale de 80 observations, des blocs de test de 20 observations et un pas de 20. Comme $H=2$, le premier échantillon ajusté de ridge contient 79 labels observables; la ligne qui précède immédiatement le bloc de test est purgée. Chaque actif fournit 11 folds de test et 220 prévisions hors échantillon par modèle. Le graphique présente la moyenne du root mean squared error (RMSE), ou racine de l'erreur quadratique moyenne, sur ces folds. Le RMSE est exprimé en unités décimales de variance annualisée, pas en points de volatilité.

![RMSE moyen par racine de future et par modèle](images/01_asset_rmse.png)

Aucun modèle ne gagne partout. La persistance domine nettement sur ES, tandis que ridge obtient un RMSE inférieur sur CL et GC. Moyenné sur toutes les combinaisons actif-fold, le RMSE de ridge atteint $0.0453$, contre $0.0481$ pour la persistance. Cet avantage étroit ne se retrouve donc pas sur tous les marchés.

| Actif | RMSE persistance | RMSE ridge | $R^2$ moyen persistance | $R^2$ moyen ridge |
|---|---:|---:|---:|---:|
| ES | 0.0257 | 0.0420 | -1.426 | -180.244 |
| CL | 0.0832 | 0.0674 | -1.280 | -0.681 |
| GC | 0.0354 | 0.0266 | -1.470 | -0.654 |

Pour interpréter $R^2$, notons $m$ le nombre d'observations d'un fold de test, $y_i$ une cible observée, $\hat y_i$ sa prévision, et $\bar y$ la moyenne des cibles observées dans le fold. Alors

$$
R^2=1-\frac{\sum_{i=1}^{m}(y_i-\hat y_i)^2}{\sum_{i=1}^{m}(y_i-\bar y)^2}.
$$

Une valeur négative signifie que le modèle fait moins bien que la moyenne constante du fold. Toutes les combinaisons actif-modèle ont un $R^2$ moyen négatif. La valeur extrêmement négative de ridge sur ES rappelle aussi qu'une moyenne de $R^2$ sur de courts folds à faible variance est fragile : lorsque le dénominateur est minuscule, quelques mauvaises prévisions écrasent le ratio. L'échec est bien réel, mais son ampleur n'est pas une mesure stable de l'importance économique.

![Variance prédite et réalisée hors échantillon](images/02_prediction_diagnostics.png)

Le nuage de points montre le problème de fond. Les deux méthodes ratent les plus fortes observations de variance réalisée. La persistance prolonge parfois un pic récent alors que les deux séances suivantes sont calmes. Ridge concentre la plupart de ses prévisions dans une bande étroite et produit même des variances négatives pour ES. Une régression linéaire sans contrainte autorise ce résultat, mais une variance inférieure à zéro n'a aucun sens.

## Ce que l'expérience permet de conclure

Les résultats corrigés ne permettent pas d'affirmer que ces variables de structure par terme prévoient correctement la variance à deux séances dans cet échantillon. Le léger avantage moyen de ridge vient de CL et GC, disparaît sur ES et s'accompagne partout d'un $R^2$ négatif. Il serait beaucoup trop tôt pour bâtir un overlay de trading sur cette base.

L'expérience reste instructive. Une étude sur futures doit gérer deux horloges : celle des données de marché et celle où la cible devient observable. Trier les dates avant de créer les splits ne suffit pas à éliminer le lookahead bias. Une cible future exige une purge adaptée à son horizon, et un benchmark doit être calculable avec l'information réellement disponible à l'origine de la prévision.

Les limites plus classiques demeurent. L'échantillon ne couvre qu'une année civile et trois marchés. Une cible à deux séances est très bruitée. La règle de roll et la méthode d'ajustement restent fixes. Ridge est un modèle linéaire non contraint, alors que la variance réalisée est positive et fortement asymétrique à droite. Une prochaine expérience plus solide prévoirait le logarithme de la variance, ajouterait un benchmark de variance exponentiellement pondérée, réglerait la pénalisation à l'intérieur de chaque fold d'entraînement et répéterait l'analyse sur plusieurs années et plusieurs horizons.

Les tests de production calculent maintenant une cible de quatre lignes à la main et contrôlent la frontière des labels pour chaque fold généré. Le choix est volontairement simple. En recherche temporelle, un petit test d'horloge protège souvent mieux le capital qu'un modèle supplémentaire.

## Références

- Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003), [“Modeling and Forecasting Realized Volatility”](https://doi.org/10.1111/1468-0262.00418), *Econometrica*, 71(2), 579–625.
- Hoerl, A. E., and Kennard, R. W. (1970), [“Ridge Regression: Biased Estimation for Nonorthogonal Problems”](https://doi.org/10.1080/00401706.1970.10488634), *Technometrics*, 12(1), 55–67.
- pandas development team, [`Series.rolling` API reference](https://pandas.pydata.org/docs/reference/api/pandas.Series.rolling.html), pour la sémantique des fenêtres rétrospectives à l'origine du défaut d'alignement.
- López de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, Chapter 7, sur la purge des observations dont les labels chevauchent l'intervalle de test.
