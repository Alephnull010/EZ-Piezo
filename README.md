# PiezoKriging — Plugin QGIS

Plugin QGIS pour générer des **cartes piézométriques** (courbes isopièzes) par interpolation **Ordinary Kriging**, à partir de mesures ponctuelles sur ouvrages (piézomètres, forages, puits).

> Cible : hydrogéologues travaillant en France (défaut Lambert 93 — EPSG:2154), compatible avec tout système projeté.

---

## Fonctionnalités

### Données
- Import CSV avec détection automatique du séparateur (`;` `,` `TAB` `|`)
- Saisie et édition manuelle dans le tableau
- Détection et blocage des points aux coordonnées identiques (doublons rendant la matrice de krigeage singulière)
- Validation du système de coordonnées : les CRS géographiques en degrés (ex. EPSG:4326) sont refusés — le krigeage euclidien nécessite un système projeté

### Variogramme
- Variogramme expérimental omni-directionnel ou directionnel (azimut + tolérance angulaire)
- 4 modèles théoriques : **sphérique**, **exponentiel**, **gaussien**, **linéaire**
- Ajustement par moindres carrés pondérés (pondération de Cressie 1985 : σᵢ ∝ 1/√N(hᵢ), favorisant les petits lags)
- Contrainte nugget ≤ sill garantie par reparamétrisation en (nugget, palier partiel, portée)
- Surcharge manuelle des paramètres (nugget, sill, range) sans recalcul automatique
- Graphique expérimental + modèle ajusté avec effectifs N(h) par classe

### Interpolation
- Ordinary Kriging vectorisé par blocs (4 096 nœuds/bloc) : mémoire bornée quelle que soit la résolution
- Ellipse de recherche configurable (rayon majeur/mineur, orientation, min/max voisins) avec solve batché par ensemble de voisins unique
- Détection des résultats non finis après `lu_solve` et re-factorisation avec jitter automatique

### Sorties (4 couches QGIS)
| Couche | Format | Description |
|---|---|---|
| Piézométrie — Kriging | GeoTIFF | Carte des niveaux interpolés, rampe bleue→rouge |
| Incertitude — Kriging (σ) | GeoTIFF | Écart-type de krigeage, rampe jaune→bordeaux |
| Isopièzes | GPKG | Courbes de niveau vectorielles avec champ ELEV, étiquettes optionnelles |
| Points ouvrages | Couche mémoire | Points d'entrée avec attributs ouvrage/x/y/z_ngf |

- Masque NoData hors enveloppe convexe des données (option)
- Intervalle isopièzes automatique (règle des nombres ronds) ou manuel
- `gdal.ContourGenerate` appelé avec `useNoData=1, noDataValue=-9999` : pas de contours parasites en bordure

### Validation croisée (LOO)
- Leave-One-Out avec les mêmes paramètres de voisinage que la carte finale
- Statistiques de calage du variogramme :
  - **Erreur moyenne** (cible ≈ 0) — non-biais
  - **RMSE** — erreur absolue
  - **Erreur standardisée moyenne** = (mesuré − estimé) / σₖ (cible ≈ 0)
  - **RMSSE** = RMS des erreurs standardisées (cible ≈ 1 ; < 1 = incertitude surestimée, > 1 = sous-estimée)
- Tableau par ouvrage avec colonne d'erreur standardisée (rouge si |err. std.| > 2)
- Graphique mesuré vs estimé avec droite best-fit
- Gestion robuste des points non résolus par l'ellipse (exclus des stats, affichés « — » dans la table)

---

## Installation

### Prérequis

**QGIS 3.16+** — numpy, scipy, matplotlib, GDAL sont inclus dans QGIS, aucune installation pip nécessaire.

### Copie directe

```
Copiez le dossier du plugin dans :

Windows : C:\Users\<user>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\piezo_kriging\
Linux   : ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/piezo_kriging/
macOS   : ~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/piezo_kriging/
```

Puis : **Extensions → Gérer les extensions → Installées → cocher PiezoKriging**.

### Depuis un ZIP

**Extensions → Gérer les extensions → Installer depuis un ZIP** → sélectionnez l'archive.

### Rechargement sans redémarrer QGIS

Après modification des fichiers Python : icône de rechargement dans le gestionnaire d'extensions, ou plugin *Plugin Reloader*.

---

## Utilisation rapide

### Données de test

Le fichier `exemple_donnees.csv` (12 points Lambert 93, cotes NGF) permet de tester le plugin immédiatement.

### Étapes

1. **Onglet Données** — Charger le CSV (séparateur `;`) ou saisir manuellement
2. **Onglet Paramètres** — Choisir le modèle de variogramme, la résolution de grille (100 px = bon compromis), l'intervalle isopièzes, le code EPSG (système projeté obligatoire)
3. **▶ Lancer le Kriging** — 4 couches apparaissent dans QGIS
4. **Validation croisée** — Vérifier que RMSSE ≈ 1 ; ajuster le variogramme si nécessaire

### Format CSV

```csv
Ouvrage;X;Y;Z
PZ01;843250.5;6518320.1;45.32
PZ02;843480.2;6518150.7;43.18
```

| Colonne | Description |
|---|---|
| Ouvrage | Identifiant (texte libre) |
| X, Y | Coordonnées dans un système **projeté** (Lambert 93, UTM…) |
| Z | Niveau piézométrique en mètres (NGF ou tout autre référence altimétrique) |

Séparateurs acceptés : `;` `,` `TAB` `|` — détection automatique à l'import.

---

## Détails techniques

### Moteur de krigeage (`kriging_engine.py`)

Module Python pur (sans dépendance QGIS), importable et testable indépendamment.

**Pipeline** :

```
compute_experimental_variogram
    → fit_variogram (curve_fit, pondération Cressie)
    → ordinary_kriging (système de Lagrange n+1, scipy.linalg.lu_factor/lu_solve)
```

**Ajustement du variogramme — pondération de Cressie (1985)**

```python
sigma = 1 / sqrt(N(h))   # σᵢ inversement proportionnel à l'effectif du lag
```

Les lags riches en paires pèsent plus ; les lags extrêmes (souvent basés sur peu de paires) sont naturellement atténués.

**Reparamétrisation (nugget, psill, range)**

L'ajustement porte sur le *palier partiel* `psill = sill − nugget ≥ 0`, ce qui garantit `sill ≥ nugget` par construction et évite les variogrammes décroissants non physiques.

**Modèles disponibles**

| Modèle | γ(h) |
|---|---|
| Sphérique | C₀ + C·[1.5(h/a) − 0.5(h/a)³] pour h ≤ a, sinon C₀+C |
| Exponentiel | C₀ + C·[1 − exp(−3h/a)] |
| Gaussien | C₀ + C·[1 − exp(−3(h/a)²)] |
| Linéaire | C₀ + pente·h |

C₀ = nugget, C = palier partiel, a = portée.

**Performances**

- Branche globale : K factorisé une fois (`lu_factor`), grille traitée en blocs de 4 096 nœuds → mémoire pic = O(n × 4 096), indépendante de la résolution totale
- Branche ellipse : appartenance à l'ellipse vectorisée par bloc `(blk × n)` ; regroupement des nœuds partageant le même voisinage → une seule factorisation K par groupe unique, RHS batché

### Architecture des fichiers

```
piezo_kriging/
├── kriging_engine.py   # Moteur géostatistique pur (numpy/scipy), sans QGIS
├── piezo_dialog.py     # Interface Qt5 multi-onglets (données, paramètres, variogramme, LOO)
├── piezo_kriging.py    # Intégration QGIS : création des couches, style, orchestration
├── exemple_donnees.csv # Jeu de test (12 points Lambert 93)
├── icons/icon.png
└── metadata.txt
```

### Limitations connues

- Calcul synchrone : les grandes grilles (> 500 × 500) bloquent le thread UI de QGIS ; découper en sous-zones si nécessaire
- Ordinary Kriging uniquement (pas de krigeage universel ni de co-krigeage)
- Pas de dérive (modèle stationnaire)
- Coordonnées en système projeté obligatoires — les degrés décimaux (EPSG:4326) sont détectés et refusés

---

## Licence

MIT — libre d'utilisation, de modification et de redistribution.

---

*Développé par Hugo LEBEL.*
