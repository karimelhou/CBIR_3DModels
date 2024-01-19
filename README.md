# Projet de Recherche de Similitude de Formes 3D

## Description
Ce projet implémente un système de recherche par similitude de formes pour des modèles 3D. Il utilise des techniques avancées pour normaliser, échantillonner, et calculer les caractéristiques des maillages 3D. L'objectif est d'identifier des objets similaires dans une base de données de modèles 3D.

## Fonctionnalités
- **Normalisation de la Pose** : Normalise la pose des modèles 3D pour une comparaison uniforme.
- **Extraction de Caractéristiques** : Extrait des moments d'inertie, distances moyennes et variances des distances pour chaque modèle.
- **Réduction de Maillage** : Implémente une méthode conceptuelle de réduction du maillage d'un modèle 3D.
- **API REST** : Permet la recherche de modèles similaires via une interface API.

## Technologies
- Python
- Trimesh
- NumPy
- Flask

## Utilisation
1. **Normalisation et Extraction de Caractéristiques** : Traite les modèles 3D et extrait des vecteurs de caractéristiques.
2. **Stockage des Caractéristiques** : Sauvegarde les caractéristiques dans un fichier CSV.
3. **API REST** : Lancez l'application Flask pour rechercher des modèles similaires.

## Auteurs
- ELHoumaini Karim


## Licence
Ce projet est sous licence MIT.
