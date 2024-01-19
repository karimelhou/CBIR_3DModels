import os
import glob
import csv
import trimesh
import numpy as np
from scipy.spatial import distance

def load_3d_model(file_path):
    """
    Charge un modèle 3D à partir d'un chemin de fichier donné.
    :param file_path: Chemin du fichier du modèle 3D.
    :return: Objet trimesh.
    """
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

def normalize_pose(mesh):
    """
    Normalise la pose du modèle 3D.
    :param mesh: Objet trimesh.
    :return: Objet trimesh normalisé.
    """
    # Calcule le centroïde (centre de masse)
    centroid = mesh.centroid

    # Translate le maillage pour le centrer à l'origine
    mesh.apply_translation(-centroid)

    # Calcule les axes principaux d'inertie
    principal_axes = mesh.principal_inertia_vectors

    # Crée une matrice de transformation 4x4
    align_matrix = np.eye(4)  # Commence avec une matrice identité
    align_matrix[:3, :3] = np.linalg.inv(principal_axes)

    # Applique la matrice de transformation au maillage
    mesh.apply_transform(align_matrix)

    dimensions_boîte_englobante = mesh.bounds[1] - mesh.bounds[0]
    facteur_échelle = 1 / np.max(dimensions_boîte_englobante)
    mesh.apply_scale(facteur_échelle)
    return mesh



def sample_surface(mesh, num_samples=1000):
    """
    Échantillonne des points sur la surface du maillage en utilisant une approche Monte-Carlo.
    :param mesh: Le maillage 3D.
    :param num_samples: Nombre de points à échantillonner.
    :return: Points échantillonnés sur la surface du maillage.
    """
    return mesh.sample(num_samples)

def compute_moments_monte_carlo(mesh, axis, num_samples=1000):
    """
    Calcule le moment d'inertie autour d'un axe donné en utilisant l'échantillonnage Monte Carlo.
    :param mesh: Objet trimesh normalisé.
    :param axis: Axe le long duquel calculer les moments.
    :param num_samples: Nombre de points à échantillonner pour l'approche Monte Carlo.
    :return: Vecteur de moment d'inertie.
    """
    # Échantillonne des points sur la surface
    points = sample_surface(mesh, num_samples)

    # Calcule la projection de chaque point sur l'axe
    projections = np.dot(points - mesh.bounds[0], axis)

    # Calcule les distances de l'axe à chaque point
    distances = np.linalg.norm(np.cross(points, axis), axis=1) / np.linalg.norm(axis)

    # Calcule le moment d'inertie pour chaque point et les somme
    moments = np.sum(distances ** 2)

    # Regroupe les moments dans un tableau pour assurer qu'il soit unidimensionnel
    moments = np.array([np.sum(distances ** 2)])
    return moments



def compute_average_distances(mesh, axis, num_slabs):
    """
    Calcule la distance moyenne aux surfaces depuis un axe donné.
    :param mesh: Objet trimesh normalisé.
    :param axis: Axe le long duquel calculer les distances moyennes.
    :param num_slabs: Nombre de tranches pour la discrétisation.
    :return: Vecteur de distance moyenne.
    """
    # Obtient les limites du maillage le long de l'axe
    bounds = mesh.bounds
    min_bound, max_bound = np.min(bounds @ axis), np.max(bounds @ axis)

    # Calculate the thickness of each slab
    slab_thickness = (max_bound - min_bound) / num_slabs

    # Initialize an array to hold the sum of distances and count of points for each slab
    distance_sums = np.zeros(num_slabs)
    point_counts = np.zeros(num_slabs)

    # Iterate over each face of the mesh
    for face in mesh.faces:
        # Get the vertices of the face
        vertices = mesh.vertices[face]

        # Calculate the centroid of the face
        centroid = np.mean(vertices, axis=0)

        # Adjust the projection calculation
        projection = np.dot(centroid - mesh.bounds[0], axis)  # Adjust centroid by the lower bound
        slab_index = int(projection / slab_thickness)
        slab_index = max(0, min(slab_index, num_slabs - 1))  # Ensure the index is within bounds

        # Calculate the average distance of the vertices of the face to the axis
        distances = np.linalg.norm(np.cross(vertices - centroid, axis), axis=1) / np.linalg.norm(axis)
        avg_distance = np.mean(distances)

        # Add this distance to the sum for the corresponding slab
        distance_sums[slab_index] += avg_distance
        point_counts[slab_index] += 1

        # Calculate the average distance for each slab
    average_distances = np.divide(distance_sums, point_counts, out=np.zeros_like(distance_sums), where=point_counts > 0)
    return average_distances



def compute_variance_distances(mesh, axis, num_slabs):
    """
    Calcule la variance de la distance aux surfaces depuis un axe donné.
    :param mesh: Objet trimesh normalisé.
    :param axis: Axe le long duquel calculer la variance des distances.
    :param num_slabs: Nombre de tranches pour la discrétisation.
    :return: Vecteur de variance de distance.
    """
    # Get the bounds of the mesh along the axis
    bounds = mesh.bounds
    min_bound, max_bound = np.min(bounds @ axis), np.max(bounds @ axis)

    # Calculate the thickness of each slab
    slab_thickness = (max_bound - min_bound) / num_slabs

    # Initialize arrays to hold the sum of distances, sum of squared distances, and count of points for each slab
    distance_sums = np.zeros(num_slabs)
    squared_distance_sums = np.zeros(num_slabs)
    point_counts = np.zeros(num_slabs)

    # Iterate over each face of the mesh
    for face in mesh.faces:
        # Get the vertices of the face
        vertices = mesh.vertices[face]

        # Calculate the centroid of the face
        centroid = np.mean(vertices, axis=0)

        # Adjust projection calculation
        projection = np.dot(centroid - mesh.bounds[0], axis)  # Adjust centroid by the lower bound
        slab_index = int(projection / slab_thickness)
        slab_index = max(0, min(slab_index, num_slabs - 1))  # Ensure the index is within bounds

        # Calculate the distance of each vertex of the face to the axis
        distances = np.linalg.norm(np.cross(vertices - centroid, axis), axis=1) / np.linalg.norm(axis)

        # Update sums and counts for the slab
        distance_sums[slab_index] += np.sum(distances)
        squared_distance_sums[slab_index] += np.sum(distances ** 2)
        point_counts[slab_index] += len(vertices)

    # Use np.divide for robust division and calculating variance
    avg_square = np.divide(squared_distance_sums, point_counts, out=np.zeros_like(squared_distance_sums), where=point_counts > 0)
    square_avg = np.divide(distance_sums, point_counts, out=np.zeros_like(distance_sums), where=point_counts > 0) ** 2
    variance_distances = np.where(point_counts > 0, avg_square - square_avg, 0)

    return variance_distances


def extract_features(mesh, num_slabs=10):
    """
    Extrait les caractéristiques de forme d'un modèle 3D.
    :param mesh: Objet trimesh normalisé.
    :param num_slabs: Nombre de tranches pour la discrétisation le long de chaque axe.
    :return: Vecteur de caractéristiques.
    """
    principal_axes = mesh.principal_inertia_vectors
    features = []

    for axis in principal_axes:
        moments = compute_moments_monte_carlo(mesh, axis, num_slabs)
        avg_distances = compute_average_distances(mesh, axis, num_slabs)
        var_distances = compute_variance_distances(mesh, axis, num_slabs)

        # Concatenate features from each axis
        features.extend([moments, avg_distances, var_distances])

    # Combine all features into a single vector
    feature_vector = np.concatenate(features)
    return feature_vector



def euclidean_distance(feature_vector1, feature_vector2):
    """
    Calcule la distance euclidienne entre deux vecteurs de caractéristiques.
    :param feature_vector1: Premier vecteur de caractéristiques.
    :param feature_vector2: Second vecteur de caractéristiques.
    :return: Distance euclidienne.
    """
    return distance.euclidean(feature_vector1, feature_vector2)


def elastic_matching_distance(feature_vector1, feature_vector2):
    """
    Calcule la distance de correspondance élastique entre deux vecteurs de caractéristiques.
    :param feature_vector1: Premier vecteur de caractéristiques.
    :param feature_vector2: Second vecteur de caractéristiques.
    :return: Distance de correspondance élastique.
    """
    len1, len2 = len(feature_vector1), len(feature_vector2)
    dp = np.zeros((len1 + 1, len2 + 1))

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = np.abs(feature_vector1[i - 1] - feature_vector2[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len1][len2]


def extract_features_and_save(directory, output_file):
    """
    Extrait les caractéristiques des modèles 3D dans un répertoire et les sauvegarde dans un fichier CSV.
    :param directory: Répertoire contenant les fichiers .obj.
    :param output_file: Chemin du fichier CSV de sortie.
    """
    # Find all .obj files in the directory and its subdirectories
    file_paths = glob.glob(os.path.join(directory, '**/*.obj'), recursive=True)

    # Open a CSV file to store the features
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['File Path', 'Feature Vector'])

        # Process each file
        for file_path in file_paths:
            print(f"Processing {file_path}...")
            model = load_3d_model(file_path)
            if model is not None:
                model = normalize_pose(model)
                feature_vector = extract_features(model)
                # Write the feature vector to the CSV file
                writer.writerow([file_path, feature_vector])


# Specify the directory and output file
directory = '/root/Project/3DPotteryDataset_v_1/3D Models'
output_file = 'model_features.csv'

# Extract features and save them
extract_features_and_save(directory, output_file)
