from flask import Flask, request, jsonify
import csv
import numpy as np
from scipy.spatial import distance
from feature_extraction import euclidean_distance, elastic_matching_distance


app = Flask(__name__)


def load_database(csv_file):
    database = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            model_name, features = row[0], np.array([float(x) for x in row[1][1:-1].split(', ')])
            database[model_name] = features
    return database


def find_similar_objects(model_name, database, top_k=5):
    if model_name not in database:
        return None

    model_features = database[model_name]
    distances = {name: euclidean_distance(model_features, features) for name, features in database.items() if
                 name != model_name}

    # Sort by distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])

    # Get top k similar objects
    similar_objects = [name for name, _ in sorted_distances[:top_k]]
    return similar_objects


@app.route('/find_similar', methods=['GET'])
def find_similar():
    model_name = request.args.get('model')
    database = load_database('model_features.csv')  # Assuming CSV file is in the same directory
    similar_objects = find_similar_objects(model_name, database)

    if similar_objects is None:
        return jsonify({'error': 'Model not found'}), 404

    return jsonify({'similar_objects': similar_objects})


if __name__ == '__main__':
    app.run(debug=True)
