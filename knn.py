import math
from collections import Counter

def euclidean_distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def knn_predict(training_data, training_labels, new_point, k):
    distances = []
    for i in range(len(training_data)):
        distance = euclidean_distance(training_data[i],new_point)
        distances.append((distance, training_labels[i]))

    # short distances
    distances.sort(key=lambda x:x[0])

    # get k nearest neighbors
    neighbors = distances[:k]

    #voiting 
    labels = [label for _, label in neighbors]
    most_common = Counter(labels).most_common(1)

    return most_common[0][0]

