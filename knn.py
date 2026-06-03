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

    # 🔍 PRINT NEIGHBORS
    print("\nTop K Neighbors:")
    for d in neighbors:
        print(f"Distance: {d[0]:.4f}, Label: {d[1]}")

    #voiting 
    labels = [label for _, label in neighbors]
    print("\nVoting labels:", labels)
    vote_count = Counter(labels)
    print("Vote count:", vote_count)
    most_common = Counter(labels).most_common(1)

    print("Final decision:", most_common[0][0])
    
    return most_common[0][0]

