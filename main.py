from knn import knn_predict
from utils import load_csv, normalize


#load data
x,y = load_csv('customer_data.csv')

# normalize data

X = normalize(x)


# input test data 

new_point = [30, 50000]

def normalize_input(x, X):
    min_values = [min(col) for col in zip(*X)]
    max_values = [max(col) for col in zip(*X)]

    norm_row = [
        (x[i] - min_values[i] / (max_values[i] - min_values[i]))
        for i in range(len(x))
    ]

    return norm_row

new_point_norm = normalize_input(new_point, X)

# predict
k = 5
result = knn_predict(X, y, new_point_norm, k)

print(f"Predicted class for new point {new_point}: {result}")
print("Prediction:", "Will Buy" if result == 1 else "Will NOT Buy")

