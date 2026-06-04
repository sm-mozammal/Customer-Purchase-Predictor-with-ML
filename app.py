from flask import Flask, request, jsonify
from knn import knn_predict
from utils import load_csv, normalize


app = Flask(__name__)

#load data

x,y = load_csv('customer_data.csv')

print(f"Loaded data: {len(x)} samples")

#normalize data

X_norm = normalize(x)

print("Data normalized")

def normalize_input(x, X):
    min_values = [min(col) for col in zip(*X)]
    max_values = [max(col) for col in zip(*X)]
    return [
        (x[i] - min_values[i]) / (max_values[i] - min_values[i])
        for i in range(len(x))
        ]



@app.route("/")
def home():
    return "Customer Purchase Predictor API"

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    try:
        age = float(data['age'])
        salary = float(data['salary'])
    except:
        return jsonify({"error":"invalide input, age and salary must be numbers"}), 400
    
    new_coustomer = [age, salary]
    new_coustomer_norm = normalize_input(new_coustomer, x) 


    result = knn_predict(X_norm,y, new_coustomer_norm, k=10)

    def calculate_accuracy(X, y, k):
        correct = 0

        for i in range(len(X)):
            # leave-one-out (simple way)
            X_train = X[:i] + X[i+1:]
            y_train = y[:i] + y[i+1:]

            prediction = knn_predict(X_train, y_train, X[i], k)

            if prediction == y[i]:
                correct += 1

        accuracy = correct / len(X)
        return accuracy
    
    accuracy = calculate_accuracy(X_norm, y, k=10)
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return jsonify({
        "age": age,
        "salary": salary,
        "predicted_class": result,
        "prediction": "Will Buy" if result == 1 else "Will NOT Buy",
        "model_accuracy": f"{accuracy:.2%}"
    })

if __name__ == "__main__":
    app.run(debug=True)
