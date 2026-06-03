import csv


def load_csv(file_path):
    x = []
    y = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            age = float(row[0])
            sallary = float(row[1])
            label = int(row[2])

            x.append((age, sallary))
            y.append(label)

    return x, y

def normalize(x):
    min_values = [min(col) for col in zip(*x)]
    max_values = [max(col) for col in zip(*x)]

    X_norm = []

    for row in x:
        norm_row = [
            (row[i] - min_values[i]) / (max_values[i] - min_values[i])
            for i in range(len(row))
        ]

        X_norm.append(norm_row)
    return X_norm


