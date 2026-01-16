x = [1, 2, 3]
y = [2, 4, 6]

learning_rate = 0.1
epochs = 5

m = 0
b = 0


for _ in range(epochs):

    dm = 0
    db = 0

    for i in range(len(x)):
        y_pred = m * x[i] + b
        error = y[i] - y_pred

        dm = dm + error
        db = db + error

    m = m + learning_rate * dm
    b = b + learning_rate * db

    print("m:", m, "b:", b)
