import numpy as np
import csv
import sys

from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path', using the weights learned during training.
Writes the predicted values to the file named "predicted_test_Y_lg.csv". It should be created in the same directory where this code file is present.
This code is provided to help you get started and is NOT a complete implementation. Modify it based on the requirements of the project.
"""

def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights_b = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    weights = np.zeros((4,len(weights_b[0])))
    for i in range(len(weights)):
        for j in range(20):
            weights[i][j] = weights_b[i][j]
    b = np.array([weights_b[4][0],weights_b[4][1],weights_b[4][2],weights_b[4][3]])
    return test_X, weights,b

def sigmoid(Z):
    sigma = 1/(1+(np.e)**(-1 * Z))
    return sigma

def predict_target_values(test_X, weights ,b):
    #print(weights)
    Z = np.dot(test_X,weights.T) + b
    Hx = sigmoid(Z)
    #print(Z)
    #print()
    #print(Hx)
    #labels = Z
    labels = np.zeros(len(test_X))
    for i in range(len(labels)):
        temp = list(Hx[i])
        labels[i] = temp.index(max(temp))
    return labels.astype(int)


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights, b = import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights, b)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    #Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg.csv") 
