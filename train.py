import csv
import numpy as np

def import_data():
    X = np.genfromtxt("train_X_lg.csv",delimiter=',',dtype = np.float128,skip_header = 1)
    Y = np.genfromtxt("train_Y_lg.csv",delimiter=',',dtype = np.float128)
    return X, Y

def sigmoid(Z):
    sigma = 1/(1+(np.e)**(-1 * Z))
    return sigma

def compute_gradient_of_cost_function(X, Y, W, b):
    Z = np.dot(X,W) + b
    A = sigmoid(Z)
    dZ = A - Y
    dW = (1/len(X))*(np.dot(X.T,dZ))
    dB = (1/len(X))*np.sum(dZ)
    return dW,dB

def loss(Y,Hx):    
    loss = np.zeros(len(Y))
    for i in range(len(Hx)):
        if Hx[i] == 0:
            Hx[i] = 0.0001
        elif Hx[i] == 1:
            Hx[i] = 0.9999
        else:
            pass

    loss = -1 *(np.multiply(Y,np.log(Hx)) + np.multiply((1 - Y),np.log((1-Hx))) )
    return loss

def compute_cost(X, Y, W, b):
    Z = np.dot(X,W) + b
    Hx = sigmoid(Z)
    Loss = loss(Y,Hx)    
    cost = np.sum(Loss)/len(X)
    return cost

def optimize_weights_using_gradient_descent(X,Y,alpha,max_iter):
    W = np.zeros((len(X[1]),1))
    B = 0
    prev_ite_cost = 0
    iter_cnt = 0
    while True:
        iter_cnt += 1
        dW,dB = compute_gradient_of_cost_function(X,Y,W,B)
        W -= alpha * dW
        B -= alpha * dB
        cost = compute_cost(X,Y,W,B)

        if iter_cnt % 5000 == 0:
            print("{:<10} {:<10} {} ".format(iter_cnt,round(cost,10),abs(prev_ite_cost - cost)))

        if abs(prev_ite_cost - cost) < 0.0000001 or iter_cnt == max_iter:
            print(iter_cnt,cost)
            break

        prev_ite_cost = cost
    return W,B


def split_classes(Y):
    Y0 = np.zeros((len(Y),1))
    Y1 = np.zeros((len(Y),1))
    Y2 = np.zeros((len(Y),1))
    Y3 = np.zeros((len(Y),1))
    for i in range(len(Y)):
        if Y[i] == 0:
            Y0[i] = 1
        elif Y[i] == 1:
            Y1[i] = 1
        elif Y[i] == 2:
            Y2[i] = 1
        else:
            Y3[i] = 1

    return (Y0.astype(int),Y1.astype(int),Y2.astype(int),Y3.astype(int))

def To_1D(w0,w1,w2,w3):
    W0 = np.zeros(len(w0))
    W1 = np.zeros(len(w1))
    W2 = np.zeros(len(w2))
    W3 = np.zeros(len(w3))
    for i in range(len(w0)):
        W0[i] = w0[i][0]
        W1[i] = w1[i][0]
        W2[i] = w2[i][0]
        W3[i] = w3[i][0]
    return (W0,W1,W2,W3)

def train_models(X,Y):
    X,Y = import_data()
    Y0,Y1,Y2,Y3 = split_classes(Y)
    alpha = 0.000001
    max_iter = 150000
    W0,B0 = optimize_weights_using_gradient_descent(X,Y0,alpha,max_iter)
    W1,B1 = optimize_weights_using_gradient_descent(X,Y1,alpha,max_iter)
    W2,B2 = optimize_weights_using_gradient_descent(X,Y2,alpha,max_iter)
    W3,B3 = optimize_weights_using_gradient_descent(X,Y3,alpha,max_iter)

    W0,W1,W2,W3 = To_1D(W0,W1,W2,W3)
    Bs = [B0,B1,B2,B3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    weights = [W0,W1,W2,W3,Bs]
    
    return weights

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X,Y = import_data()
    weights = train_models(X,Y)
    save_model(weights,"WEIGHTS_FILE.csv")
