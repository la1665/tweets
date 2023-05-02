import numpy as np

def sigmoid(X):

    return 1/(1+np.exp(-X))

def feed_forward(W, b, X):

    Z =  np.dot(W, X)+b
    A = sigmoid(Z)

    return Z, A

def cost(A, Y):

    m = Y.shape[1]
    L = np.dot(Y, np.log(A.T))+ np.dot(1-Y, np.log(1-A.T))

    return -1/m*L

def initilize_weigth(n_x):

    W = np.zeros((1, n_x))
    b = np.zeros((1, 1))

    return W, b
def back_propagation(A, X, Y):

    m = X.shape[1]
    dZ = A- Y
    dW = np.dot(dZ, X.T)/m
    db = np.sum(dZ, axis=1)/m

    return dW, db

def update(W, b, dW, db, learning_rate):

    W = W - learning_rate*dW
    b = b- learning_rate*db

    return W, b


def train(num_iter, learning_rate, X, Y):
    nx = X.shape[0]
    W, b = initilize_weigth(nx)
    costs = []
    for i in range(num_iter):
        Z, A = feed_forward(W, b, X)
        cs = cost(A, Y)
        costs.append(np.squeeze(cs))
        dW, db = back_propagation(A, X, Y)

        W, b = update(W, b, dW, db, learning_rate)

X = np.random.random((2, 4))
Y = np.array([1, 0, 1, 0])
Y = np.expand_dims(Y, axis=0)

train(10, .001, X, Y)



