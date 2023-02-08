import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SingleNeuronClassifier:
    def __init__(self, learning_rate=0.1, threshold=0):
        self.weights = None
        self.threshold = threshold
        self.learning_rate = learning_rate
      

    def fit(self, X, y, final_weights, epochs=100 ):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1)  # add bias term
        for epoch in range(epochs):
            weighted_sum = np.dot(X, self.weights.T)
            predictions = np.where(weighted_sum >= self.threshold, 1, 0)
            error = y - predictions
            if error.any() == 0:
                print("Converged")
                final_weights.append(list(self.weights))
                break
            self.weights += self.learning_rate * np.dot(X.T, error)
            if epoch%2 == 0:
                final_weights.append(list(self.weights))

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # add bias term
        weighted_sum = np.dot(X, self.weights.T)
        predictions = np.where(weighted_sum >= self.threshold, 1, 0)
        return predictions

X = np.array([[1,1,1],[0.5,0.5,0.5],[1,2,3],[2,1,4],[10,10,10],[10,10.5,10],[11,11,11],[11,10.5,12]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
nue = SingleNeuronClassifier()
final_weights = []
nue.fit(X,y,final_weights,100)
print(nue.predict(X))
print(final_weights)
print(len(final_weights))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

counts = 0
for w in range(len(final_weights)):
    ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], color='blue', label='0')
    ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', label='1')
    ax.legend()
    xx, yy = np.meshgrid(range(-10, 20), range(-10, 20))
    zz = (-final_weights[w][0] - final_weights[w][1] * xx - final_weights[w][2] * yy) / final_weights[w][3]
    with plt.ion():
        ax.plot_surface(xx, yy, zz)
        plt.title(f'Graph for epochs = {counts}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ######
        plt.show()
        plt.pause(5)
        plt.cla()
    counts += 1
    #plt.close()

