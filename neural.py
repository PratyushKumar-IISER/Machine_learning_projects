import numpy as np
import matplotlib.pyplot as plt

class SingleNeuronClassifier:
    def __init__(self, learning_rate=0.1, threshold=0):
        self.weights = None
        self.threshold = threshold
        self.learning_rate = learning_rate
      

    def fit(self, X, y, final_weights,epochs=100 ):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1)  # add bias term
        #print(X)
        for epoch in range(epochs):
            weighted_sum = np.dot(X, self.weights.T)
            predictions = np.where(weighted_sum >= self.threshold, 1, 0)
            error = y - predictions
            if error.any() == 0:
                print("Converged")
                print(error)
                print(self.weights)
                #self.show_graph(X,y,self.weights,epoch)
                final_weights.append(list(self.weights))
                print("NOw breaking")
                break
            self.weights += self.learning_rate * np.dot(X.T, error)
            #if epoch%2 == 0:
            final_weights.append(list(self.weights))
                #self.show_graph(X,y,self.weights,epoch)



    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # add bias term
        weighted_sum = np.dot(X, self.weights.T)
        predictions = np.where(weighted_sum >= self.threshold, 1, 0)
        return predictions
X = np.array([[1,1],[0.5,0.5],[1,2],[2,1],[10,10],[10,10.5],[11,11],[11,10.5]])
y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
nue = SingleNeuronClassifier()
final_weights = []
nue.fit(X,y1,final_weights,100)
print(nue.predict(X))
print(final_weights)
print(len(final_weights))
counts=0
x = np.linspace(-5,5,100)
for w in range(len(final_weights)):
        plt.scatter(X[y1==0, 0], X[y1==0, 1], color='blue', label='0')
        plt.scatter(X[y1==1, 0], X[y1==1, 1], color='red', label='1')
        plt.legend()
        y = -((final_weights[w][1])/float(final_weights[w][2]))*x - final_weights[w][0]/float(final_weights[w][2])
        with plt.ion():
            plt.plot(x, y, '-r', label='epochs = {k}')
            plt.title(f'Graph for epochs = {counts}')
        #plt.xlabel('x', color='#1C2833')
        #plt.ylabel('y', color='#1C2833')
        #plt.legend(loc='upper left')
            plt.grid()
            plt.show()
            plt.pause(1)
            plt.clf()
        counts+=1
        #plt.close()
    