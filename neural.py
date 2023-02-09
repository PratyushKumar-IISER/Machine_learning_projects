import numpy as np
import matplotlib.pyplot as plt

class SingleNeuronClassifier:
    def __init__(self, learning_rate=0.1, threshold=0):
        self.weights = None
        self.threshold = threshold
        self.learning_rate = learning_rate
      

    def fit(self, X, y, final_weights,epochs=100 ):
        self.weights = np.random.randn(X.shape[1] + 1)
        X = np.insert(X, 0, 1, axis=1)  # add constant term
        #print(X)
        for epoch in range(epochs):
            weighted_sum = np.dot(X, self.weights.T)
            predictions = np.where(weighted_sum >= self.threshold, 1, 0)
            error = y - predictions
            if error.any() == 0:
                print(f"Converged in {epoch} iterations")
                #print(error)
                #print(self.weights)
                #self.show_graph(X,y,self.weights,epoch)
                final_weights.append(list(self.weights))
                break
            self.weights += self.learning_rate * np.dot(X.T, error)
            #if epoch%2 == 0:
            final_weights.append(list(self.weights))
                #self.show_graph(X,y,self.weights,epoch)



    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # add constant term
        weighted_sum = np.dot(X, self.weights.T)
        predictions = np.where(weighted_sum >= self.threshold, 1, 0)
        return predictions
N = 20    #number of random normal points
dim = 2
Xn1 = np.random.normal(0,2,size=(N,dim))
Xn2 = np.random.normal(12,2,size=(N, dim))
Y_zeros = np.zeros(N)
Y_ones = np.ones(N)
y1 = np.concatenate((Y_zeros,Y_ones))
X = np.concatenate((Xn1,Xn2),axis=0)
x_min = X.min(axis=0)[0]
x_max = X.max(axis=0)[0]
y_min = X.min(axis=0)[1]
y_max = X.max(axis=0)[1] 
#X = np.array([[1,1],[0.5,0.5],[1,2],[2,1],[10,10],[10,10.5],[11,11],[11,10.5]])
#y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
nue = SingleNeuronClassifier()
final_weights = []
nue.fit(X,y1,final_weights,100)
print("Prediction:",nue.predict(X))
#print(final_weights)
#print(len(final_weights))
counts=0
x = np.linspace(x_min-6,x_max+6,100)
for w in range(len(final_weights)):
        plt.xlim(x_min-6,x_max+6)
        plt.ylim(y_min-6,y_max+6)
        plt.scatter(X[y1==0, 0], X[y1==0, 1], color='blue', label='0')
        plt.scatter(X[y1==1, 0], X[y1==1, 1], color='red', label='1')
        plt.legend()
        y = -((final_weights[w][1])/float(final_weights[w][2]))*x - final_weights[w][0]/float(final_weights[w][2])
        with plt.ion():
            plt.plot(x, y, '-g', label='epochs = {k}')
            plt.title(f'Graph for epochs = {counts}')
            plt.grid()
            plt.show()
            plt.pause(0.5)
            plt.clf()
        counts+=1
        
    