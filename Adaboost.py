import numpy as np
from math import log2

def loge(x):
    return np.log10(x)/np.log10(np.exp(1))

class DecisionStump:
    
    def __init__(self, data, y, weights):
        self.data = data
        self.y = y
        self.weights = weights
        self.word_column = None
        self.left_child = None   #pc1x1 > 1 - pc1x1
        self.right_child = None  #pc1x0 > 1 - pc1x0
        self.predictions = np.empty(y.shape[0])
        self.total_error = 0
        self.alpha = None

    def calculateTotalError(self, ycorrect):

      wrong_rows = np.where(self.predictions != self.y)[0] #rows that were incorrectly classified

      for row in range(self.weights.shape[0]):

        if row in wrong_rows:

          self.total_error += self.weights[row]
         

    def calculateAmountOfSay(self, error):
        EPS = 1e-10
        self.alpha = loge( (1-error + EPS) / (error + EPS) )/2

    def fit(self):
        self.word_column, pc1x0, pc1x1 = self.calculateInformationGain(self.data, self.y, self.weights)
        self.left_child = 1 if pc1x1 > (1 - pc1x1) else 0
        self.right_child = 1 if pc1x0 > (1 - pc1x0) else 0

    def predict(self):
        predictions = np.empty(self.y.shape[0])
        xtrain = self.data[:,self.word_column]
        
        for i in range(predictions.shape[0]):
            if xtrain[i] == 1:
                predictions[i] = self.left_child
            else:
                predictions[i] = self.right_child
        self.predictions = predictions

    def printData(self):
        print('Column:', self.word_column)
        print("Total Error:",self.total_error,'\nAmount of say:',self.alpha)
        print("Left child:",self.left_child,"\nRight child:",self.right_child)

    def calculateBinEntropy(self, prob):
        if prob < 0.0 or prob > 1.0:
            print("ERROR-Function calculateBinEntropy:\n\tInvalid probability given (" + str(prob) + ")")
            return
        elif prob == 0.0 or prob == 1.0:
            return 0
        else:
            return -(prob*log2(prob) + (1-prob)*log2(1-prob))

    def YEntropy(self, y):
        prob = np.where(y == 1)[0].shape[0] / y.shape[0]
        return self.calculateBinEntropy(prob)

    def calculateInformationGain(self, table, y, weights):
        rows_len = table.shape[0]
        cols_len = table.shape[1]
        if rows_len == 0:
            print("ERROR-Function calculateInformationGain:\n\tData table has length 0")
            return
        if len(y) == 0:
            print("ERROR-Function calculateInformationGain:\n\tY table has length 0")
            return
        
        HC = self.YEntropy(y) #entropy of y

        PX1 = np.zeros(cols_len)
        PC1X1 = np.zeros(cols_len)
        PC1X0 = np.zeros(cols_len)
        HCX1 = np.zeros(cols_len)
        HCX0 = np.zeros(cols_len)

        IG = np.zeros(cols_len)

        
        for j in range(cols_len):

            column = table[:,j]

            cX1 = np.sum(weights[column == 1])
            cC1X1 = np.sum(weights[column + y == 2])            
            cC1X0 = np.sum(weights[column < y])

            PX1[j] =  cX1

            if(cX1 == 0):
                PC1X1[j] = 0.0
            else:
                PC1X1[j] = cC1X1/cX1
            
            if(cX1 == rows_len):
                PC1X0[j] = 0.0
            else:
                PC1X0[j] = cC1X0/(1-cX1)

            HCX1[j] = self.calculateBinEntropy(PC1X1[j])
            HCX0[j] = self.calculateBinEntropy(PC1X0[j])
            IG[j] = HC - ( (PX1[j] * HCX1[j]) + ( (1.0 - PX1[j]) * HCX0[j]) )

        max_ig_index = np.where(IG == max(IG))[0][0]
        return max_ig_index, PC1X0[max_ig_index], PC1X1[max_ig_index]


class Adaboost:

    def __init__(self, M, data, y):
        self.M = M
        self.data = data
        self.y = y
        self.stump_forest = np.empty(self.M, dtype=DecisionStump)
        self.weights = np.full(self.y.shape[0], 1/self.y.shape[0])

    def fit(self):

        for m in range(self.M):
            
            stump = DecisionStump(self.data, self.y, self.weights)
            stump.fit()  
            stump.predict()
            stump.calculateTotalError(self.y)
            stump.calculateAmountOfSay(stump.total_error)

            self.stump_forest[m] = stump  #update stump forest
            wrong_rows = np.where(stump.predictions != self.y)[0] #rows that were incorrectly classified
            for row in range(self.weights.shape[0]):
                if row in wrong_rows:
                    self.weights[row] = self.weights[row] * np.exp(stump.alpha)
                else:
                    self.weights[row] = self.weights[row] * np.exp(-1*stump.alpha)
            #Normalize
            self.weights /= sum(self.weights) 

    def predict(self, xtest):
        predictions = np.empty(self.y.shape[0])
        for i in range(predictions.shape[0]):
            #s = np.sum([stump.alpha * stump.predictions[i] for stump in self.stump_forest])
            s = 0
            for stump in self.stump_forest:
                stump.predictions[stump.predictions==0] = -1
                s += stump.alpha * stump.predictions[i]
            if s > 0:
                predictions[i] = 1
            else:
                predictions[i] = 0
        
        return predictions