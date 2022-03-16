import numpy as np
from math import log2

class Node:
    """
    Node of a tree
    """
    def __init__(self):
        self.value = None
        self.children = []   
        self.is_leaf = False

    def printData(self):   
        print("Node:")     
        print("_"*6 + "\n" 
               + str(self.value) +"\n" +
               "\u203E"*6)
        print("Children:")
        for i in range(len(self.children)):
            print("_"*6 + "\n" 
                + str(self.children[i].value) +"\n" +
                "\u203E"*6)

class ID3:

    def calculateBinEntropy(self, prob):
        """
        Returns the entropy for a given probability:type(float) ranged from 0 to 1\n
        Parameters: a given probability prob:type(float)
        """
        if prob < 0.0 or prob > 1.0:
            print("ERROR-Function calculateBinEntropy:\n\tInvalid probability given (" + str(prob) + ")")
            return
        elif prob == 0.0 or prob == 1.0:
            return 0
        else:
            return -(prob*log2(prob) + (1-prob)*log2(1-prob))

    def calculateInformationGain(self, table, y):
        """
        Returns an np.array() with the information gain of each feature in the table.\n
        Parameters: a data table:type(np.ndarray), result table y:type(np.array)
        """
        rows_len = table.shape[0]
        cols_len = table.shape[1]
        if rows_len == 0:
            print("ERROR-Function calculateInformationGain:\n\tData table has length 0")
            return
        if y.shape[0] == 0:
            print("ERROR-Function calculateInformationGain:\n\tY table has length 0")
            return

        prob = np.where(y == 1)[0].shape[0] / y.shape[0]
        HC = self.calculateBinEntropy(prob) #entropy of y

        PX1 = np.zeros(cols_len)
        PC1X1 = np.zeros(cols_len)
        PC1X0 = np.zeros(cols_len)
        HCX1 = np.zeros(cols_len)
        HCX0 = np.zeros(cols_len)

        IG = np.zeros(cols_len)

        for j in np.nditer(np.arange(cols_len)):
            column = table[:,j]
            cX1 = np.sum( column )
            cC1X1 = np.where( column + y == 2 )[0].shape[0] 
            cC1X0 = np.where( column < y )[0].shape[0] 
        
            PX1[j] = float(cX1/rows_len)

            if(cX1 == 0):
                PC1X1[j] = 0.0
            else:
                PC1X1[j] = float(cC1X1/cX1)
            
            if(cX1 == rows_len):
                PC1X0[j] = 0.0
            else:
                PC1X0[j] = cC1X0/(rows_len-cX1)

            HCX1[j] = self.calculateBinEntropy(PC1X1[j])
            HCX0[j] = self.calculateBinEntropy(PC1X0[j])
            IG[j] = HC - ( (PX1[j] * HCX1[j]) + ( (1.0 - PX1[j]) * HCX0[j]) )

        return IG


    def getMaxIGIndex(self, ig):
        """
        Returns the index of the maximum value in the IG array: type(float)\n
        Parameters: a table including the information gain:type(float)
        """
        if len(ig) == 0:
            print("ERROR-Function getMaxIGIndex:\n\tempty array given")
            return
        return np.where(ig == max(ig))[0][0]
    
    def getMostCommonCategory(self, y):
        """
        Returns the most common category in the table y (0 or 1)
        Parameters: table y containing 0s and 1s:type(np.array)
        Note: if count of ones is equal to count of zeros, we return 0
        """
        ones = np.where(y==1)[0].shape[0]
        zeros = np.where(y==0)[0].shape[0]
        if ones > zeros:
            return 1
        else:
            return 0
    
    def fit(self, table, y, ftrs, m = 0):
        """
        Returns the root of the tree, the result of the ID3 algorithm\n
        Parameters: data table:type(np.ndarray), decision table y:type(np.array)
        Internal nodes contain the name of the feature and leaf nodes value is 0 or 1
        """
        root = Node()
        if table.shape[0] == 0 or y.shape[0] == 0:  #examples = {}
            root.is_leaf = True
            root.value = m
            return root
        elif np.unique(y).shape[0] == 1:  #pure class
            root.is_leaf = True
            root.value = y[0]
            return root
        elif table.shape[1] == 0 or len(ftrs) == 0:  #features = {}
            root.is_leaf = True
            root.value = self.getMostCommonCategory(y)
            return root
        else:
            max_ig_index = self.getMaxIGIndex(self.calculateInformationGain(table,y))  #get index of max information gain for current table
            root.value = ftrs[max_ig_index]  #insert the word as value in internal node
            m = self.getMostCommonCategory(y)
            for value in [0,1]:
                new_y = y[table[:,max_ig_index]==value]
                new_table = table[table[:,max_ig_index]==value]
                new_table = np.delete(new_table, max_ig_index, axis=1)
                new_features = np.delete(ftrs, max_ig_index)
                root.children.append(self.fit(new_table, new_y, new_features, m))

            return root



    #testing/development
    def predict(self, xtest, root, ftrs):
        """
        Tests the ID3 algorithm
        Returns: a vector including the predictions based on the tree trained:type(np.array)
        Parameters: the testing data xtest:type(np.array), root of the tree:type(Node),
        the vocabulary:type(list)  
        """
        
        if xtest.shape[0] == 0:
            print("WARNING-Function getTrainResults\n\tLength of train data is 0")
            return 
        if root == None:
            print("ERROR-Function predict:\n\tRoot is None")
            return

        ypredict = np.zeros(xtest.shape[0])
        for i in range(xtest.shape[0]):

            node = root
            comment = xtest[i,:]
            for _ in range(comment.shape[0]+1):
                if node.is_leaf == True:
                    prediction = node.value
                    break
                
                current_feature_idx = ftrs.index(node.value)
                    
                if comment[current_feature_idx] == 0:
                    node = node.children[0]
                    continue

                if comment[current_feature_idx] == 1:
                    node = node.children[1]
                    continue
            ypredict[i] = prediction
            
        return ypredict