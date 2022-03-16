import numpy as np

class NaiveBayesClassification:

    def __init__(self, A, B):
        self.A = A 
        self.B = B 

        #used in both methods
        self.word_is_positive_prob = np.zeros(len(self.A[0])) #length = words loaded -2
        self.word_is_negative_prob = np.zeros(len(self.A[0])) #length = words loaded -2

        self.Prob_reviews_are_postitive = 0
        self.Prob_reviews_are_negative = 0

        self.k = 2  #Laplache smoothness
  
    def fit(self):
        
        total_positive_words = 0 #sum of ALL the words used in a positive review
        total_negative_words = 0 #sum of ALL the words used in a negative review

        m = len(self.A) #num of reviews (2500)
        n = len(self.A[0]) #words loaded -2

        rows1 = np.where(self.B==1)[0]
        rows0 = np.where(self.B==0)[0]
        for j in range(n):
            self.word_is_positive_prob[j] = np.sum(self.A[rows1,j])
            self.word_is_negative_prob[j] = np.sum(self.A[rows0,j])


        for j in range(n):
            total_positive_words += np.where(self.A[:,j] + self.B == 2)[0].shape[0]
            total_negative_words += np.where(self.A[:,j] > self.B)[0].shape[0]

        for j in range(n):
            self.word_is_positive_prob[j] = (self.word_is_positive_prob[j] + 1)/(total_positive_words + self.k) #Apply Laplache smoothness
            self.word_is_negative_prob[j] = (self.word_is_negative_prob[j] + 1)/(total_negative_words + self.k) #Apply Laplache smoothness

        self.Prob_reviews_are_postitive = np.sum(self.B)
        self.Prob_reviews_are_negative = len(self.B) - self.Prob_reviews_are_postitive

        self.Prob_reviews_are_postitive = self.Prob_reviews_are_postitive / len(self.B)#0.5
        self.Prob_reviews_are_negative = self.Prob_reviews_are_negative  / len(self.B)#0.5



    def predict(self,X):

        m = len(X)
        n = len(X[0])

        Y = np.zeros(m)
  
        
        for i in range(m): #for each review
            
            cols = np.where(X[i,:]==1)[0]
            P_review_1 = self.Prob_reviews_are_postitive*np.prod(self.word_is_positive_prob[cols])
            P_review_0 = self.Prob_reviews_are_negative*np.prod(self.word_is_negative_prob[cols])

            #Normalization
            temp_1 = P_review_1
            P_review_1 = P_review_1 / (P_review_1 + P_review_0)
            P_review_0 = P_review_0 / (temp_1 + P_review_0)

            Y[i] = 1 if P_review_1 > P_review_0 else 0

        return Y
