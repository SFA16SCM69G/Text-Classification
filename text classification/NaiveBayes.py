import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
#To plot pc curves
#import matplotlib.pyplot as plt

#Data: train(labeled) and test(unlabled)
#Use add-k smoothing
#Main fomula: P = (count(W = wi and Y = yi) + k)/(count(Y = yi) + |V|k)

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        #File data is read by imdb.py
        #ALPHA is the parameter for add-k smoothing
        self.ALPHA = ALPHA
        self.data = data # Training data
        #Initalize parameters
        self.vocab_len = data.X.shape[1]
        self.count_positive = [0 for i in range(self.vocab_len)]
        self.count_negative = [0 for i in range(self.vocab_len)]
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg = 0.0
        #Store the weight of each words to get most positive and negative words
        #self.word_weight_pos = []
        #self.word_weight_neg = []
        self.Train(data.X,data.Y)
    
    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        #X_pos is the matrix of positive reviews
        #X_neg is the matrix of negative reviews
        X_pos = X[positive_indices]
        X_neg = X[negative_indices]
        
        self.num_positive_reviews = X_pos.shape[0]
        self.num_negative_reviews = X_neg.shape[0]
        
        self.P_positive = float(self.num_positive_reviews / len(Y))
        self.P_negative = float(self.num_negative_reviews / len(Y))
        
        #Add counts on each column to get total count of W=wi and Y=yi
        self.count_positive_matrix = X_pos.sum(axis = 0)
        self.count_negative_matrix = X_neg.sum(axis = 0)
        self.count_positive = np.array(self.count_positive_matrix)[0]
        self.count_negative = np.array(self.count_negative_matrix)[0]

        self.total_positive_words = X_pos.sum()
        self.total_negative_words = X_neg.sum()
        
        #Denominator is count(Y=yi) + |V|k
        self.deno_pos = self.deno_pos + self.num_positive_reviews + float(self.vocab_len * self.ALPHA)
        self.deno_neg = self.deno_neg + self.num_negative_reviews + float(self.vocab_len * self.ALPHA)
        
        #Get weight of each word
        #for i in range(self.vocab_len):
            #pos_weight = np.log(self.count_positive[i] + self.ALPHA)-np.log(self.deno_pos)
            #neg_weight = np.log(self.count_negative[i] + self.ALPHA)-np.log(self.deno_neg)
            #self.word_weight_pos.append(pos_weight)
            #self.word_weight_neg.append(neg_weight)
        #Get and print 20 most positive and negative words
        #Form of print is array of wordId,sorted weight
        #self.get_most_20_vocab(self.word_weight_pos)
        #self.get_most_20_vocab(self.word_weight_neg)
        
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #Implement Naive Bayes Classification
        self.P_positive = 0.0
        self.P_negative = 0.0
        pred_labels = []
        
        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            for j in range(len(z[0])):
                #Index of none zero value on X[i]
                index = z[1][j]
                #Calculate add-k smoothing probability and take log
                prob_posi = float(X[i,index] * (np.log(self.count_positive[index] + self.ALPHA)-np.log(self.deno_pos)))
                self.P_positive = self.P_positive + prob_posi
                prob_nega = float(X[i,index] * (np.log(self.count_negative[index] + self.ALPHA)-np.log(self.deno_neg)))
                self.P_negative = self.P_negative + prob_nega
            
            if self.P_positive > self.P_negative:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)
            
            self.P_positive = 0.0
            self.P_negative = 0.0

        return pred_labels

    def LogSum(self, logx, logy):   
        # Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    def PredictProb(self, test, indexes):
    #PredictProb pc curves of given threshold
    #def PredictProb(self, test, indexes, probThresh):
        #Store the predict matrix to calulate precision and recall
        #predicted = []
        for i in indexes:
            # Predict the probability of the i_th review in test being positive review
            # Use the LogSum function to avoid underflow/overflow
            predicted_label = 0
            z = test.X[i].nonzero()
            #log(p(w|Y=1.0))
            posi_nume_w = 0.0
            #log(p(w|Y=-1.0))
            nega_nume_w = 0.0
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                posi_nume_w = posi_nume_w + float(test.X[row_index,col_index] * (np.log(self.count_positive[col_index] + self.ALPHA)-np.log(self.deno_pos)))
                nega_nume_w = nega_nume_w + float(test.X[row_index,col_index] * (np.log(self.count_negative[col_index] + self.ALPHA)-np.log(self.deno_neg)))
            
            predicted_prob_positive = 0.5
            predicted_prob_negative = 0.5
            #log(p(w|Y=1.0)p(Y=1.0))
            posi_nume = posi_nume_w + log(predicted_prob_positive)
            #log(p(w|Y=-1.0)p(Y=-1.0))
            nega_nume = nega_nume_w + log(predicted_prob_negative)
            
            #Denominator
            deno = self.LogSum(posi_nume,nega_nume)
            
            predicted_prob_positive = exp(posi_nume - deno)
            predicted_prob_negative = exp(nega_nume - deno)
            
            if predicted_prob_positive > predicted_prob_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
            
            #PredictProb for pc curves without threshold
            #if predicted_prob_positive > predicted_prob_negative:
                #predicted.append(1.0)
            #else:
                #predicted.append(-1.0)
            
            #PredictProb for pc curves with threshold
            #if predicted_prob_positive > probThresh:
                #predicted.append(1.0)
            #else:
                #predicted.append(-1.0)
            
            #Print test.Y[i], test.X_reviews[i]
            #Comment line below when PredictProb with threshold
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])
    
        #Print positive precision,positive recall,negative precision and negative recall without threshold
        #print self.EvalPrecision(predicted,test.Y,1.0)
        #print self.EvalPrecision(predicted,test.Y,-1.0)
        #print self.EvalRecall(predicted,test.Y,1.0)
        #print self.EvalRecall(predicted,test.Y,-1.0)
            
        #Return positive precision, positive recall, negative precision and negative recall with given threshold
        #posi_precision = self.EvalPrecision(predicted,test.Y,1.0)
        #posi_recall = self.EvalRecall(predicted,test.Y,1.0)
        #nega_precision = self.EvalPrecision(predicted,test.Y,-1.0)
        #nega_recall = self.EvalRecall(predicted,test.Y,-1.0)
        #return [posi_precision,posi_recall,nega_precision,nega_recall]

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    #Compute the precision for a given class
    def EvalPrecision(self,predicted,condition,label):
        if label == 1.0:
            TP = 0.0
            FP = 0.0
            for i in range(len(predicted)):
                if predicted[i] == 1.0 and condition[i] == 1.0:
                    TP = TP + 1
                elif predicted[i] == 1.0 and condition[i] == -1.0:
                    FP = FP + 1
            return float(TP/(TP+FP))
        else:
            TN = 0.0
            FN = 0.0
            for i in range(len(predicted)):
                if predicted[i] == -1.0 and condition[i] == -1.0:
                    TN = TN + 1
                elif predicted[i] == -1.0 and condition[i] == 1.0:
                    FN = FN + 1
            return float(TN/(TN+FN))

    #Compute the recall for a given class
    def EvalRecall(self,predicted,condition,label):
        if label == 1.0:
            TP = 0.0
            FN = 0.0
            for i in range(len(predicted)):
                if predicted[i] == 1.0 and condition[i] == 1.0:
                    TP = TP + 1
                elif predicted[i] == -1.0 and condition[i] == 1.0:
                    FN = FN + 1
            return float(TP/(TP+FN))
        else:
            TN = 0.0
            FP = 0.0
            for i in range(len(predicted)):
                if predicted[i] == -1.0 and condition[i] == -1.0:
                    TN = TN + 1
                elif predicted[i] == 1.0 and condition[i] == -1.0:
                    FP = FP + 1
            return float(TN/(TN+FP))

    def get_most_20_vocab(self,word_weight):
        sorted_index = np.argsort(word_weight)
        rev_sorted_index = sorted_index[::-1]
        most_20 = rev_sorted_index[:20]
        print most_20
        for i in range(20):
            index = most_20[i]
            print word_weight[index]

if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % "../data/aclImdb")
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % "../data/aclImdb", vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, 1.0)
    print("Evaluating")
    #Estimate probability for the first 10 reviews in the test data
    #nb.PredictProb(testdata,range(10))
    #Compute precision and recall without threshold
    #nb.PredictProb(testdata,range(25000))
    """
    #Plot pc curves
    #Thresholds are 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9
    positive_precision = []
    positive_recall = []
    negative_precision = []
    negative_recall = []
    thresholds = np.linspace(0.1, 0.9, num=17)
    for t in thresholds:
        pr = nb.PredictProb(testdata,range(25000),t)
        positive_precision.append(pr[0])
        positive_recall.append(pr[1])
        negative_precision.append(pr[2])
        negative_recall.append(pr[3])

    print positive_precision
    print positive_recall
    print negative_precision
    print negative_recall
    plt.plot(positive_recall, positive_precision, 'k-',label='',color='red')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig('positive.jpg')
    plt.show()
    plt.plot(negative_recall, negative_precision, 'k--',color='blue')
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig('negative.jpg')
    plt.show() """

    print("Test Accuracy: ", nb.Eval(testdata))

