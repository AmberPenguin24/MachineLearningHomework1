import math
import random

class NaiveBayesModel:
    '''Class representing the implementation of the Naive Bayes model'''

    def __init__(self) -> None:
        self.occurrence_table = {}
        self.probability_table = {}
        self.sentiment = [] #in reality is the sentiment/classes
        

    def train_model(self, data, labels):
        '''Runs the training process for the model, building the occurrence table and probability table'''

        #TODO - complete this function which runs the entire training process of Naive Bayes
        self.sentiment = list(set(labels))  #store unique labels
        self.occurrence_table = self.build_occurrence_table(data, labels)
        self.probability_table = self.build_probability_table()



    def build_occurrence_table(self, data, labels):
        '''Private function to create the occurrence table given the training data and labels'''

        #TODO - complete this function which creates a nested dictionary table of frequencies based on the training data.
        occurrence_table = {}    
        #iterate through each data point and its corresponding label
        data = data.to_list()
        labels = labels.to_list()
        for word_list, label in zip(data, labels): 

            #iterate through each word in the data point
            for word in word_list:
                if word not in occurrence_table.keys(): #if no word in label, add count 1
                    occurrence_table[word] = {}
                   
                if label not in occurrence_table[word].keys():
                    occurrence_table[word][label] = 1
                else: 
                    occurrence_table[word][label] += 1 #if in both, increment the count

        
        return occurrence_table
        

    def build_probability_table(self):
        '''Private function to create the probability table based on the occurrence table'''

        #this funtion builds the probabilities of each word given each label/sentiment
        #NOTE: this is P(word|Ck) where Ck is the class/label/sentiment idk why sentiment is not just sentiment and is label >:(
        #TODO - complete this function which generates a table of probabilities based on the frequencies recorded in the occurrence table
        probability_table = {}

        label_counts = {}
        #Count total words for each label/sentiment
        for word, labels in self.occurrence_table.items():

            for label, count in labels.items(): 
                if label not in label_counts:
                    label_counts[label] = count
                else:
                    label_counts[label] += count  #sum up counts of each word for the label {word, count}
        
        #Calculate probabilities of each word given each label/sentiment
        for word, labels in self.occurrence_table.items():
            for label, count in labels.items():
                if word not in probability_table.keys():
                    probability_table[word] = {}
                
                if label not in probability_table[word].keys():
                    probability_table[word][label] = count / label_counts[label] 


        return probability_table
                

    def predict(self, data):
        '''Takes a set of data points, and predicts the class they should belong to'''

        predictions = []

        for variables in data:
            prediction = self.predict_single_tweet(variables)
            predictions.append(prediction)

        return predictions

    def predict_single_tweet(self, variables):
        '''Takes a set of variables, and predicts the class they should belong to'''

        #TODO - implement prediction using the Naive Bayes method
        # remember that if the model has the same probability of either class, it should pick randomly between the two
        label_probabilities = {}
        a = 1e-16  #smoothing factor to avoid zero probabilities 
        for label in self.sentiment:
            probability = 1.0
            prob_label = 1.0 / len(self.sentiment) #the prior prob of getting each label P(Label)

            for word in variables:
                #if the word is not in the probability table, we assume its probability is 0 then add the smoothing factor
                #get funtion returns 0 if word not found and if label not found
                word_probability = 0

                if word in self.probability_table.keys():
                    word_probability = self.probability_table[word].get(label, 0) + a
                else:
                    word_probability = a
                    
                probability *= word_probability

            label_probabilities[label] = probability * prob_label

        max_prob = max(label_probabilities.values())

        #gets all the labels with the max probability
        best_labels = []

        for label, prob in label_probabilities.items():
            if prob == max_prob:
                best_labels.append(label)
        
        return random.choice(best_labels)
    