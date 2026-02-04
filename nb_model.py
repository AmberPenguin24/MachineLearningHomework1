import math
import random

class NaiveBayesModel:
    '''Class representing the implementation of the Naive Bayes model'''

    def __init__(self) -> None:
        self.occurrence_table = {}
        self.probability_table = {}
        self.labels = [] #in reality is the sentients/classes
        

    def train_model(self, data, labels):
        '''Runs the training process for the model, building the occurrence table and probability table'''

        #TODO - complete this function which runs the entire training process of Naive Bayes
        self.labels = list(set(labels))  #store unique labels
        self.occurrence_table = self.build_occurrence_table(data, labels)
        self.probability_table = self.build_probability_table()



    def build_occurrence_table(self, data, labels):
        '''Private function to create the occurrence table given the training data and labels'''

        #TODO - complete this function which creates a nested dictionary table of frequencies based on the training data.
        occurrence_table = {}    
        #iterate through each data point and its corresponding label
        for word_list, label in zip(data, labels): 

            #iterate through each word in the data point
            if label not in occurrence_table:
                occurrence_table[label] = {}

            for word in word_list:
                if word not in occurrence_table[label]: #if no word in label, add count 1
                    occurrence_table[label][word] = 1
                else: 
                    occurrence_table[label][word] += 1 #if in both, increment the count


        return occurrence_table
        

    def build_probability_table(self):
        '''Private function to create the probability table based on the occurrence table'''

        #this funtion builds the probabilities of each word given each label/sentiment
        #NOTE: this is P(word|Ck) where Ck is the class/label/sentiment idk why sentiment is not just sentiment and is label >:(
        #TODO - complete this function which generates a table of probabilities based on the frequencies recorded in the occurrence table
        probability_table = {}

        label_counts = {}
        #Count total words for each label/sentiment
        for label, word_counts in self.occurrence_table.items():
            label_counts[label] = 0

            for word, count in word_counts.items(): 
                label_counts[label] += count  #sum up counts of each word for the label {word, count}
        
        #Calculate probabilities of each word given each label/sentiment
        for label, words in self.occurrence_table.items():
            probability_table[label] = {}
            for word, count in words.items():
                probability_table[label][word] = count / label_counts[label]

        return probability_table
                

    def predict(self, variables):
        '''Takes a set of variables, and predicts the class they should belong to'''

        #TODO - implement prediction using the Naive Bayes method
        # remember that if the model has the same probability of either class, it should pick randomly between the two
        label_probabilities = {}
        a = 1e-16  #smoothing factor to avoid zero probabilities
        for label in self.labels:
            probability = 1.0
            prob_label = 1.0 / len(self.labels) #the prior prob of getting each label P(Label)

            for word in variables:
                #if the word is not in the probability table, we assume its probability is 0 then add the smoothing factor
                #get funtion returns 0 if word not found and if label not found
                word_probability = 0
                if word in self.probability_table[label].keys():
                    word_probability = self.probability_table[label].get(word, 0) + a
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
    