import os
import pandas as pd
import re
import numpy as np
import math
#import pickle
class Naive_Bayesian:

    def __init__(self,verbose = False,dictionary = None):
        if dictionary is not None:
            self.dictionary_list = dictionary
        else:
            self.dictionary_list = None
        self.verbose = verbose

    def create_data_csv(self):
        location = os.getcwd()
        location = location + "/20_newsgroups"
        print(os.listdir(location))
        df = pd.DataFrame(columns=['Text', 'Category'])
        for directory in os.listdir(location):
            if directory != '.DS_Store':
                temp = location
                temp = temp + "/" + directory
                print(os.listdir(temp))
                list_of_files = os.listdir(temp)
                for file_name in list_of_files:
                    file_name = temp + "/" + file_name
                    # print(file_name)
                    f = open(file_name, 'rb')
                    text = f.read().decode(errors='ignore')
                    # print(text)
                    f.close()
                    # print(directory)
                    df = df.append({'Text': str(text), 'Category': str(directory)}, ignore_index=True)
        print(df.head(10))
        df.to_csv("data.csv")

    def remove_Stopwords(self,vector_of_text):#need some return type
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during',
                      'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours',
                      'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
                      'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his',
                      'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                      'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
                      'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves',
                      'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he',
                      'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after',
                      'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
                      'further', 'was', 'here', 'than', 'Newsgroups', 'talk']
        for i in range(0, len(vector_of_text)):
            text = vector_of_text[i]
            text = text.lower()
            text = re.sub('[^A-Za-z ]+', '', text)
            text = text.split()
            text = [x for x in text if len(x) < 12]
            # text = re.split(' |,|>|\-|<|\+|.|!',text)
            text = [x for x in text if x not in stop_words]
            text = " ".join(text)
            vector_of_text[i] = text

    def create_dictionary(self,vector_of_text):
        import re
        dict = {}
        for i in range(0, len(vector_of_text)):
            text = vector_of_text[i]
            text = text.split()
            # text = re.split(' |,|>|\-|<|\+|.|!', text)
            for word in text:
                if word in dict.keys():
                    dict[word] = 0
                else:
                    dict[word] = 0

        for i in range(0, len(vector_of_text)):
            dict2 = dict.copy()
            # print(dict2)
            text = vector_of_text[i]
            text = text.split()
            for word in text:
                # print("Increased")
                dict2[word] = dict2[word] + 1
                # print(word)
                # print(dict2[word])
            vector_of_text[i] = dict2

    def create_dictionary_other(self,vector_of_text):
        import re
        dict = {}
        for i in range(0, len(vector_of_text)):
            text = vector_of_text[i]
            text = text.split()
            # text = re.split(' |,|>|\-|<|\+|.|!', text)
            for word in text:
                if word in dict.keys():
                    dict[word] = 0
                else:
                    dict[word] = 0
        count = 0
        dict2 = dict.copy()
        new_list = []
        for i in range(0, len(vector_of_text)):
            # dict2 = dict.copy()
            # new_list=[]
            # print(dict2)
            text = vector_of_text[i]
            text = text.split()
            for word in text:
                # print("Increased")
                dict2[word] = dict2[word] + 1
                # print(word)
                # print(dict2[word])
            count = count + 1
            if count >= 500:
                new_list.append(dict2)
                dict2 = dict.copy()
                count = 0
                # vector_of_text[i] = dict2
        self.dictionary_list = new_list
        return new_list

    def divide_train_test_data(self,X, y):
        X_train = X[:500]
        y_train = y[:500]
        X_test = X[500:1000]
        y_test = y[500:1000]
        #print("Y")
        #print(y_train.shape)
        '''X_train = np.vstack(X_train,X[1001:1501])
        X_train = np.vstack(X_train, X[2001:1501])
        X_train = np.vstack(X_train, X[3001:3501])
        X_train = np.vstack(X_train, X[4001:4501])
        X_train = np.vstack(X_train, X[5001:5501])
        X_train = np.vstack(X_train, X[6001:6501])
        X_train = np.vstack(X_train, X[7001:7501])
        X_train = np.vstack(X_train, X[8001:8501])
        X_train = np.vstack(X_train, X[9001:9501])
        X_train = np.vstack(X_train, X[10001:10501])
        X_train = np.vstack(X_train, X[11001:11501])
        X_train = np.vstack(X_train, X[12001:12501])
        X_train = np.vstack(X_train, X[13001:13501])
        X_train = np.vstack(X_train, X[14001:14501])
        X_train = np.vstack(X_train, X[15001:15501])
        X_train = np.vstack(X_train, X[16001:16501])
        X_train = np.vstack(X_train, X[17001:17501])
        X_train = np.vstack(X_train, X[18001:18501])
        X_train = np.vstack(X_train, X[19001:19501])'''

        for i in range(1, 20):
            if 1000 * (i + 1) <= 19997:
                #print(y_train.shape)
                X_train = np.concatenate((X_train, X[1000 * i:1000 * i + 500]), axis=0)
                X_test = np.concatenate((X_test, X[1000 * i + 500:1000 * (i + 1)]), axis=0)
                y_train = np.concatenate((y_train, y[1000 * i:1000 * i + 500]), axis=0)
                y_test = np.concatenate((y_test, y[1000 * i + 500:1000 * (i + 1)]), axis=0)
                # print(X_train.shape)
            else:
                #print(X_train.shape)
                X_train = np.concatenate((X_train, X[1000 * i:1000 * i + 500]), axis=0)
                X_test = np.concatenate((X_test, X[1000 * i + 500:]), axis=0)
                y_train = np.concatenate((y_train, y[1000 * i:1000 * i + 500]), axis=0)
                y_test = np.concatenate((y_test, y[1000 * i + 500:]), axis=0)
                # print(X_train.shape)
        return X_train, X_test, y_train, y_test

    def create_TM(self,dictionarylist):
        new_dict = {}
        list_dict = []
        for i in range(0, 20):
            new_dict = {}
            for j in range(i * 500, i * 500 + 501):
                for key in dictionarylist[j].keys():
                    if key not in new_dict.keys():
                        new_dict[key] = 0
                    else:
                        new_dict[key] = new_dict[key] + dictionarylist[j][key]
            list_dict.append(new_dict)
        return list_dict

    def test_data_predict(self,X_test):
        index = np.zeros((X_test.shape[0],))
        for i in range(0, len(X_test)):
            if self.verbose:
                print("Documents Processed: "+ str(i))
            text = X_test[i]
            text = text.split()
            prob_list = np.zeros((20, 1))
            for j in range(0, 20):
                prob_list[j] = self.calculate_probability_extra(text, self.dictionary_list, j)
            index[i] = np.argmax(prob_list)
            # print(prob_list)
            # print(index)
        return index

    def calculate_accuracy(self,y_test, y_pred, categorie_dictionary):
        count = 0
        self.convert_y_to_categorical(y_test, categorie_dictionary)
        #print(y_test)
        for i in range(0, len(y_test)):
            if y_test[i] == y_pred[i]:
                count = count + 1
        return count / len(y_test)

    def convert_y_to_categorical(self,y_test, category_dictionary):
        # unique_set = np.unique(y_test,return_index=True)
        # print(unique_set)
        for i in range(0, len(y_test)):
            y_test[i] = category_dictionary[y_test[i]]

    def calculate_probability(self,text, global_dictionary, category):
        vocab_size = len(global_dictionary[category].keys())
        p_c = 500 / 10000
        probability = 0
        count_of_all_words = sum(global_dictionary[category].values())
        conditional_probability = 1.0
        print("Calculate Probability")
        # print(global_dictionary[category][text[0]])
        # print(count_of_all_words)
        # print(vocab_size)
        # print(p_c)
        # print((count_of_all_words+vocab_size))
        # print(category)
        for i in range(0, len(text)):
            if text[i] in global_dictionary[category].keys():
                # print((global_dictionary[category][text[i]]+1))
                # conditional_probability = conditional_probability*(((global_dictionary[category][text[i]]+1)/(count_of_all_words+vocab_size)))
                conditional_probability = conditional_probability * (global_dictionary[category][text[i]] + 1)
                # print(conditional_probability)
            else:
                conditional_probability = conditional_probability * 1
        # print(conditional_probability*p_c)
        return conditional_probability * p_c

    def calculate_probability_extra(self,text, global_dictionary, category):
        vocab_size = len(global_dictionary[category].keys())
        p_c = 500 / 10000
        probability = 0
        count_of_all_words = sum(global_dictionary[category].values())
        conditional_probability = 1.0
        # print("Calculate Probability")
        # print(global_dictionary[category][text[0]])
        # print(count_of_all_words)
        # print(vocab_size)
        # print(p_c)
        # print((count_of_all_words+vocab_size))
        # print(category)
        for i in range(0, len(text)):
            if text[i] in global_dictionary[category].keys():
                # print((global_dictionary[category][text[i]]+1))
                # conditional_probability = conditional_probability*(((global_dictionary[category][text[i]]+1)/(count_of_all_words+vocab_size)))
                # conditional_probability = conditional_probability*(global_dictionary[category][text[i]]+1)
                # print(conditional_probability)
                word_prob = global_dictionary[category].get(text[i], 0.0) + 0.0001

                probability = probability + math.log(float(word_prob) / float(count_of_all_words))
            else:
                # conditional_probability = conditional_probability*1
                probability = probability
        # print(conditional_probability*p_c)
        # return conditional_probability*p_c
        return probability

if __name__=='__main__':
    print("Enter 1 for training a new model, 2 for using the already trained dictionary which is stored. Using already trained dictionary speeds up computation")
    n = int(input())
    print("Enter 1 to display the processing, 2 to display only the accuracy. Displaying only the accuracy speeds up computation")
    verbose = int(input())
    if n == 1:
        if verbose ==1:
            model = Naive_Bayesian(dictionary=None,verbose=True)
        else:
            model = Naive_Bayesian(dictionary=None)
    else:
        if verbose == 1:
            model = Naive_Bayesian(dictionary=np.load('list_dict_array.npy'),verbose=True)
        else:
            model = Naive_Bayesian(dictionary=np.load('list_dict_array.npy'))
    dataframe = pd.read_csv('data.csv')
    text_vector = dataframe.iloc[:, 1].values
    categorie_vector = dataframe.iloc[:, 2].values
    print("Removing The Stop Words. This might take less than a minute")
    model.remove_Stopwords(text_vector[:19997])
    X_train, X_test, y_train, y_test = model.divide_train_test_data(text_vector, categorie_vector)
    X_train2 = X_train.copy()
    if n == 1:
        print("Training the Model")
        list_dict = model.create_dictionary_other(X_train2)
    index_count = 0
    categorie_dictionary = {}
    for i in range(0, 9997, 500):
        categorie_dictionary[y_test[i]] = index_count
        index_count = index_count + 1
        # print(y_test[i])

    new_list = []
    print("Testing the Model This might take 3 to 4 minutes to get computed")
    y_pred = model.test_data_predict(X_test)
    #print(y_pred)
    accuracy = model.calculate_accuracy(y_test, y_pred, categorie_dictionary)
    print("The accuracy is: "+ str(accuracy*100))
    #print(accuracy*100)
    #list_dict_array = np.asarray(list_dict)
    #np.save('list_dict_array.npy',list_dict_array)
