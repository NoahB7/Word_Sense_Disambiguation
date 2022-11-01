import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
import sys


def extract_contexts(file, train_test_files,window_size):
    data = file.readlines()
    for line in data:
        
        words = nltk.tokenize.word_tokenize(line)
        words = [word.lower() for word in words if word.isalpha()]
        for index, word in enumerate(words):
            
            if word in train_test_files.keys():
                train_test_files[word].write(' '.join(words[lower(index,window_size):upper(index, window_size, len(words))]))
                train_test_files[word].write("\n")
    
def lower(i,window_size):
    if(i - window_size < 0):
        return 0
    else:
        return i - window_size
    
def upper(i,window_size,max):
    if(i + window_size + 1 > max):
        return max
    else:
        return i + window_size + 1
    
# closes files to finish their respective writes
def finish_writes(train_test_files):
    for item in train_test_files.items():
        item[1].close()
        
def bayes_disambiguate(word1,word2, disambiguation_phrase, sense_counts, sense_totals, base_probs):
    disambiguation_phrase = disambiguation_phrase.split()
    for index, word in enumerate(disambiguation_phrase):
        if word == word1 or word == word2:
            disambiguation_phrase.pop(index)
            break
            
    pseudoword = word1+word2
    sense1prob = base_probs[pseudoword][word1]
    sense2prob = base_probs[pseudoword][word2]
    
    for word in disambiguation_phrase:
        if word in sense_counts[pseudoword][word1]:
            sense1prob *= (sense_counts[pseudoword][word1][word] + 1)/(sense_totals[pseudoword][word1] + len(sense_counts[pseudoword][word1]))
        else:
            sense1prob *= 1/(sense_totals[pseudoword][word1] + len(sense_counts[pseudoword][word1]))

        if word in sense_counts[pseudoword][word2]:
            sense2prob *= (sense_counts[pseudoword][word2][word] + 1)/(sense_totals[pseudoword][word2] + len(sense_counts[pseudoword][word2]))
        else:
            sense2prob *= 1/(sense_totals[pseudoword][word2] + len(sense_counts[pseudoword][word2]))

    if sense1prob > sense2prob:
        return word1
    
    return word2
        
def train_bayes(train_test_files, balanced):
    sense_counts = {}
    sense_totals = {}
    test_data = {}
    base_probs = {}
    
    pseudo_word_vec = []
    for key in train_test_files.keys():
        pseudo_word_vec.append(key)
#         goofy work around since it wouldnt be the same size since bike is in two pseudo words so the list is size 11 and cant be split in a 6 x 2
        if key == 'manufacturer':
            pseudo_word_vec.append('bike')
    pseudo_word_vec = np.array(pseudo_word_vec).reshape(6,2)
    
    for i in range(len(pseudo_word_vec)):
        count_vec = {}
        total_vec = {}
        base_prob = {}
        
        sensefile1 = pseudo_word_vec[i][0] + '_data.txt'
        sensefile2 = pseudo_word_vec[i][1] + '_data.txt'
        
        sense1 = open(sensefile1).readlines()
        sense2 = open(sensefile2).readlines()
        
        if balanced:
            if len(sense1) > len(sense2):
                sense1 = sense1[:len(sense2)]
            else:
                sense2 = sense2[:len(sense1)]
        
        sense1_train, sense1_test, throwaway1, throwaway2 = train_test_split(sense1, np.zeros(len(sense1)), test_size = 0.2, random_state=42)
        test_data[pseudo_word_vec[i][0]] = sense1_test

        sense2_train, sense2_test, throwaway1, throwaway2 = train_test_split(sense2, np.zeros(len(sense2)), test_size = 0.2, random_state=42)
        test_data[pseudo_word_vec[i][1]] = sense2_test

        
        len1 = len(sense1_train)
        len2 = len(sense2_train)
        sense1counts = {}
        sense2counts = {}
        sense1total = 0
        sense2total = 0

        for line in sense1_train:
            words = line.split()
            for word in words:
                sense1total += 1
                if word in sense1counts.keys():
                    sense1counts[word] = sense1counts[word]+1
                else:
                    sense1counts[word] = 1

        for line in sense2_train:
            words = line.split()
            for word in words:
                sense2total += 1
                if word in sense2counts.keys():
                    sense2counts[word] = sense2counts[word]+1
                else:
                    sense2counts[word] = 1
                    
        count_vec[pseudo_word_vec[i][0]] = sense1counts
        count_vec[pseudo_word_vec[i][1]] = sense2counts
        sense_counts[pseudo_word_vec[i][0] + pseudo_word_vec[i][1]] = count_vec

            
        total_vec[pseudo_word_vec[i][0]] = sense1total
        total_vec[pseudo_word_vec[i][1]] = sense2total
        sense_totals[pseudo_word_vec[i][0] + pseudo_word_vec[i][1]] = total_vec
        
        
        base_prob[pseudo_word_vec[i][0]] = len1/(len1+len2)
        base_prob[pseudo_word_vec[i][1]] = len2/(len1+len2)
        base_probs[pseudo_word_vec[i][0] + pseudo_word_vec[i][1]] = base_prob
    return sense_counts,sense_totals,base_probs,test_data


def test_accuracy(sense_counts,sense_totals,base_probs,test_data):
    pseudowords = [['night','seat'],
                   ['kitchen', 'cough'],
                   ['car', 'bike'],
                   ['manufacturer', 'bike'],
                   ['big', 'small'],
                   ['huge', 'heavy']]
    totalcorrect = 0
    total = 0
    for i in range(len(pseudowords)):
        correct = 0
        total += len(test_data[pseudowords[i][0]])
        total += len(test_data[pseudowords[i][1]])
        for line in test_data[pseudowords[i][0]]:
            
            pred = bayes_disambiguate(pseudowords[i][0], pseudowords[i][1], line, sense_counts,sense_totals,base_probs)
            if pred == pseudowords[i][0]:
                correct += 1
                totalcorrect += 1
        print(f'Accuracy for pseudoword: {pseudowords[i][0]}{pseudowords[i][1]}, predicting: {pseudowords[i][0]} : {round((correct / len(test_data[pseudowords[i][0]]))*100,2)}%')
        
        correct = 0
        for line in test_data[pseudowords[i][1]]:
            
            pred = bayes_disambiguate(pseudowords[i][0], pseudowords[i][1], line, sense_counts,sense_totals,base_probs)
            if pred == pseudowords[i][1]:
                correct += 1
                totalcorrect += 1
        print(f'Accuracy for pseudoword: {pseudowords[i][0]}{pseudowords[i][1]}, predicting: {pseudowords[i][1]} : {round((correct / len(test_data[pseudowords[i][1]]))*100,2)}%')
    print(f'Overall accuracy for all pseudowords: {round((totalcorrect/total)*100,2)}%')
        


def initialize_files():
    return {'night' : open('night_data.txt', 'w'), 
                        'seat' : open('seat_data.txt', 'w'), 
                        'kitchen' : open('kitchen_data.txt', 'w'), 
                        'cough' : open('cough_data.txt', 'w'),
                        'car' : open('car_data.txt', 'w'), 
                        'bike' : open('bike_data.txt', 'w'),
                        'manufacturer' : open('manufacturer_data.txt', 'w'),
                        'bike' : open('bike_data.txt', 'w'),
                        'big' : open('big_data.txt', 'w'),
                        'small' : open('small_data.txt', 'w'),
                        'huge' : open('huge_data.txt', 'w'),
                        'heavy' : open('heavy_data.txt', 'w')}

if __name__ == '__main__':
    if sys.argv[4] == 'true':
        nltk.download('punkt')
        
    filename = sys.argv[1]
    file = open(filename, 'r')
    window_size = int(sys.argv[2])
    balanced = False
    btext = 'unbalanced'
    if sys.argv[3] == 'balanced':
        balanced = True
        btext = 'balanced'
    
    print(f'Window size of {window_size}, {btext} data for each word in pseudoword')
    print('--------------------------------------------------------------')
    train_test_files = initialize_files()
    extract_contexts(file, train_test_files,window_size)
    finish_writes(train_test_files)
    sense_counts, sense_totals, base_probs, test_data = train_bayes(train_test_files, balanced)
    test_accuracy(sense_counts,sense_totals,base_probs,test_data)
    