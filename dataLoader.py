from nltk.stem import WordNetLemmatizer, PorterStemmer
import sklearn
import re
import numpy as np
import nltk

nltk.download('wordnet')

def loadData(path):
    """
    pre-processing data and splits data into 80% train, 20% test
    :parameter: data file path              TYPE - string
    :return: training and testing datasets. TYPE - nparray
    """

    x,y,vocabularyList =splitFile(path)
    splitData(x,y,0.2)

    return x,y,vocabularyList

def splitFile(path):
    """
    splits text file into not cleaned word array
    :param path: SMSSpamCollection             TYPE - string
    :return: not cleaned words array           TYPE - list
    """
    lemma = WordNetLemmatizer()
    x,y,vocabularyList = [],[],[]
    stem = PorterStemmer()

    """
    clean words
    """

    with open(path,'r',encoding ='utf-8',errors='ignore') as file:
        for line in file:
            temp =[]

            words = line.split()
            for word in words:

                temp.append(re.sub('[^a-z0-9]','',word.lower()))

            afterLem = [lemma.lemmatize(i) for i in temp]
            afterStem = [stem.stem(j) for j in afterLem]



            x.append(afterStem[1:])
            if afterStem[:1][0] =='ham':
                y.append(1)
            else:
                y.append(0)


            vocabularyList.extend(afterStem[1:])

    return x,y,vocabularyList



def splitData(x,y,testRatio):
    """
    split data set by ratio
    :param x:
    :param y:
    :param testRatio:
    :return:
    """

    x_test = x[:int(len(x)*testRatio)]
    y_test = y[:int(len(x)*testRatio)]
    x_train = x[int(len(x)*testRatio):]
    y_train = y[int(len(x)*testRatio):]

    return x_train,y_train,x_test,y_test
# if __name__ =='__main__':
#     loadData('SMSSpamCollection')
