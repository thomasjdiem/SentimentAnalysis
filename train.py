import pandas as pd
import json
from math import e
from collections import defaultdict

filename = "Tweets.csv"
textname = "text"
classificationname = "airline_sentiment"
TrainingProportion = 0.8
alpha = 0.01

def ExtractFeatures(text):
    features = defaultdict(int)
    features['one'] = 1
    for word in text.split(' '):
        for char in ",.:)(;":
            word = word.replace(char,'')
        features[f'count {word.upper()}'] += 1
    features["length"] = len(text)
    return features

def Predict(text,weights):
    features = ExtractFeatures(text)
    return sum(weights[feature]*features[feature] for feature in features)

def h(x,weights):
    s = sum(x[key]*weights[key] for key in x)
    try:
        return 1/(1+e**-s)
    except OverflowError:
        return 0

   

if __name__ == "__main__":
    df = pd.read_csv(filename)
    weights = defaultdict(int)
    TextSentimentsTrain = {}
    TextFeaturesTrain = {}
    TextSentimentsTest = {}
    TextFeaturesTest = {}
    i = 0

            
    while i < TrainingProportion*len(df.index):
        b = df[classificationname][i]
        if b == "positive":
            a = df[textname][i]
            TextSentimentsTrain[a] = 1
            TextFeaturesTrain[a] = ExtractFeatures(a)
        elif b == "negative":
            a = df[textname][i]
            TextSentimentsTrain[a] = 0
            TextFeaturesTrain[a] = ExtractFeatures(a)
        i += 1
    while i < len(df.index):
        b = df[classificationname][i]
        if b == "positive":
            a = df[textname][i]
            TextSentimentsTest[a] = 1
            TextFeaturesTest[a] = ExtractFeatures(a)
        elif b == "negative":
            a = df[textname][i]
            TextSentimentsTest[a] = 0
            TextFeaturesTest[a] = ExtractFeatures(a)
        i += 1

    for _ in range(200):
        for text in TextFeaturesTrain:
            q = alpha*(TextSentimentsTrain[text] - h(TextFeaturesTrain[text],weights))
            for feature in TextFeaturesTrain[text]:
                weights[feature] += q


    #Test Accuracy
    Total = len(TextSentimentsTest)
    Ncorrect = 0
    for text in TextSentimentsTest:
        a = Predict(text,weights)
        if a > 0.5 and TextSentimentsTest[text] == 1:
            Ncorrect += 1
        elif a < 0.5 and TextSentimentsTest[text] == 0:
            Ncorrect += 1
    
    print(f"Correctly classified {Ncorrect*100/Total:0.2f}% of texts.")

    with open("weights.json","w") as outfile:
        json.dump(weights,outfile)






