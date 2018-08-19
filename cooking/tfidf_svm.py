import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

"""-------------------- LOADING DATA --------------------""" 
train_data = json.load(open("../input/cooking_train.json", "r"))
test_data = json.load(open("../input/cooking_test.json", "r"))
train_labels = [sample['cuisine'] for sample in train_data]

train_text = [" ".join(sample['ingredients']).lower() for sample in train_data]
test_text = [" ".join(sample['ingredients']).lower() for sample in test_data]

# Initializations
tfidf = TfidfVectorizer(binary=True)
lbl = LabelEncoder()

"""-------------------- TRAINING --------------------"""
train_features = tfidf.fit_transform(train_text).astype('float16')
train_labels = lbl.fit_transform([sample['cuisine'] for sample in train_data])

clf = SVC(C=100, gamma=1, coef0=1, decision_function_shape=None)
model = OneVsRestClassifier(clf, n_jobs=4)
print("Preprocessing done.")
model.fit(train_features, train_labels)
print("Training done.")

"""-------------------- TESTING --------------------""" 
test_features = tfidf.transform(test_text).astype('float16')
test_labels = model.predict(test_features)
predictions = lbl.inverse_transform(test_labels)

test_ids = [sample["id"] for sample in test_data]
df = pd.DataFrame(data = {"id":test_ids, "cuisine":predictions})
df.to_csv("tfidf_svc.csv", index=False)
print("Testing done.")