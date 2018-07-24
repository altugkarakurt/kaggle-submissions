import json
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

"""-------------------- LOADING DATA --------------------""" 
dataset = json.load(open("../input/train.json", "r"))

# Mine all ingredients and cuisines(labels)
all_ings, all_labels = [], []
for dish in dataset:
	label = dish["cuisine"]
	if(label not in all_labels):
		all_labels.append(label)
	for ing in dish["ingredients"]:
		if(ing not in all_ings):
			all_ings.append(ing)

print("Data processing done.")

# Generate bag of words
labels = [all_labels.index(dish["cuisine"]) for dish in dataset]
feature_idxs = [np.array([all_ings.index(ing) for ing in dish["ingredients"]]) for dish in dataset]
features = []

for idxs in feature_idxs:
	temp = np.zeros(len(all_ings))
	temp[idxs] = 1
	features.append(temp)

print("Training feature extraction done.")

test_dataset = json.load(open("../input/test.json", "r"))
test_feature_idxs = [np.array([all_ings.index(ing) 
					 for ing in dish["ingredients"] if(ing in all_ings)])
					 for dish in test_dataset]
test_ids = [dish["id"] for dish in test_dataset]
test_features = []
for idxs in test_feature_idxs:
	temp = np.zeros(len(all_ings))
	temp[idxs] = 1
	test_features.append(temp)
print("Test feature extraction done.")

"""-------------------- Random Forest --------------------"""
clf = RandomForestClassifier(n_estimators=200, max_depth=5000).fit(features, labels)
test_labels = []

print("RF: Training done.")

for dish in test_features:
	prediction = all_labels[clf.predict([dish])[0]]
	test_labels.append(prediction)

print("RF: Testing done.")

df = pd.DataFrame(data = {"id":test_ids, "cuisine":test_labels})
df.to_csv("random_forest.csv", index=False)

"""-------------------- Random Forest --------------------"""
"""
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
est_cnts = range(100, 600, 100)
depths = [2500, 5000]#range(2000,10000,2000)
for est_cnt in est_cnts:
	for depth in depths:
		clf = RandomForestClassifier(n_estimators=est_cnt, max_depth=depth).fit(X_train, y_train)
		print("Random Forest - depth:%d, est_cnt:%d, score:%1.4f" % ( depth, est_cnt, clf.score(X_test, y_test)))
"""
