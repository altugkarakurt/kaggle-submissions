import json
import numpy as np
import pandas as pd

"""-------------------- LOADING DATA --------------------""" 
dataset = json.load(open("train.json", "r"))

# Mine all ingredients and cuisines(labels)
all_ings, all_labels = [], []
for dish in dataset:
	label = dish["cuisine"]
	if(label not in all_labels):
		all_labels.append(label)
	for ing in dish["ingredients"]:
		if(ing not in all_ings):
			all_ings.append(ing)

print("Data processing done")

# Generate bag of words
labels = [all_labels.index(dish["cuisine"]) for dish in dataset]
features = [np.array([all_ings.index(ing) for ing in dish["ingredients"]]) for dish in dataset]

naive_matrix = np.zeros((len(all_ings), len(all_labels)))

for label, feature in zip(labels, features):
	for ing in feature:
		naive_matrix[ing][label] += 1

naive_matrix = np.array([row/sum(row) for row in naive_matrix])

print("Training done")

test_features = json.load(open("test.json", "r"))
test_labels = []
test_ids = [dish["id"] for dish in test_features]

for dish in test_features:
	ing_idxs = [all_ings.index(ing) for ing in dish["ingredients"] if(ing in all_ings)]
	likelihoods = np.sum(naive_matrix[ing_idxs], axis = 0)
	test_labels.append(all_labels[np.argmax(likelihoods)])

df = pd.DataFrame(data = {"id":test_ids, "cuisine":test_labels})
df.to_csv("naive_mle.csv")