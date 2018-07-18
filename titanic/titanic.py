import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC

"""-------------------- UTIL FUNCTIONS -------------------- """
def evaluate(test_features, test_labels, model):
    evals = [] # Binary list of the tree's accuracy
    for sample, true_label in zip(test_features, test_labels):
        prediction = model.predict([sample])
        evals.append(int(prediction == true_label))
    accuracy = (sum(evals) * 1.0)/len(evals)
    return accuracy

"""-------------------- LOAD & CLEAN DATA -------------------- """
ignored_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
dataset = pd.read_csv("train.csv")
features = dataset[[col for col in dataset.columns if((col!="Survived") and (col not in ignored_columns))]]
labels = np.array(dataset["Survived"])

# TODO: Include Cabin and transform them to numericals
# Transform "Sex" attribute to numerical ["male", "female"] --> [1, 0]
features = features.replace({"male":1, "female":0})

# Replace NaNs with the average of respective column
features = features.fillna(value={col:np.mean(features[col]) for col in features.columns})

"""-------------------- TRY DIFFERENT SKLEARN ALGORITHMS -------------------- """
clf = SVC()
clf.fit(features, labels)
train_accuracy = evaluate(features.values, labels, clf)
print("Training loss of SVM: %1.4f" % (train_accuracy))

