import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

"""-------------------- UTIL FUNCTIONS -------------------- """
def evaluate(test_features, test_labels, model):
    evals = [] # Binary list of the tree's accuracy
    for sample, true_label in zip(test_features, test_labels):
        prediction = model.predict([sample])
        evals.append(int(prediction == true_label))
    accuracy = (sum(evals) * 1.0)/len(evals)
    return accuracy

def clean_data(features):
	# Transform "Sex" attribute to numerical ["male", "female"] --> [1, 0]
	#features.loc[(features['Sex'] == "male"), "Sex"] = 1
	#features.loc[(features['Sex'] == "female"), "Sex"] = 0
	features["Sex"] = features["Sex"].map({'female': 1, 'male': 0})

	# Quantize ages as (missing, child, adult) as (-1, 0, 1) 
	features.loc[np.isnan(features['Age']), "Age"] = -1 # Missing
	features.loc[(features['Age'] >= 18), "Age"] = 1 # Adult
	features.loc[(features['Age'] > 0) & (features['Age'] < 18), "Age"] = 0 # Children

	# Replace NaNs with the average of respective column
	features.loc[np.isnan(features['Fare']), "Fare"] = np.mean(features["Fare"]) # Missing

	# Replaces SibSb and Parch columns with binary isAlone column
	isAlone = [1 if(parch+sibsp > 0) else 0
		for parch, sibsp in zip(features["Parch"], features["SibSp"])]
	features = features.drop(columns=["Parch", "SibSp"])
	features = features.add(pd.DataFrame({"isAlone":isAlone}))

def test_submission(model, model_name):
	features = pd.read_csv("test.csv")
	passenger_ids = np.array(features["PassengerId"])
	features = features.drop(columns=ignored_columns)
	clean_data(features)
	labels = np.zeros_like(passenger_ids)

	for idx, sample in enumerate(features.values):
		labels[idx] = model.predict([sample])

	df = pd.DataFrame({"PassengerId":passenger_ids, "Survived":labels})
	df.to_csv(("%s_titanic.csv" % (model_name)), index=False)

"""-------------------- LOAD & CLEAN DATA -------------------- """
ignored_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]
features = pd.read_csv("train.csv")
features = features.drop(columns=ignored_columns)
labels = np.array(features["Survived"])
features = features.drop(columns=["Survived"])

# Data cleaning
# TODO: Include Cabin and transform them to numericals
clean_data(features)

"""-------------------- TRY DIFFERENT SKLEARN ALGORITHMS -------------------- """
### Support Vector Machines
clf = SVC()
clf.fit(features, labels)
train_accuracy = evaluate(features.values, labels, clf)
print("Training loss of SVM: %1.4f" % (train_accuracy))
test_submission(clf, "svm")

### Linear Kernel Support Vector Machines
clf = LinearSVC()
clf.fit(features, labels)
train_accuracy = evaluate(features.values, labels, clf)
print("Training loss of Linear SVM: %1.4f" % (train_accuracy))
test_submission(clf, "linear_svm")

### Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(features, labels)
train_accuracy = evaluate(features.values, labels, clf)
print("Training loss of Gaussian Naive Bayes: %1.4f" % (train_accuracy))
test_submission(clf, "gaussian_naive_bayes")

### Gaussian Naive Bayes
clf = RandomForestClassifier()
clf.fit(features, labels)
train_accuracy = evaluate(features.values, labels, clf)
print("Training loss of Random Forest: %1.4f" % (train_accuracy))
test_submission(clf, "random_forest")