## Description
This is a binary classification competition, aiming to predict whether passengers of Titanic have survived or not based on features like demographics, their ticket information, etc. 

## Data & Preprocessing
The data is given as csv files, which parsed and cleaned using `pandas` DataFrames. For the current submission, I have dropped some of the features in my model; "Name", "Ticket", "Cabin" and "Embarked". Here are some ideas for my future self if I decide to revisit this challenge:
* Although Name itself does not look very useful at first glance, mining fancy prefixes can reveal hints about one's social status. This might be inferred fomr their cabin, too, but this is worth trying.
* Ticket, Cabin abd Embarked can be used (potentially partially joint with the mined social status from Name) to quantize the social class the passenger belongs to. Unfortunately understanding Cabin feature (which is like C2) requires digging and finding where which cabin group was in the ship. Rich people with higher level cabins obviously have an advantage for survival. However, I haven't gone that extra mile. Where they embarked might or might not help with this ranking, as well.

Instead of using the relatively spread out datapoints of Age, I decided to quantize it as (missing, child, adult), encoded as (-1, 0, 1). Hopefully this quantization is refined enough to capture the information in this feature. I also mapped male and female to 0 and 1, because not all learning methods can handle categorical features.

I replaced missing Fare values with the mean of the feature. Doing a smarter decision based on the aforementioned derived "Social Status" feature or maybe even merging them can improve the accuracy.

I joined the SibSp and Parch features as the binary feature isAlone. I think the connection of a passenger's company is not as important as whether he/she was by herself or not.

## Learning Model
I tested multiple models out of the box from `sklearn` such as Gaussian Naive Bayes, SVC, Linear SVM and Random Forest. The latter turned out to be the best fit (at least when the default parameter sets were chosen for all) and that's what I went with. 