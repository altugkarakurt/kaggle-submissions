## Description
This is a 20-class classification competition, aiming to predict the cuisine a recipe belongs to, based on the list on ingredients.

## Data & Preprocessing
The data is given as a JSON file as list of dictionaries. As you can imagine there are a bunch of redundant ingredients that are not informative for distinguishing the cuisine of a dish, such as salt, onion or garlic. On the other hand, some of them are almsot guaranteed to distinctive like garam masala. Hence, the naive binary bag-of-words approach (which was my first attempt) contains a lot of noise as it weighs salt and garam masala equally important. 

There is a lot of potential for some feature engineering and dimension redunction here using methods like PCA (or probably even fancier techniques), but I went with what was readily available out of the box: TFIDF Vectorization, thanks to `sklearn`. I never heard about this method, but got a lot of mentions of it when I looked up ways to engineer features from plain text and extract important keywords. TFIDF stands for "Term Frequency Inverse Document Frequency", the former half refers to counting the occurences of the erm in the text (which is binary since ingredient list is a set) and the latter half normalizes based on how rare it is across all classes, by taking the log of reciprocal of the count of the number of occurences in different classes. TFIDF is the multiplication of the two. We are basically after IDF to suppress the contribution of salt and boost garam masala, but since TF is 1 anyway, we can simply use TFIDF.

The feature engineering was basically joining the list of ingredients as space separated texts for each dish and running TFIDF to get a feature vector the size of unique ingredients, a.k.a a smarter and weighted bag-of-words.

## Learning Model
I basically threw it on a bunch of classification algorithms to see what would stick and a SVC with a large penalty parameter seemed to do the best job. Most of the challenge with this task is the feature engineering portion.