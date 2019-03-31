# news_classifier

Implement a general Naïve Bayes classifier for categorizing news
articles as either SPORTS or BUSINESS (i.e., not SPORTS). The dataset 
consists of news articles (http://mlg.ucd.ie/datasets/bbc.html) from a BBC news dataset that
have been labeled as either SPORTS or BUSINESS, and which we have split into a training set
and a testing set.


# Testing

```
java NewsClassifier <modeFlag> <trainFilename> <testFilename>
```
where trainingFilename and testFilename are the names of the training and testing
dataset files, respectively. modeFlag is an integer from 0 to 3, controlling what the program
will output:
0. Prints the number of documents for each label in the training set
1. Prints the number of words for each label in the training set
2. For each instance in test set, prints a line displaying the predicted class and the log
probabilities for both classes
3. Prints the confusion matrix

an example command:
```
java NewsClassifier <modeFlag> train.bbc.txt test.bbc.txt
```
