# Diabetes classification

## 1. Introduction
Data exploration and machine learning classification on the Pima Indians Diabetes Data Set from UCI.

### 1.1 Problem
The Pima indians (Akimel Oodham) of Arizona have the highest rate of diabetes of any population in the world. Hence we should be able by analysing data and using machine learning make predictive indications on how likely a person is to get diabetes.

### 1.2 The data
The Pima Indian diabetes database, donated by Vincent Sigillito, is a collection of medical diagnostic reports. The dataset is publicly available both at [UCI](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) and [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

### 1.3 File structure
![alt text](/images/struct.png)

## 2. Method
The methodology consists roughly of two chronological steps. First the data is thoroughly explored and dimensionality reduction is applied. Secondly different models are tested, one is chosen and optimized to our problem. The work-flow of these two steps are carried out in two python-notebooks mentioned above. 

### 2.1 Data exploration
Initial exploration and analysis of the dataset.

#### 2.1.1 Distributions and relationship
First the distributions of the different attributes and their relationship to the target are inspected by the means of histograms and correlation.

#### 2.1.2 Splitting and standardize the data
At this stage we drop missing data. Before dimensionality reduction (so that no bias is introduced) the data is separated into an training set (Train.csv, 691 instances) and a test set (Test.csv, 77 instances) by a factor of 10%. The test set wont be used at any point, this set will eventually simulate new data used to evaluate the final model. Additionally the training data is then z-normalized, i.e each attribute distribution is transformed to a N(0,1) distribution. We normalize the data to reduce sensitivity to the scale of the attributes and hence eventually reduce sensitivity of potential models.

#### 2.1.3 Dimensionality reduction
The dimensionality reduction will be done by the means of two different algorithms with different approaches to the problem, one that is unsupervised and changes the attributes (attribute extraction) and one that is supervised but does not change the attributes (attribute selection).

##### 2.1.3.1 Principal Components Analysis
With respect to the explained variance we reduce the dimensionality. Unfortunately applying PCA means that we are losing the attribute interpretability, something that can be useful for a doctor. The algorithm is implemented in mypca.py.

##### 2.1.3.2 Backward Attribute Elimination
Backward Attribute Elimination (BFE) is an recursive feature elimination. You start with all n attributes. Then some metric is computed on the model n times, one for all combinations of n-1 attributes. The attribute which decreases the error the least is then dropped. We do this recursively finally leaving us with one attribute and hence an importance order. To be sure to prune of the correct attributes we look at the accuracy using two different classifiers, logistic regression and a k-nearest neighbour classifier. The algorithm is implemented in bfe.py.

### 2.2 Model selection
Testing and deciding on a predictive model. The training data is split into a validation set and a training set (10%).

#### 2.2.1 Comparing classification models with K-fold cross validation
A baseline for the problem is established by using cross validation and a few classification models. If we use k-fold cross validation, since our dataset is quite small, a splitting of our data into to few folds could introduce a substantial bias. On the other hand if we chose k to large we will have a lot of variance. With the small data-set in the back of out head we chose k to be moderately large, 18. The models used are Logistic regression, Gaussian Naive Bayes, 4-Neighbors, Decision tree with Gini impurity criterion, Decision tree with Information gain criterion and Support vector machine.

#### 2.2.2 Evaluate models using Confusion Matrixes and validation data
A closer look on how the different models perform on the validation data and if the confusion matrixes reveals any further information about the models.

#### 2.2.3 Inspecting and optimizing the model
With respect to the result of the previous two sections we should have reduced the number of models to one and decided on a transform of the data. Next we inspect how the model varies across different parameter choices. We analyse if there is any substantial variance or bias in the model. This is then followed up by a complete parameter search (Gridsearch) to find the optimal choice. The algorithm is implemented in utils.py and is called gridsearch. The algorithm supports, in contrast to the sklearn algorithm, evaluation of arbitrary metric on validation data to find the optimal parameter combination as well as an tiebreaker metric.

#### 2.2.4 Pipeline
To automate the work-flow the final model is incorporated into an pipeline that can handle eventual missing values and performs the necessary transformations.

#### 2.2.5 Test
Finally we test the final model on the held out test-set.

## 3. Result

### 3.1 The data
Inspecting the data reveals some missing values (values that are physically impossible), labelled as 0’s. We replaced all of these with NaN place-holders instead. The distributions of the attributes are a mix of normal-looking, often with an positive skew, and exponential looking-distributions. The correlations reveal minor positive linear relationships and not anything unexpected.
![alt text](/images/Corr_matrix.png)

#### 3.1.1 Principal component analysis
As the explained variance ratio is inspected we note that most of the components will be needed to account for all the variance. Using two components a scatter plot of the data is viewed to see how well the classes is separated, not very good in a linear sense. 5 components is chosen, that will cover almost 100 percent of the variance.
![alt text](/images/explained_variance.png)

#### 3.1.2 Backward Attribute Elimination
Both logistic regression and the 6-nearest neighbour classifier (euclidean distance) agrees on the importance of the attributes to some extent. The accuracy increase if we drop two or three attributes depending on the set, taking the union of these two subsets we get the new set: Glucose, Pregnancies, BMI, SkinThickness, Insulin and BloodPressure Hence we drop DiabetesPedi- greeFunction and Age.

### 3.2 Models
Evaluating the models using 18-fold cross validation and the three different versions of the dataset (Raw data, PCA data and BFE data) does not reveal any significantly better accuracy for any of the data sets, different models perform differently across the datasets. Logistic regression seems to be the best performing model.
![alt text](/images/modelsvsdata.png)

#### 3.2.1 Confusion Matrixes
Almost all models predict the unseen validation worse than the 18-fold cross validation score revealed. The models that classifies the validation set significantly better is the decision trees. As we inspect the confusion matrix we note that most of the models have almost identical predictions when it comes to instances with true negative outcomes (Figure 4). Where they differ and where some of the the decision trees is superior is when we try to predict true positive outcomes. Many of the models are worse than random guessing. This could be the False positive paradox. Basically since the majority of the instances has outcome not diabetes our models will favour predicting not diabetes. This leads to another paradox, namely the accuracy paradox. Basically it means that because of the imbalance in outcomes predictive models with a given accuracy might have greater predictive capability than a model with higher accuracy. Hence we exchange accuracy as metric in favour of AUC - The area under the ROC (receiver operating characteristic) curve. The ROC curve is the true positive rate (TPR) against the false positive rate (FPR) at various thresholds/ranks for the instances. The area under the curve measures discrimination, the ability to correctly classify those with and without diabetes in our case.

Evaluating our models using the validation set and AUC instead of accuracy the decision trees using all the data is superior.
![alt text](/images/confusion.png)

#### 3.2.2 Explore the model
As familiar decision trees have high variance. As such we now look at the effect of increasing the maximum depth of the tree and at the same time inspecting the deviation of 100 trees at that depth. The maximum leaf nodes is fixed to avoid overfitting. The resulting plot does not imply any overfitting, by inspection the best model is the decision tree with entropy criterion however the gini trees seem to handle false positives better.
![alt text](/images/xploredepth.png)

#### 3.2.3 Optimizing the parameters
By using a grid search the entire parameter space is explored with AUC as the metric to optimize and zero-one-loss as an tiebreaker. The best trees chosen usually have the same AUC score and zero-one-loss, most often 0.8875. Every time we run the parameter search another combination of parameters will be the best because of the intrinsic variance of decision trees, though most of the best trees performs well because of the natural low bias of decision trees. However reusing the best parameter combination yields different result every time we train a new tree because of the intrinsic variance, sometimes the predictions are really good and sometimes worse. Remember that the trees are tuned to the validation set and might, probably wont, generalize good to new data.

There is some pattern to what sort of combinations work for the problem. For the entropy criterion trees the best performing ones the maximum depth is between 5 and 7, the maximum number of leaves between 22 and 26 and using balanced weighting for the classes. The balanced weighting adjust the weights for the classes inversely proportional to class frequencies, basically counteracting the false positive hypothesis paradox mentioned earlier. For the gini criterion trees the maximum depth is around 7, the maximum number of leaves varies a lot but tend to approach higher values (≈25) and using the proposed 1.105, 1.15 weighting for the classes.

#### 3.2.4 Ensembles
We could use an ensemble to decrease variance and try to make our model generalize well. Both Bootstrap aggregating and ran- dom forest is tried using the generally best decision tree: entropy, max_depth=5, max_leaf_nodes=22, and class_weight=’balanced’. It should theoretically provide the stability we need and reduce variance. Both the algorithms provide the decrease in variance but does instead introduce an substantial unacceptable bias. Looking back at when we compared entropy and gini criterion and looked at the effect of the depth and variance. Remember that the gini trees generally had few false positives while entropy trees had few false negatives. An viable hypothesis might be that the two complement each other, and because we in the ensembles above only use the one or the other they doesn’t improve our predictive capability.
![alt text](/images/bagging.png)
![alt text](/images/randfor.png)

#### 3.2.5 Final model: Hard Voting Classifier
The voting classifier consists of 100 trees and a 5:4 ratio of entropy and gini trees since the gini trees showed less potential in the grid search. The classification is done using a majority vote rule. The parameters for the entropy tree is as above and for the gini tree max_depth=7, max_leaf_nodes=25, class_weight={0: 1.105, 1: 1.15}. This yields an amazing result, AUC of nearly 89 for the validation set and 93 for the training set. That’s not a gigantic difference and hopefully the constraints on the trees have prevented the model from overfitting on the training data.

#### 3.2.6 Pipeline
Before evaluating the model on the test set we train our model on the entire training dataset with some simple mean-imputing. We also put it into a pipeline to automate the workflow. The model is found in model.py.

### 3.3 Test
Evaluating our model on the test set without missing values gives an accuracy of 0.88 and AUC of 0.89. 
![alt text](/images/test_conf.png)

## 4. Discussion
I think that the best accuracy I’ve seen on this dataset was an optimized gradient boosting classifier with approximately 86%. On the other hand this classifier could handle missing values and used 20% of the data as an testing set. So an future project could be to by further looking in to bagging, construct something using both entropy and gini and that samples both the features and data with replacement making it prone to missing values.

Also since dimensionality reduction clearly did not work out feature-construction might be something that could boost the predictive capability of an model.

## 5. Conclusion
The model does classify the new test data with really good accuracy but do not have the interpretability that you might want or any sophisticated way to handle missing values.

## Author

* **Ludwig Tranheden**
