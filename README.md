# Diabetes classification

## 1. Introduction
Data exploration and machine learning classification on the Pima Indians Diabetes Data Set from UCI.

### 1.1 Problem
The Pima indians (Akimel O?odham) of Arizona have the highest rate of diabetes of any population in the world. Hence we should be able by analysing data and using machine learning make predictive indications on how likely a person is to get diabetes.

### 1.2 The data
The Pima Indian diabetes database, donated by Vincent Sigillito, is a collection of medical diagnostic reports. The dataset is publicly available both at [UCI](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) and [Kaggle](https: //www.kaggle.com/uciml/pima-indians-diabetes-database).

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

## 3 Result

### 3.1 The data
Inspecting the data reveals some missing values (values that are physically impossible), labelled as 0’s. We replaced all of these with NaN place-holders instead. The distributions of the attributes are a mix of normal-looking, often with an positive skew, and exponential looking-distributions. The correlations reveal minor positive linear relationships and not anything unexpected.
![alt text](/images/Corr_matrix-eps-converted-to.pdf)






## Author

* **Ludwig Tranheden**
