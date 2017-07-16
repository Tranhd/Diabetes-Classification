# Diabetes classification

## Introduction
Data exploration and machine learning classification on the Pima Indians Diabetes Data Set from UCI.

### Problem
The Pima indians (Akimel O?odham) of Arizona have the highest rate of diabetes of any population in the world. Hence we should be able by analysing data and using machine learning make predictive indications on how likely a person is to get diabetes.

### The data
The Pima Indian diabetes database, donated by Vincent Sigillito, is a collection of medical diagnostic reports. The dataset is publicly available both at [UCI](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes) and [Kaggle](https: //www.kaggle.com/uciml/pima-indians-diabetes-database).

### File structure
![alt text](/images/struct.png)

## Method
The methodology consists roughly of two chronological steps. First the data is thoroughly explored and dimensionality reduction is applied. Secondly different models are tested, one is chosen and optimized to our problem. The work-flow of these two steps are carried out in two python-notebooks mentioned above. 

### Data exploration
Initial exploration and analysis of the dataset.

#### Distributions and relationship
First the distributions of the different attributes and their relationship to the target are inspected by the means of histograms and correlation.

#### Splitting and standardize the data
At this stage we drop missing data. Before dimensionality reduction (so that no bias is introduced) the data is separated into an training set (Train.csv, 691 instances) and a test set (Test.csv, 77 instances) by a factor of 10%. The test set wont be used at any point, this set will eventually simulate new data used to evaluate the final model. Additionally the training data is then z-normalized, i.e each attribute distribution is transformed to a N(0,1) distribution. We normalize the data to reduce sensitivity to the scale of the attributes and hence eventually reduce sensitivity of potential models.

#### Dimensionality reduction
The dimensionality reduction will be done by the means of two different algorithms with different approaches to the problem, one that is unsupervised and changes the attributes (attribute extraction) and one that is supervised but does not change the attributes (attribute selection).

##### Principal Components Analysis
With respect to the explained variance we reduce the dimensionality. Unfortunately applying PCA means that we are losing the attribute interpretability, something that can be useful for a doctor. The algorithm is implemented in mypca.py.

##### Backward Attribute Elimination
Backward Attribute Elimination (BFE) is an recursive feature elimination. You start with all n attributes. Then some metric is computed on the model n times, one for all combinations of n-1 attributes. The attribute which decreases the error the least is then dropped. We do this recursively finally leaving us with one attribute and hence an importance order. To be sure to prune of the correct attributes we look at the accuracy using two different classifiers, logistic regression and a k-nearest neighbour classifier. The algorithm is implemented in bfe.py.

### Model selection
Testing and deciding on a predictive model. The training data is split into a validation set and a training set (10%).



## Author

* **Ludwig Tranheden**

<!-- Markdeep: --><style class="fallback">body{visibility:hidden;white-space:pre;font-family:monospace}</style><scriptsrc="markdeep.min.js"></script><script src="https://casual-effects.com/markdeep/latest/markdeep.min.js"></script><script>window.alreadyProcessedMarkdeep||(document.body.style.visibility="visible")</script>
