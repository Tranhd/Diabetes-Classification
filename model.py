from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer,StandardScaler
import pandas as pd
import numpy as np

mods = []
for i in range(1,100): # 100 Trees provides low variance.
    # A parameter combination that were successful for entropy trees.
    mods.append((str(i),DecisionTreeClassifier(criterion='entropy', splitter='best', 
                               max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
                               max_leaf_nodes=22, min_impurity_split=1e-07, class_weight='balanced', presort=False)))
    if(i < 80):
        # A parameter combination that were successful for gini trees.
        mods.append((str(i)+"gi",DecisionTreeClassifier(criterion='gini', splitter='best', 
                               max_depth=7, min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, max_features=None, random_state=None, 
                               max_leaf_nodes=25, min_impurity_split=1e-07, 
                                                        class_weight={0: 1.105, 1: 1.15}, presort=False)))
model = VotingClassifier(estimators=mods, voting='hard', n_jobs=1)

X = pd.read_csv('input/Train.csv')
Y = X['Outcome']
X = X.drop(["Outcome"], axis=1)

model = VotingClassifier(estimators=mods, voting='hard', n_jobs=1)
pipeline = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=0)),
                      ("standardizer", StandardScaler()),
                      ("VotingClassifier", model)])
pipeline.fit(X,Y)