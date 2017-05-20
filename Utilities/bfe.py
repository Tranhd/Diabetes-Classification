import numpy as np

class BFE(object):
    """A class to perform Backward Feature Elimination"""

    def __init__(self):
        pass

    def apply(self, model, metric, trainO, test,loss = True):
        """
        Performs Backward Feature Elimination using model and metric on the data (TrainO, Test)
    
        Parameters
        ----------
        model : sklearn model
            model to train.
    
        metric : sklearn metric
            To evaluate the model.
    
        train0 : Pandas dataframe
                Training data
        test : Pandas dataframe
                ground truth.
        loss : Boolean
                If we have an loss or reverse metric as accuracy.
        Returns
        -------
        rankarray : numpy array
            tuples of (Attribute, metric without it and all before) ordered in the order
            they were removed.
    
        """
        self.metric = metric
        train = trainO.copy(deep=True) # Copy the training data since we will be modifying it.
        train_columns = train.columns
        nfeatures = len(train.values[0,:])  # Number of initial features.
        ntemp = nfeatures
        self.rankarray = ["" for x in range(nfeatures+1)]
        model.fit(train.values,test.values)
        m = metric(test.values, model.predict(train.values))
        self.rankarray[-1] = ("All attributes", m)   # Benchmark, when using all attributes.
        for i in range(nfeatures):
            errorarray = np.zeros(ntemp)
            for j in range(ntemp):              # Drop one attribute at a time and evaluate model.
                indarray = [k != j for k in range(ntemp)]
                if(len(indarray) == 1):
                    errorarray[j] = np.nan
                    break
                model.fit(train.values[:,indarray], test.values)
                predictions = model.predict(train.values[:,indarray])
                if loss:
                    errorarray[j] = metric(test.values, predictions) # The loss.
                else:
                    errorarray[j] = -metric(test.values, predictions) # The negative, since metric not loss
            worstfeatureindex = np.argmin(errorarray)   # Drop the one we manage the best without.
            self.rankarray[-(i+2)] = (train_columns[worstfeatureindex], np.abs(np.min(errorarray)))
            train.drop([train_columns[worstfeatureindex]],axis=1,inplace=True)
            train_columns = train_columns.delete(worstfeatureindex)
            ntemp = ntemp - 1
        return self.rankarray
    
    
    def summarize(self):
        """
        Sumarizes the result.
        """
        all = [e[0] for e in self.rankarray]
        acc = [e[1] for e in self.rankarray]
        print("Using all attributes we achieved an " + str(self.metric) + " of", np.abs(acc[-1]))
        for i,_ in enumerate(all[:-2]):
            print("without " +  str(all[i:-2])) 
            print(str(self.metric) + " of", np.abs(acc[i]))
