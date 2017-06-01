import numpy as np
from sklearn.model_selection import train_test_split


class BFE(object):
    """A class to perform Backward Feature Elimination"""

    def __init__(self):
        pass

    def apply(self, model, metric, x, y, loss=True):
        """
        Performs Backward Feature Elimination using model and metric on the data (x, y)

        Parameters
        ----------
        model : sklearn model
            model to train.

        metric : sklearn metric
            To evaluate the model.

        x : Pandas dataframe
                Training data
        y : Pandas dataframe
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
        x_train, x_val, y_train, y_val = train_test_split(x.values, y.values, test_size=0.4, random_state=2)
        x_columns = x.columns
        nfeatures = len(x.values[0, :])  # Number of initial features.
        ntemp = nfeatures
        self.rankarray = ["" for x in range(nfeatures + 1)]
        model.fit(x_train, y_train)
        m = metric(y_val, model.predict(x_val))
        self.rankarray[-1] = ("All attributes", m)  # Benchmark, when using all attributes.
        for i in range(nfeatures):
            errorarray = np.zeros(ntemp)
            for j in range(ntemp):  # Drop one attribute at a time and evaluate model.
                indarray = [k != j for k in range(ntemp)]
                if (len(indarray) == 1):
                    errorarray[j] = np.nan
                    break
                model.fit(x_train[:, indarray], y_train)
                predictions = model.predict(x_val[:, indarray])
                if loss:
                    errorarray[j] = metric(y_val, predictions)  # The loss.
                else:
                    errorarray[j] = -metric(y_val, predictions)  # The negative, since metric not loss
            worstfeatureindex = np.argmin(errorarray)  # Drop the one we manage the best without.
            self.rankarray[-(i + 2)] = (x_columns[worstfeatureindex], np.abs(np.min(errorarray)))
            x_train = np.delete(x_train, worstfeatureindex, axis=1)
            x_val = np.delete(x_val, worstfeatureindex, axis=1)
            x_columns = x_columns.delete(worstfeatureindex)
            ntemp = ntemp - 1
        return self.rankarray

    def summarize(self):
        """
        Sumarizes the result.
        """
        all = [e[0] for e in self.rankarray]
        acc = [e[1] for e in self.rankarray]
        print("Using all attributes we achieved an " + str(self.metric) + " of", np.abs(acc[-1]))
        for i, _ in enumerate(all[:-2]):
            print("without " + str(all[i:-2]))
            print(str(self.metric) + " of", np.abs(acc[i]))
