import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import zero_one_loss

def gridsearch(model, x_train, y_train, x_val, y_val, param_grid, metric, n_best_grids, loss=True, tiebreaker=zero_one_loss):
        """
        Does a grid search in the parameter space specified.

        Parameters
        ----------
        model : sklearn model
            model who’s parameter space is going to be searched.
        x_train, y_train : numpy arrays
            training data
        x_val, y_val : numpy arrays
            validation data
        param_grid : dictionary
            dictionary over the parameter space to be searched.
        metric : sklearn.metric
            metric to evaluate performance
        n_best_grids: int
            How many parameter combinations to return
        loss : Boolean
            If our metric is an loss or not
        tiebreaker: sklearn.metric
            Tiebreaker metric
        Returns
        -------
        best_score_sorted : numpy array
            The best scores in sorted order
        best_grid_sorted : numpy array
            The best grids in sorted order
        tiebreaker_sorted : numpy array
            The tiebreaker scores

        """
        best_score = [1 if loss else 0 for _ in range(n_best_grids)]
        best_grid = [0 for _ in range(n_best_grids)]
        tie_breaker = [0 for _ in range(n_best_grids)]

        if loss:
                get = lambda x: np.max(x)
                arg = lambda x: np.argmax(x)
                comp = lambda x,y: x < y
        else:
                get = lambda x: np.min(x)
                arg = lambda x: np.argmin(x)
                comp = lambda x,y: x > y 

        for g in ParameterGrid(param_grid):
                model.set_params(**g)
                model.fit(x_train,y_train)
                if comp(metric(y_val,model.predict(x_val)),get(best_score)):
                    best_score[arg(best_score)] = metric(y_val,model.predict(x_val))
                    tie_breaker[arg(best_score)] = tiebreaker(y_val,model.predict(x_val))
                    best_grid[arg(best_score)] = g

        ind =  np.argsort(best_score)
        if not loss:
                ind = np.flip(ind, axis=0)
        best_score_sorted = [best_score[j] for j in ind]
        best_grid_sorted = [best_grid[j] for j in ind]
        tiebreaker_sorted = [tie_breaker[j] for j in ind]
        return best_score_sorted, best_grid_sorted, tiebreaker_sorted
