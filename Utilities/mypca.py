import numpy as np

class MyPCA(object):
    """A class to compute and work with Pricipal component analysis"""

    def __init__(self):
        pass

    def fit(self, data):
        """
        Calculates eigenpairs and the explaineda variance ratio based on data-

        Parameters
        ----------
        data : Pandas dataframe
            Of form (Number of samples x Attributes)
        """
        self.n = len(data.values[0, :])  # Initial number of attributes/dimension.
        corrmat = data.cov().values  # Covariance matrix.
        eigval, eigvec = np.linalg.eig(corrmat)  # Eigenvalues and Eigenvectors.
        self.eig = [(np.abs(eigval[i]), eigvec[:, i]) for i in range(len(eigval))]  # Group eigen pairs.
        self.eig.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order of eigenvalue.
        sum = 0
        for i in self.eig: sum = sum + i[0]
        self.explained_variance_ratio = [i[0] / sum for i in self.eig]  # Normalize to get the explained variance ratio.

    def fit_transform(self, n_components, data):
        """
        Calculates the projection matrix based on fit() and transforms the data.

        Parameters
        ----------
        n_components : int
            The number of principal components.
        data : Pandas dataframe
            Of form (Number of samples x Attributes)
        Returns
        -------
        Y : numpy array
            The data projected onto the principal axes.

        """
        self.projmatrix = np.hstack((self.eig[i][1].reshape(self.n, 1)  # Construct the projection matrix
                                for i in range(n_components)))  # stack num_components highest eigenvalue arrays.
        Y = data.values.dot(self.projmatrix)
        return Y # Pricipal components of data.

    def transform(self, data):
        """
        Projected onto the principal axes described by the projection matrix.

        Parameters
        ----------
        data : Pandas dataframe
            Of form (Number of Samples x Attributes)
        Returns
        -------
        _ : numpy array
            The data projected onto the principal axes.

        """
        try:
            return data.values.dot(self.projmatrix)
        except:
            print("Something went wrong:")
            print("Is the transformed fitted?")
            print("Are the dimensions of the data correct?")

    def inverse_transform(self, pca_data):
        """
        Inverts the principal components transform.

        Parameters
        ----------
        data_pca : Pandas dataframe
            Of form (Number of Samples x Attributes)
        Returns
        -------
        _ : numpy array
            The data projected back on the original attribute-space.

        """
        try:
            return np.dot(pca_data, self.projmatrix.T) + pca_data.mean()
        except:
            print("Something went wrong:")
            print("Is the transformed fitted?")
            print("Are the dimensions of the data correct?")
