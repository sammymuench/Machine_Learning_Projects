import numpy as np
from scipy.optimize import minimize


# NOTE: follow the docstrings. In-line comments can be followed, or replaced.
#       Hence, those are the steps, but if it does not match your approach feel
#       free to remove.

def linear_kernel(X1, X2):
    """    Matrix multiplication.

    Given two matrices, A (m X n) and B (n X p), multiply: AB = C (m X p).

    Recall from hw 1. Is there a more optimal way to implement using numpy?
    :param X1:  Matrix A
    type       np.array()
    :param X2:  Matrix B
    type       np.array()

    :return:    C Matrix.
    type       np.array()
    """
    return np.matmul(X1, X2.T)


def nonlinear_kernel(X1, X2, sigma=0.5):
    """
     Compute the value of a nonlinear kernel function for a pair of input vectors.

     Args:
         X1 (numpy.ndarray): A vector of shape (n_features,) representing the first input vector.
         X2 (numpy.ndarray): A vector of shape (n_features,) representing the second input vector.
         sigma (float): The bandwidth parameter of the Gaussian kernel.

     Returns:
         The value of the nonlinear kernel function for the pair of input vectors.

     """
    # Compute the Euclidean distance between the input vectors
    dist = np.linalg.norm(X1 - X2)

    # Compute the value of the Gaussian kernel function
    # Return the kernel value
    return np.exp(-dist ** 2 / (2 * sigma ** 2))


def objective_function(X, y, a, kernel):
    """
    Compute the value of the objective function for a given set of inputs.

    Args:
        X (numpy.ndarray): An array of shape (n_samples, n_features) representing the input data.
        y (numpy.ndarray): An array of shape (n_samples,) representing the labels for the input data.
        a (numpy.ndarray): An array of shape (n_samples,) representing the values of the Lagrange multipliers.
        kernel (callable): A function that takes two inputs X and Y and returns the kernel matrix of shape (n_samples, n_samples).

    Returns:
        The value of the objective function for the given inputs.
    """
    # TODO: implement
    
    # Reshape a and y to be column vectors
    
    y = np.reshape(y, (-1, 1))
    a = np.reshape(a, (-1, 1))

    # The first term is the sum of all Lagrange multipliers

    first_term = np.sum(a)
    
    # The second term involves the kernel matrix, the labels and the Lagrange multipliers

    a_mat = np.dot(a, a.T)
    y_mat = np.dot(y, y.T)
    K = kernel(X, X)
    second_term = -0.5 * np.sum(a * y * a.T * y.T * K)

    return first_term + second_term


class SVM(object):
    """
         Linear Support Vector Machine (SVM) classifier.

         Parameters
         ----------
         C : float, optional (default=1.0)
             Penalty parameter C of the error term.
         max_iter : int, optional (default=1000)
             Maximum number of iterations for the solver.

         Attributes
         ----------
         w : ndarray of shape (n_features,)
             Coefficient vector.
         b : float
             Intercept term.

         Methods
         -------
         fit(X, y)
             Fit the SVM model according to the given training data.

         predict(X)
             Perform classification on samples in X.

         outputs(X)
             Return the SVM outputs for samples in X.

         score(X, y)
             Return the mean accuracy on the given test data and labels.
         """

    def __init__(self, kernel=nonlinear_kernel, C=1.0, max_iter=1e3):
        """
        Initialize SVM

        Parameters
        ----------
        kernel : callable
          Specifies the kernel type to be used in the algorithm. If none is given,
          ‘rbf’ will be used. If a callable is given it is used to pre-compute 
          the kernel matrix from data matrices; that matrix should be an array 
          of shape (n_samples, n_samples).
        C : float, default=1.0
          Regularization parameter. The strength of the regularization is inversely
          proportional to C. Must be strictly positive. The penalty is a squared l2
          penalty.
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.a = None
        self.w = None
        self.b = None

    def y_constraint(self, alpha, y):
        # Constraint 2: sum(alpha[i] * y[i]) = 0 for all alphas
        return np.dot(alpha, y)

    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vectors, where n_samples is the number of samples and n_features 
          is the number of features. For kernel=”precomputed”, the expected shape 
          of X is (n_samples, n_samples).

        y : array-like of shape (n_samples,)
          Target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
          Fitted estimator.
        """
        #Define the constraints for the optimization problem

        constraints = [{'type': 'ineq', 'fun': lambda x: x}, {'type': 'eq', 'fun': self.y_constraint, 'args': (y,)}]
        
        #Use minimize from scipy.optimize to find the optimal Lagrange multipliers
        
        self.a = np.zeros(y.shape)
        res = minimize(lambda a: -objective_function(X, y, a, self.kernel), self.a, constraints=constraints)
        self.a = np.array(res.x)
        
        #Substitute into dual problem to find weights
        num_obs = X.shape[0]
        support_vectors = np.array(range(y.shape[0]))[self.a>1e-8]
        weights = [sum([self.a[support] * y[support] * X[support, weight] for support in support_vectors]) for weight in range(X.shape[1])]
        self.w = np.array([[0 if weight < 0.001 and weight > -0.001 else weight for weight in weights]])

        #Substitute into a support vector to find bias
        X_pos = []
        X_neg = []
        for i, x in enumerate(X):
            if y[i] == 1:
                X_pos.append(x)
            elif y[i] == 0 or y[i] == -1:
                X_neg.append(x)
        self.b = -0.5 * (max([np.dot(self.w, x_i) for x_i in X_neg]) + min([np.dot(self.w, x_j) for x_j in X_pos]))
        
        return self

    
    def predict(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """

        return np.sign(np.dot(X, self.w.T) + self.b)

    def outputs(self, X):
        """
        Perform classification on samples in X.

        For a one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples_test, n_samples_train)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
          Class labels for samples in X.
        """
        return np.dot(self.w.T, X) + self.b

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels. 

        In multi-label classification, this is the subset accuracy which is a harsh 
        metric since you require for each sample that each label set be correctly 
        predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          True labels for X.

        Return
        ------
        score : float
          Mean accuracy of self.predict(X)
        """

        return np.mean([self.predict(X) == y])

if __name__ == "__main__":

    #TESTING CODE

    
    from scipy.io import loadmat
    from implementation import SVM, linear_kernel, nonlinear_kernel
    # from solution import SVM, linear_kernel, nonlinear_kernel
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    import matplotlib
    import matplotlib.pyplot as plt


    # Input data - of the form [Bias term, x_1 value, x_2 value]
    X = np.array([
        [1, -2, 4,],
        [1, 4, 1,],
        [1, 1, 6,],
        [1, 2, 4,],
        [1, 6, 2,],
    ])

    # Associated output labels - first 2 examples are labeled '-1' and last 3 are labeled '+1'
    y = np.array([-1,-1,1,1,1]) 

    a = np.ones(y.shape)

    object_ = objective_function(X, y, a, linear_kernel)

    # print('our objective function:', object_)
    
    result = SVC(kernel="linear")
    result = SVC(kernel=nonlinear_kernel)
    result.fit(X, y.ravel())

    # support_vectors = result.support_vectors_
    # coefficients = result.dual_coef_
    # decision_values = result.decision_function(X)
    # objective_value = 0.5 * np.dot(coefficients, coefficients.T) - np.sum(decision_values)
    # print('dual coefficients:', coefficients)
    # print('scikit learn objective:', objective_value)

    # print("scikit-learn indices of support vectors:", result.support_)

    Support_vector = SVM(kernel=nonlinear_kernel)
    
    Support_vector.fit(X, y)

    # print(Support_vector.predict(X))

    # print(Support_vector.score(X,y))

    print("Their weights:", result.coef0)
    print("Out Weights:", Support_vector.w)

    
