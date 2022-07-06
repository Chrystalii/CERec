from ranking import (ModelRecommendation as Model,
                                       LearnerRecommendation as Learner)
from BaseRecRepo.utils.format_data import *
from BaseRecRepo.opt import *
from BaseRecRepo.utils.datacaching import cache_rows
from collections import defaultdict
from scipy.special import expit as sigmoid
import numpy as np
import time
import warnings
from tqdm import tqdm

__all__ = ['CLiMFLearner']
__sparse_format__ = lil_matrix


def _g(x):
    """sigmoid function"""
    return sigmoid(x)


def _dg(x):
    ex = np.exp(-x)
    y = ex / (1 + ex) ** 2
    return y


def _matrix_factorization(ratings, shape, num_factors, num_iter, learning_rate,
                          lmbda, optimizer, verbose=False, random_state=None,
                          callback=None):
    # Seed the generator
    if random_state is not None:
        np.random.seed(random_state)

    # Get featured matrices dimensions
    num_users, num_items = shape

    # Initialize low-rank matrices
    U = 0.01 * np.random.rand(num_users, num_factors)  # User-feature matrix
    V = 0.01 * np.random.rand(num_items, num_factors)  # Item-feature matrix

    # Configure optimizer
    update_ui = create_opt(optimizer, learning_rate).update
    update_vw = create_opt(optimizer, learning_rate).update

    # Cache rows
    users_cached = defaultdict(list)

    # Print information about the verbosity level
    if verbose:
        print('CLiMF factorization started.')
        print('\tLevel of verbosity: ' + str(int(verbose)))
        print('\t\t- Verbosity = 1\t->\t[time/iter]')
        print('\t\t- Verbosity = 2\t->\t[time/iter, loss]')
        print('\t\t- Verbosity = 3\t->\t[time/iter, loss, MRR]')
        print('')

        # Prepare sample of users
        if verbose > 2:
            queries = None
            num_samples = min(num_users, 1000)  # max. number to sample
            users_sampled = np.random.choice(np.arange(num_users), num_samples)

    # Catch warnings
    with warnings.catch_warnings():

        # Turn matching warnings into exceptions
        warnings.filterwarnings('error')
        try:

            # Factorize matrix using SGD
            for step in tqdm(range(num_iter)):
                if verbose:
                    start = time.time()
                    print('- Step: %d' % (step + 1))

                # Send information about the process
                if callback:
                    callback(step + 1)

                # Optimize rating prediction
                for i in range(len(U)):
                    dU = -lmbda * U[i]

                    # Precompute f (f[j] = <U[i], V[j]>)
                    items = cache_rows(ratings, i, users_cached)
                    f = np.einsum('j,ij->i', U[i], V[items])

                    for j in range(len(items)):  # j=items
                        w = items[j]

                        dV = _g(-f[j]) - lmbda * V[w]

                        # For I
                        vec1 = _dg(f[j] - f) * \
                               (1 / (1 - _g(f - f[j])) - 1 / (1 - _g(f[j] - f)))
                        dV += np.einsum('i,j->ij', vec1, U[i]).sum(axis=0)

                        update_vw(-dV, V, w)

                        dU += _g(-f[j]) * V[w]

                        # For II
                        vec2 = (V[items[j]] - V[items])
                        vec3 = _dg(f - f[j]) / (1 - _g(f - f[j]))
                        dU += np.einsum('ij,i->ij', vec2, vec3).sum(axis=0)

                    update_ui(-dU, U, i)

                # Print process
                if verbose:
                    print('\t- Time: %.3fs' % (time.time() - start))

                    if verbose > 1:
                        # Set parameters and compute loss
                        low_rank_matrices = (U, V)
                        params = lmbda
                        objective = compute_loss(ratings, low_rank_matrices, params)
                        print('\t- Training loss: %.3f' % objective)

                        if verbose > 2:
                            model = CLiMFModel(U=U, V=V)
                            mrr, queries = \
                                model.compute_mrr(ratings, users_sampled, queries)
                            print('\t- Train MRR: %.4f' % mrr)
                    print('')

        except RuntimeWarning:
            callback(num_iter) if callback else None
            raise RuntimeError('Training diverged and returned NaN.')

    return U, V


def compute_loss(data, low_rank_matrices, params):

    # Set parameters
    ratings = data
    U, V = low_rank_matrices
    lmbda = params

    # Check data type
    if isinstance(ratings, __sparse_format__):
        pass
    elif isinstance(ratings, Table):
        # Preprocess Orange.data.Table and transform it to sparse
        ratings, order, shape = preprocess(ratings)
        ratings = table2sparse(ratings, shape, order, m_type=__sparse_format__)
    else:
        raise TypeError('Invalid data type')

    # Cache rows
    users_cached = defaultdict(list)

    F = -0.5*lmbda*(np.sum(U*U)+np.sum(V*V))

    for i in range(len(U)):
        # Precompute f (f[j] = <U[i], V[j]>)
        items = cache_rows(ratings, i, users_cached)
        f = np.einsum('j,ij->i', U[i], V[items])

        for j in range(len(items)):  # j=items
            F += np.log(_g(f[j]))
            F += np.log(1 - _g(f - f[j])).sum(axis=0)  # For I
    return F


class CLiMFLearner(Learner):
    """CLiMF: Collaborative Less-is-More Filtering Matrix Factorization

    This model uses stochastic gradient descent to find two low-rank
    matrices: user-feature matrix and item-feature matrix.

    CLiMF is a matrix factorization for scenarios with binary relevance data
    when only a few (k) items are recommended to individual users. It improves top-k
    recommendations through ranking by directly maximizing the Mean Reciprocal
    Rank (MRR).


    Attributes:
        num_factors: int, optional
            The number of latent factors.

        num_iter: int, optional
            The number of passes over the training data (aka epochs).

        learning_rate: float, optional
            The learning rate controlling the size of update steps (general).

        lmbda: float, optional
            Controls the importance of the regularization term (general).
            Avoids overfitting by penalizing the magnitudes of the parameters.

        optimizer: Optimizer, optional
            Set the optimizer for SGD. If None (default), classical SGD will be
            applied.

        verbose: boolean or int, optional
            Prints information about the process according to the verbosity
            level. Values: False (verbose=0), True (verbose=1) and INTEGER

        random_state: int, optional
            Set the seed for the numpy random generator, so it makes the random
            numbers predictable. This a debbuging feature.

        callback: callable

    """

    name = 'CLiMF'

    def __init__(self, num_factors=5, num_iter=25, learning_rate=0.0001,
                 lmbda=0.001, preprocessors=None, optimizer=None, verbose=False,
                 random_state=None, callback=None):
        self.num_factors = num_factors
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.optimizer = SGD() if optimizer is None else optimizer
        self.random_state = random_state
        self.callback = callback

        super().__init__(preprocessors=preprocessors, verbose=verbose)

    def fit_storage(self, data):
        """Fit the model according to the given training data.

        Args:
            data: Orange.data.Table

        Returns:
            self: object
                Returns self.

        """

        # Prepare data
        data = super().prepare_fit(data)

        # Check convergence
        if self.learning_rate == 0:
            warnings.warn("With learning_rate=0, this algorithm does not "
                          "converge well.", stacklevel=2)

        # Transform ratings matrix into a sparse matrix
        data = table2sparse(data, self.shape, self.order,
                            m_type=__sparse_format__)

        # Factorize matrix
        U, V = _matrix_factorization(ratings=data, shape=self.shape,
                                     num_factors=self.num_factors,
                                     num_iter=self.num_iter,
                                     learning_rate=self.learning_rate,
                                     lmbda=self.lmbda, optimizer=self.optimizer,
                                     verbose=self.verbose,
                                     random_state=self.random_state,
                                     callback=self.callback)

        # Construct model
        model = CLiMFModel(U=U, V=V)
        return super().prepare_model(model)


class CLiMFModel(Model):

    def __init__(self, U, V):
        self.U = U
        self.V = V
        super().__init__()

    def predict(self, X, top_k=None):
        """Perform predictions on samples in X for all items.

        Args:
            X: array, optional
                Array with the indices of the users to which make the
                predictions. If None (default), predicts for all users.

            top_k: int, optional
                Returns the k-first predictions. (Do not confuse with
                'top-best').

        Returns:
            C: ndarray, shape = (n_samples, n_items)
                Returns predicted values. A matrix (U, I) with the indices of
                the items recommended, sorted by ascending ranking. (1st better
                than 2nd, than 3rd,...)

        """

        # Check if is an array
        if isinstance(X, np.ndarray):
            if X.ndim != 1:
                X = X[:, self.order[0]]

        elif not isinstance(X, int):  # Check if is not an int
            raise TypeError("'Invalid input data. Array or 'int'")

        # Compute scores
        predictions = np.dot(self.U[X], self.V.T)

        # Return indices of the sorted predictions
        # (Return top-k recommendations, optional)
        if isinstance(X, int):  # Array of 1D
            predictions = np.argsort(predictions)
            predictions = predictions[::-1]
            predictions = predictions[:top_k] if top_k else predictions
        else:
            predictions = np.argsort(predictions, axis=1)
            predictions = np.fliplr(predictions)
            predictions = predictions[:, :top_k] if top_k else predictions

        return predictions

    def getUTable(self):
        variable = self.original_domain.variables[self.order[0]]
        return feature_matrix(variable, self.U)

    def getVTable(self):
        variable = self.original_domain.variables[self.order[1]]
        return feature_matrix(variable, self.V)
