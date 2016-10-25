import copy
import inspect
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.utils import check_X_y

class MultivariateSVR(BaseEstimator, RegressorMixin):

    class AttributedKernelFunc(BaseEstimator):
        def __init__(self, kernel_func, **kargs):
            self.kernel_func = kernel_func
            for parameter in inspect.signature(kernel_func).parameters.values():
                if not parameter.name in ["x", "X", "y", "Y"]:
                    setattr(self, parameter.name, parameter.default)
            self.set_params(**kargs)

        def get_params(self, deep = True):
            return  self.__dict__.copy()

        def __call__(self, X, Y=None):
            parameters = self.__dict__.copy()
            parameters.pop("kernel_func")
            return self.kernel_func(X, Y, **parameters)

    def __init__(self, kernel_func=None, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        assert(callable(kernel_func))
        if isinstance(kernel_func, MultivariateSVR.AttributedKernelFunc):
            self.kernel_func = kernel_func
        else:
            self.kernel_func = MultivariateSVR.AttributedKernelFunc(kernel_func)
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X, y = check_X_y(X, y,
                         multi_output=True,
                         accept_sparse=True)
        if len(y.shape) != 2:
            raise RuntimeError("You must supply a multivariate (2d) y matrix!")
        self.fitted_X_ = X
        kernel = self.kernel_func(X)
        proto_svr = SVR(
            kernel='precomputed',
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            verbose=self.verbose,
            max_iter=self.max_iter)
        self.fitted_svrs_ = [copy.deepcopy(proto_svr).fit(kernel, y[:, i])
                             for i in range(y.shape[1])]
        return self

    def predict(self, X, y=None):
        kernel = self.kernel_func(X, self.get_fitted_X_())
        list_of_predictions = [regressor.predict(kernel)
                               for regressor in self.get_fitted_svrs_()]
        return np.array(list_of_predictions).transpose()

    def score(self, X, y=None):
        kernel = self.kernel_func(X, self.get_fitted_X_())
        list_of_scores = [regressor.score(kernel, y_vec)
                          for y_vec, regressor in zip(y.transpose(), self.get_fitted_svrs_())]
        return sum(list_of_scores)/len(list_of_scores)

    def get_fitted_svrs_(self):
        try:
            svrs = getattr(self, "fitted_svrs_")
        except AttributeError:
            raise RuntimeError("You must fit MultivariateSVR first!")
        return svrs

    def get_fitted_X_(self):
        try:
            x = getattr(self, "fitted_X_")
        except AttributeError:
            raise RuntimeError("You must fit MultivariateSVR first!")
        return x
