"""measurement_process_layers.py: Classes representing such layers."""
import numpy as np
import pandas as pd
import pdb

# Used in wrapper
from functools import wraps

# Scipy imports
from scipy.special import betaincinv, erfinv
from scipy.stats import expon
from scipy.stats import norm

# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.math import tanh, sigmoid
from tensorflow.nn import relu
from tensorflow.keras.layers import Layer, Concatenate

# MAVE-NN imports
from mavenn.src.error_handling import check, handle_errors

from mavenn.src.reshape import _shape_for_output, \
    _get_shape_and_return_1d_array, \
    _broadcast_arrays

from mavenn.src.validate import validate_seqs, \
    validate_1d_array, \
    validate_alphabet
from mavenn.src.utils import mat_data_to_vec_data, \
    vec_data_to_mat_data, \
    x_to_stats, \
    p_lc_to_x, _x_to_mat

pi = np.pi
e = np.exp(1)

# To clarify equations
# Log = K.log
# LogGamma = tf.math.lgamma
# Exp = K.exp
Sqrt = K.sqrt
Square = K.square

# MAX_EXP_ARG = np.float32(20.0)
MAX_EXP_ARG = np.float32(50.0)
MIN_LOG_ARG = np.float32(np.exp(-MAX_EXP_ARG))

# Create safe functions


def Log(x):
    x = tf.clip_by_value(x, MIN_LOG_ARG, np.inf)
    return K.log(x)


def Log10(x):
    x = tf.clip_by_value(x, MIN_LOG_ARG, np.inf)
    #return K.log(x)/K.log(tf.constant(10.0, dtype=tf.float32))
    return K.log(x) / K.log(10.0)


def Exp10(x):
    x = tf.clip_by_value(x, -MAX_EXP_ARG, MAX_EXP_ARG)
    x = tf.cast(x, dtype=tf.float32)
    return tf.math.pow(10.0, x)


def LogGamma(x):
    x = tf.clip_by_value(x, MIN_LOG_ARG, np.inf)
    return tf.math.lgamma(x)


def Exp(x):
    x = tf.clip_by_value(x, -MAX_EXP_ARG, MAX_EXP_ARG)
    x = tf.cast(x, dtype=tf.float32)
    return K.exp(x)


def handle_arrays(func):
    """
    Allow functions that take tf.Tensor objects to take np.ndarrays.

    Decorator function that allows functions that take and return
    tensors to operate on numpy arrays. If use_arrays=True is set
    as a keyword argument, every input of one of the input_types
    is converted to a tf.constant before being passed to func. Then,
    all outputs from func that are tf.Tensor objects are converted to
    np.ndarrays.
    """
    # Input types to convert to tf.Tensor objects upon input
    input_types = (np.ndarray, list, pd.Series)

    # Use @wraps to preserve docstring
    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Get use_arrays keyword argument.
        use_arrays = kwargs.pop('use_arrays', False)

        # Make sure use_arrays is boolean
        check(isinstance(use_arrays, bool),
              f'use_arrays={use_arrays}; must be bool.')

        # If use_arrays is selected
        if use_arrays:

            # Convert positional arguments
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, input_types):
                    args[i] = tf.constant(arg, dtype=tf.float32)

            # Convert keyword arguments
            for key, arg in kwargs.items():
                if isinstance(arg, input_types):
                    kwargs[key] = tf.constant(arg, dtype=tf.float32)

            # Pass inputs to func and get outputs
            result = func(*args, **kwargs)

            # Convert results within a tuple
            if isinstance(result, tuple):
                result = (r.numpy() if isinstance(r, tf.Tensor) else r
                          for r in result)

            # Convert a sole result
            else:
                result = result.numpy() if isinstance(result, tf.Tensor) \
                    else result

        # Otherwise, just use raw function.
        else:
            result = func(*args, **kwargs)

        # Retun (potentially) adjusted results
        return result

    # Return wrapped function
    return wrapped_func


class MeasurementProcess(Layer):
    """
    Represents a measurement process base class.
    """

    def __init__(self,
                 eta,
                 info_for_layers_dict):

        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # Call superclass constructor
        super().__init__()

    def get_config(self):
        assert False

    def get_params(self):
        assert False

    def set_params(self):
        assert False

    def p_of_y_given_phi(self,
                         phi,
                         y,
                         paired=True):
        assert False

    def sample_y_given_phi(self,
                           phi,
                           y):
        assert False

    def negative_log_likelihood(self,
                                phi,
                                y):
        assert False

    def call(self):
        assert False


class GlobalEpsistasisMP(MeasurementProcess):

    def __init__(self,
                 K=50,
                 eta=1e-5,
                 number_latent_nodes=1,
                 monotonic=True,
                 ge_heteroskedasticity_order=0,
                 info_for_layers_dict={},
                 ge_noise_model_type='Gaussian'):

        # TODO: input checks
        self.K = K
        self.eta = eta
        self.monotonic = monotonic
        self.info_for_layers_dict = info_for_layers_dict
        self.number_latent_nodes = number_latent_nodes
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order

        if number_latent_nodes == 1:
            yhat = GlobalEpistasisLayer(K=self.K,
                                        eta=self.eta,
                                        monotonic=self.monotonic)

        elif number_latent_nodes > 1:
            yhat = MultiLatentGlobalEpistasisLayer(K=self.K,
                                                   eta=self.eta,
                                                   number_latent_nodes=self.number_latent_nodes,
                                                   monotonic=self.monotonic)
        else:
            print(f'number_latent_nodes = {self.number_latent_nodes} cannot be negative or float')
            print('Need to implement number_latent_nodes check in multi GE MP.')

        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':
            noise_model_layer = GaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta)

        elif ge_noise_model_type == 'Cauchy':
            noise_model_layer = CauchyNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta)

        elif ge_noise_model_type == 'SkewedT':
            noise_model_layer = SkewedTNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta)

        self.yhat = yhat
        self.mp_layer = noise_model_layer

    def phi_to_yhat(self, phi):
        return self.yhat.phi_to_yhat(phi)

    def p_of_y_given_phi(self,
                         phi,
                         y,
                         paired=True):

        yhat = self.phi_to_yhat(phi)
        return self.mp_layer.p_of_y_given_yhat(y, yhat)


class ExponentialEnrichmentMP(MeasurementProcess):
    """
    For experiments in which sequences undergo exponential enrichment,
    e.g., selective growth, affinity enrichment, etc.
    Data provided as table of counts for bins y = 0,1,..., Y − 1, Y ≥ 2.
    Bin y contains sequences having undergone selection for a time duration ty. Typically,
    y = 0 will correspond to the input library, with t0 = 0 indicating no selection.
    Trainable parameters: b_y and (optionally) t_y.

    Y: (int)
        Total number of bins. Y>=2.

    t_y (array-like)
        User sets t_y = [t_0, t_1,...,t_{Y-1}] values. If t_y[i] = None, that ith value should be trainable parameter.
        However, at least two enrichment times must be specified (e.g. t_0 = 0, t_1 = 1).

    # TODO: What is eta?

    # TODO: need to implement logic if t_y is not None 
    # RESOLVED: The t_y values which are not None are fixed otherwise trainable (MK).
    """

    def __init__(self,
                 Y,
                 t_y,
                 eta,
                 info_for_layers_dict,
                 **kwargs
                 ):

        self.Y = Y
        self.t_y_input = t_y

        # TODO: User should provide the t_y vector of length Y. Check this and raise error.
        # TODO: Check Y>=2 otherwise raise error.Ti

        # attributes of base MeasurementProcess class
        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.Y}

    def build(self, input_shape):
        """Build layer."""

        self.b_y = self.add_weight(name='b_y',
                                   dtype=tf.float32,
                                   shape=(self.Y,),
                                   initializer=RandomNormal(),
                                   trainable=True,
                                   regularizer=self.regularizer)
        # Set the t_y weights
        t_y = []
        for i, t in enumerate(self.t_y_input):
            if t is None:
                # When t_y_i from user is not fixed and it is trainable
                t_y.append(self.add_weight(name=f't_y_{i}', dtype=tf.float32,
                                           shape=(1,),
                                           initializer=Constant(1.0),
                                           trainable=True,
                                           constraint=non_neg(),
                                           regularizer=self.regularizer))
            else:
                # When the t_y from user is fixed
                # and not-trainable use the initializer to be Constant(t_y)
                t_y.append(self.add_weight(name=f't_y_{i}', dtype=tf.float32,
                                           shape=(1,), trainable=False,
                                           initializer=Constant(t)))
        self.t_y = t_y

        super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,1] is phi
            inputs[:,1:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """

        # Extract and shape inputs
        phi = inputs[:, 0]
        ct_my = inputs[:, 1:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)

        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""

        # Shape phi
        phi_m = tf.reshape(phi, [-1, 1])

        # Reshape parameters
        b_my = tf.reshape(self.b_y, [-1, self.Y])
        t_my = tf.reshape(self.t_y, [-1, self.Y])

        # Compute weights
        psi_my = t_my * phi_m + b_my
        psi_my = tf.reshape(psi_my, [-1, self.Y])
        w_my = Exp(psi_my)

        # Compute and return distribution
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])

        if return_var == 'w':
            return w_my
        elif return_var == 'psi':
            return psi_my
        elif return_var == 'p':
            return p_my
        else:
            assert False

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        # Cast y as integers
        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        # Get values for all bins
        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        p = _shape_for_output(p, p_shape)
        return p


class SortSeqMP(MeasurementProcess):
    """
    Represents a Sortseq based measurement process.
    The result is that phi(x) is interpreted as the mean
    log_10 fluorescence of a clonal population of cells harboring
    sequence x.

    Y: (int)
        Number of sorting bins, i.e, y = 0, ..., Y-1.

    N_y: (array-like)
        Number of reads in every bin.
        Note that this should be inferred from the training data
        i.e. for cts data: ct_0, ct_1, ... ct_(Y-1), N_y -> sum(axis=0) (sum rows)

    mu_pos: (float)
        Mean of log_10 fluorescence for positive control
        (i.e., highly fluorescent) sample.

    sigma_pos: (float)
        Standard deviation of log_10 fluorescence for positive control
        (i.e., highly fluorescent) sample.

    mu_neg: (float)
        Mean of log_10 fluorescence for negative control
        (i.e., non-fluorescent) sample.

    sigma_neg: (float)
        Standard deviation of log_10 fluorescence for negative control
        (i.e., non-fluorescent) sample.

    f_y_upper_bounds: (array-like)
        Upper bounds on log_10 fluorescence for each sorting bin.
        I.e., {f_y^ub}_[y=0,Y-1]

    f_y_lower_bounds: (array-like)
        Lower bounds on log_10 fluorescence for each sorting bin.
        I.e., {f_y^lb}_[y=0,Y-1]


    eta: (float)
        L2 regularization.
    """

    def __init__(self,
                 N_y,
                 Y,
                 mu_pos,
                 sigma_pos,
                 mu_neg,
                 sigma_neg,
                 f_y_upper_bounds,
                 f_y_lower_bounds,
                 eta,
                 info_for_layers_dict,
                 **kwargs
                 ):
        """Construct layer."""
        # Set attributes
        self.Y = Y
        self.N_y = N_y
        self.mu_pos = mu_pos
        self.sigma_pos = sigma_pos
        self.mu_neg = mu_neg
        self.sigma_neg = sigma_neg
        self.f_y_upper_bounds = f_y_upper_bounds
        self.f_y_lower_bounds = f_y_lower_bounds

        # attributes of base MeasurementProcess class
        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # total number of reads
        self.N = np.sum(self.N_y)

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}    # TODO: check if self.number_bins is defined/works

    # Note that this layer doesn't have trainable weights so we only
    # should need to use the 'call' method
    # def build(self, input_shape):
    #     """Build layer."""
    #     pass

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,1] is phi
            inputs[:,1:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """

        # Extract and shape inputs
        phi = inputs[:, 0]
        ct_my = inputs[:, 1:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)

        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        # Cast y as integers
        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        # Get values for all bins
        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        p = _shape_for_output(p, p_shape)
        return p

    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""

        # Reshape phi.
        phi = tf.reshape(phi, [-1, 1, 1])

        # Perform computations for Sort-seq measurement process
        mu_of_phi = phi

        # transform phi between 0 and 1
        lambda_of_phi = (mu_of_phi - self.mu_neg) / (self.mu_pos - self.mu_neg)

        # Log_sigma_pos_over_sigma_neg = tf.reshape(
        #     Log(self.sigma_pos / self.sigma_neg), [-1, 1, 1])
        # Log_sigma_pos_over_sigma_neg = tf.cast(
        #     Log_sigma_pos_over_sigma_neg, dtype=tf.float32)
        #
        # sigma_of_phi = self.sigma_neg * \
        #     Exp(lambda_of_phi * Log_sigma_pos_over_sigma_neg)

        sigma_of_phi = (1 - lambda_of_phi) * self.sigma_neg + \
            lambda_of_phi * self.sigma_pos

        # upper and lower bounds transformed into z-values.
        z_y_upper_bound = (self.f_y_upper_bounds - mu_of_phi) / sigma_of_phi
        z_y_lower_bound = (self.f_y_lower_bounds - mu_of_phi) / sigma_of_phi

        # prob mass of normal distribution p(y|phi) between z_y_ub and z_y_lb
        u_y_of_phi = tf.math.erf(
            np.sqrt(2) * z_y_upper_bound) - tf.math.erf(np.sqrt(2) * z_y_lower_bound)

        # relative probability of sequence in dataset being found in bin y

        N_y = tf.reshape(self.N_y, [-1, 1, self.Y])
        N_y = tf.cast(N_y, dtype=tf.float32)

        w_my = (N_y / self.N) * u_y_of_phi
        w_my = tf.reshape(w_my, [-1, self.Y])

        # normalized probability
        # shape of w_my is [None,1,Y], that's why the sum in the line below has to go on dimension 2
        # i.e., sum over Y
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])
        # print(f'w_my shape = {w_my.shape}, u_y_of_phi shape = {u_y_of_phi.shape}, p_my = {p_my.shape}, N_y shape = {N_y.shape}')
        return p_my


class TiteSeqMP(MeasurementProcess):
    """
    Represents a parametric measurement process for analyzing
    Titeseq data.

    c: (array-like)
        The labeling concentration.

    Y: (int)
        Number of sorting bins, i.e, y = 0, ..., Y-1.

    N_y: (array-like)
        Number of reads in every bin.
        Note that this should be inferred from the training data
        i.e. for cts data: ct_0, ct_1, ... ct_(Y-1), N_y -> sum(axis=0) (sum rows)

    mu_pos: (float)
        Mean of log_10 fluorescence for positive control
        (i.e., highly fluorescent) sample.

    sigma_pos: (float)
        Standard deviation of log_10 fluorescence for positive control
        (i.e., highly fluorescent) sample.

    mu_neg: (float)
        Mean of log_10 fluorescence for negative control
        (i.e., non-fluorescent) sample.

    sigma_neg: (float)
        Standard deviation of log_10 fluorescence for negative control
        (i.e., non-fluorescent) sample.

    f_y_upper_bounds: (array-like)
        Upper bounds on log_10 fluorescence for each sorting bin.
        I.e., {f_y^ub}_[y=0,Y-1]

    f_y_lower_bounds: (array-like)
        Lower bounds on log_10 fluorescence for each sorting bin.
        I.e., {f_y^lb}_[y=0,Y-1]

    eta: (float)
        L2 regularization.
    """

    def __init__(self,
                 c,
                 a,
                 N_y,
                 Y,
                 mu_pos,
                 sigma_pos,
                 mu_neg,
                 sigma_neg,
                 f_y_upper_bounds,
                 f_y_lower_bounds,
                 eta,
                 info_for_layers_dict,
                 **kwargs
                 ):
        """Construct layer."""
        # Set attributes
        self.c = c
        self.a = a,
        self.Y = Y
        self.N_y = N_y
        self.mu_pos = mu_pos
        self.sigma_pos = sigma_pos
        self.mu_neg = mu_neg
        self.sigma_neg = sigma_neg
        self.f_y_upper_bounds = f_y_upper_bounds
        self.f_y_lower_bounds = f_y_lower_bounds

        # attributes of base MeasurementProcess class
        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # total number of reads
        self.N = np.sum(self.N_y)

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}    # TODO: check if self.number_bins is defined/works

    # def build(self, input_shape):
    #     """Build layer."""
    #
    #     # this weight is the constant in front of Hill-function mu(phi)
    #     self.D = self.add_weight(name='D',
    #                              dtype=tf.float32,
    #                              shape=(1,),
    #                              initializer=Constant(1.0),
    #                              trainable=True,
    #                              regularizer=self.regularizer)
    #
    #     super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,1] is phi
            inputs[:,1:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """

        # # code from one-dimensional phi
        phi = inputs[:, 0]
        ct_my = inputs[:, 1:]

        # Extract and shape inputs
        # ' phi shape in call (None, 2)')
        # phi = inputs[:, 0: 2]
        # ct_my = inputs[:, 2:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)
        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs

        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # y, y_shape = _get_shape_and_return_1d_array(y)
        # # phi, phi_shape = _get_shape_and_return_1d_array(phi)
        # phi_shape = phi.shape

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)
            # Set output shape
            p_shape = y_shape + phi_shape
        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        # Cast y as integers
        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        # Get values for all bins
        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        p = _shape_for_output(p, p_shape)
        return p

    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""

        # phi is the latent phenotype representing binding
        phi = tf.reshape(phi, [-1, 1])

        # phi_0 = phi[:, 0]
        # phi_1 = phi[:, 1]
        #
        # phi_0 = tf.reshape(phi_0, [-1, 1])
        # phi_1 = tf.reshape(phi_1, [-1, 1])
        #
        # phi_0 = tf.cast(phi_0, dtype=tf.float32)
        # phi_1 = tf.cast(phi_1, dtype=tf.float32)

        K_a_of_phi = Exp10(phi)
        A_of_phi = Exp10(self.a)
        #print(f' SHAPE OF phi_0  = {phi_0.shape}')
        # K_a_of_phi = Exp(phi_0)
        # A_of_phi = Exp(phi_1)

        #self.mu_neg = tf.cast(self.mu_neg, dtype=tf.float32)
        B = Exp10(self.mu_neg)
        B = tf.cast(B, dtype=tf.float32)

        # mu_of_phi = Log10(A_of_phi * ((Exp10(self.c) * K_a_of_phi) /
        #                             (1 + Exp10(self.c) * K_a_of_phi)) + B)

        #mu_of_phi = self.D*Log10(A_of_phi * ((self.c * K_a_of_phi) /
        mu_of_phi = Log10(A_of_phi * ((self.c * K_a_of_phi) /
                                (1 + self.c * K_a_of_phi)) + B)

        # transform phi between 0 and 1
        lambda_of_phi = (mu_of_phi - self.mu_neg) / (self.mu_pos - self.mu_neg)

        sigma_of_phi = (1 - lambda_of_phi) * self.sigma_neg + \
            lambda_of_phi * self.sigma_pos

        # upper and lower bounds transformed into z-values.
        z_y_upper_bound = (self.f_y_upper_bounds - mu_of_phi) / sigma_of_phi
        z_y_lower_bound = (self.f_y_lower_bounds - mu_of_phi) / sigma_of_phi
        # prob mass of normal distribution p(y|phi) between z_y_ub and z_y_lb
        u_y_of_phi = tf.math.erf(
            np.sqrt(2) * z_y_upper_bound) - tf.math.erf(np.sqrt(2) * z_y_lower_bound)

        # relative probability of sequence in dataset being found in bin y
        N_y = tf.reshape(self.N_y, [-1, self.Y])
        N_y = tf.cast(N_y, dtype=tf.float32)
        w_my = (N_y / self.N) * u_y_of_phi

        # normalized probability
        # sum over Y
        # p_my = w_my / tf.math.reduce_sum(w_my)
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])

        return p_my


class DiscreteAgnosticMP(MeasurementProcess):
    """Represents an Discrete agnostic measurement process.
        (known as MPA in mave-nn 1)
    """

    def __init__(self,
                 Y,
                 K,
                 eta,
                 info_for_layers_dict,
                 **kwargs):
        """Construct layer."""
        # Set attributes
        self.Y = Y
        self.K = K
        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}

    def build(self, input_shape):
        """Build layer."""

        self.a_y = self.add_weight(name='a_y',
                                   dtype=tf.float32,
                                   shape=(self.Y,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        b_yk0 = expon(scale=1 / self.K).rvs([self.Y, self.K])
        self.b_yk = self.add_weight(name='b_yk',
                                    dtype=tf.float32,
                                    shape=(self.Y, self.K),
                                    initializer=Constant(b_yk0),
                                    trainable=True,
                                    regularizer=self.regularizer)

        self.c_k = self.add_weight(name='c_k',
                                   dtype=tf.float32,
                                   shape=(self.Y, self.K,),
                                   initializer=Constant(1.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   dtype=tf.float32,
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,1] is phi
            inputs[:,1:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """
        # Extract and shape inputs
        phi = inputs[:, 0]
        ct_my = inputs[:, 1:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)

        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""
        # Shape phi
        phi = tf.reshape(phi, [-1, 1, 1])

        # Reshape parameters
        a_y = tf.reshape(self.a_y, [-1, self.Y])
        b_yk = tf.reshape(self.b_yk, [-1, self.Y, self.K])
        c_k = tf.reshape(self.c_k, [-1, self.K])
        d_k = tf.reshape(self.d_k, [-1, self.K])

        # Compute weights
        psi_my = a_y + K.sum(b_yk * tanh(c_k * phi + d_k), axis=2)

        #print(f'tanh(K.sum(c_kl * phi, axis=1) = {(psi_my).shape}')
        psi_my = tf.reshape(psi_my, [-1, self.Y])
        w_my = Exp(psi_my)
        # print(f'w_my shape {w_my.shape}')

        # Compute and return distribution
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])

        if return_var == 'w':
            return w_my
        elif return_var == 'psi':
            return psi_my
        else:
            return p_my


class MultiLatentDiscreteAgnosticMP(MeasurementProcess):
    """Represents an Discrete agnostic measurement process which takes
        in multiple latent phenotypes as input
    """

    def __init__(self,
                 Y,
                 K,
                 eta,
                 number_latent_nodes,
                 info_for_layers_dict,
                 **kwargs):
        """Construct layer."""
        # Set attributes
        self.Y = Y
        self.K = K
        self.eta = eta
        self.number_latent_nodes = number_latent_nodes  # represents total number of latent nodes.
        self.info_for_layers_dict = info_for_layers_dict

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}

    def build(self, input_shape):
        """Build layer."""

        self.a_y = self.add_weight(name='a_y',
                                   dtype=tf.float32,
                                   shape=(self.Y,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        #b_yk0 = expon(scale=1 / self.K).rvs([self.Y, self.K])
        self.b_yk = self.add_weight(name='b_yk',
                                    dtype=tf.float32,
                                    shape=(self.Y, self.K),
                                    #shape=(self.number_latent_nodes, self.Y, self.K),
                                    #initializer=Constant(b_yk0),
                                    trainable=True,
                                    regularizer=self.regularizer)

        self.c_kl = self.add_weight(name='c_kl',
                                   dtype=tf.float32,
                                   shape=(self.K, self.number_latent_nodes),
                                   initializer=Constant(1.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   dtype=tf.float32,
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,number_latent_nodes] is the vector phi
            inputs[:,number_latent_nodes:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """
        # Extract and shape inputs
        # phi = inputs[:, 0]
        # ct_my = inputs[:, 1:]

        phi = inputs[:, 0:self.number_latent_nodes]
        ct_my = inputs[:, self.number_latent_nodes:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)

        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""
        # Shape phi to make the tensor multiplication work.
        phi = tf.reshape(phi, [-1, self.number_latent_nodes,  1])

        # Reshape parameters
        a_y = tf.reshape(self.a_y, [-1, self.Y])
        #b_yk = tf.reshape(self.b_yk, [-1, self.number_latent_nodes,self.Y, self.K])
        b_yk = tf.reshape(self.b_yk, [-1, self.Y, self.K])
        c_kl = tf.reshape(self.c_kl, [-1, self.number_latent_nodes, self.K])
        d_k = tf.reshape(self.d_k, [-1, self.K])

        # Compute weights
        # Axis 2 represents sum over, k
        # axis 1 represents sum over latent phi nodes.

        # print(f'tanh(K.sum(c_kl * phi, axis=1) = {(b_yk*tanh(c_kl * phi + d_k)).shape}')
        
        #psi_my = a_y + K.sum(b_yk * tanh(K.sum(c_kl * phi, axis=1) + d_k), axis=2)
        psi_my = a_y + K.sum(b_yk * tanh(c_kl * phi + d_k), axis=2)
#        print(f'psi_my = {psi_my.shape}')
        # shape has to be none by Y
        psi_my = tf.reshape(psi_my, [-1, self.Y])

        # print(f'phi = {phi.shape}')
        # print(f'psi_my = {psi_my.shape}')

        w_my = Exp(psi_my)
        # print(f'w_my shape {w_my.shape}')

        # Compute and return distribution
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])

        #print(f'p_my = {p_my.shape}')

        if return_var == 'w':
            return w_my
        elif return_var == 'psi':
            return psi_my
        else:
            return p_my


class DiscreteMonotonicMP(MeasurementProcess):
    """
    Monotonic version of DiscreteAgnosticMP
    """

    def __init__(self,
                 Y,
                 K,
                 eta,
                 info_for_layers_dict,
                 **kwargs):
        """Construct layer."""
        # Set attributes
        self.Y = Y
        self.K = K
        self.eta = eta
        self.info_for_layers_dict = info_for_layers_dict

        # Set regularizer
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        super().__init__(eta, info_for_layers_dict, **kwargs)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                "info_for_layers_dict": self.info_for_layers_dict,
                "number_bins": self.number_bins}

    def build(self, input_shape):
        """Build layer."""

        self.a_y = self.add_weight(name='a_y',
                                   dtype=tf.float32,
                                   shape=(self.Y,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        b_yk0 = expon(scale=1 / self.K).rvs([self.Y, self.K])
        self.b_yk = self.add_weight(name='b_yk',
                                    dtype=tf.float32,
                                    shape=(self.Y, self.K),
                                    initializer=Constant(b_yk0),
                                    trainable=True,
                                    constraint=tf.keras.constraints.non_neg(),
                                    regularizer=self.regularizer)

        self.c_k = self.add_weight(name='c_k',
                                   dtype=tf.float32,
                                   shape=(self.K,),
                                   initializer=Constant(1.),
                                   constraint=tf.keras.constraints.non_neg(),
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   dtype=tf.float32,
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,Y+1) tensor containing counts, where B is batch
            size and Y is the number of bins.
            inputs[:,1] is phi
            inputs[:,1:Y+1] is c_my

        Returns
        -------
        negative_log_likelihood: (np.array)
            A (B,) tensor containing negative log likelihood contributions
        """
        # Extract and shape inputs
        phi = inputs[:, 0]
        ct_my = inputs[:, 1:]

        # Compute p(y|phi)
        p_my = self.p_of_all_y_given_phi(phi)

        # Compute negative log likelihood
        negative_log_likelihood = -K.sum(ct_my * Log(p_my), axis=1)
        ct_m = K.sum(ct_my, axis=1)

        # Add I_var metric
        H_y = self.info_for_layers_dict['H_y_norm']
        H_y_given_phi = np.log2(e) * \
            K.sum(negative_log_likelihood) / K.sum(ct_m)
        I_y_phi = H_y - H_y_given_phi
        self.add_metric(I_y_phi, name="I_var", aggregation="mean")

        return negative_log_likelihood

    def p_of_y_given_phi(self, y, phi, paired=False):
        """
        Compute probabilities p( ``y`` | ``phi`` ).
        Parameters
        ----------
        y: (np.ndarray)
            Measurement values. For GE models, must be an array of floats.
            For MPA models, must be an array of ints representing bin numbers.
        phi: (np.ndarray)
            Latent phenotype values, provided as an array of floats.
        paired: (bool)
            Whether values in ``y`` and ``phi`` should be treated as paired.
            If ``True``, the probability of each value in ``y`` value will be
            computed using the single paired value in ``phi``. If ``False``,
            the probability of each value in ``y`` will be computed against
            all values of in ``phi``.
        Returns
        -------
        p: (np.ndarray)
            Probability of ``y`` given ``phi``. If ``paired=True``,
            ``p.shape`` will be equal to both ``y.shape`` and ``phi.shape``.
            If ``paired=False``, ``p.shape`` will be given by
            ``y.shape + phi.shape``.
        """
        # Prepare inputs
        y, y_shape = _get_shape_and_return_1d_array(y)
        phi, phi_shape = _get_shape_and_return_1d_array(phi)

        # If inputs are paired, use as is
        if paired:
            # Check that dimensions match
            check(y_shape == phi_shape,
                  f"y shape={y_shape} does not match phi shape={phi_shape}")

            # Use y_shape as output shape
            p_shape = y_shape

        # Otherwise, broadcast inputs
        else:
            # Broadcast y and phi
            y, phi = _broadcast_arrays(y, phi)

            # Set output shape
            p_shape = y_shape + phi_shape

        # Ravel arrays
        y = y.ravel()
        phi = phi.ravel()

        y = y.astype(int)

        # Make sure all y values are valid
        check(np.all(y >= 0),
              f"Negative values for y are invalid for MAP regression")

        check(np.all(y < self.Y),
              f"Some y values exceed the number of bins {self.Y}")

        p_of_all_y_given_phi = self.p_of_all_y_given_phi(phi, use_arrays=True)

        # Extract y-specific elements
        _ = np.newaxis
        all_y = np.arange(self.Y).astype(int)
        y_ix = (y[:, _] == all_y[_, :])

        p = p_of_all_y_given_phi[y_ix]

        # Shape for output
        p = _shape_for_output(p, p_shape)
        return p

    # @tf.function
    @ handle_arrays
    def p_of_all_y_given_phi(self, phi, return_var='p'):
        """Compute p(y|phi) for all values of y."""
        # Shape phi
        phi = tf.reshape(phi, [-1, 1, 1])

        # Reshape parameters
        a_y = tf.reshape(self.a_y, [-1, self.Y])
        b_yk = tf.reshape(self.b_yk, [-1, self.Y, self.K])
        c_k = tf.reshape(self.c_k, [-1, self.K])
        d_k = tf.reshape(self.d_k, [-1, self.K])

        # Compute weights
        psi_my = a_y + K.sum(b_yk * tanh(c_k * phi + d_k), axis=2)

        u_y = tf.reshape(psi_my, [-1, self.Y, 1])

        # need to sum psi_my for all bins less than the current bin
        lower_m = np.tri(self.Y, k=0)
        lower_m_tf = tf.convert_to_tensor(lower_m, dtype=tf.float32)

        w_my = Exp(tf.matmul(lower_m_tf, u_y))
        w_my = tf.reshape(w_my, [-1, self.Y])

        # Compute and return distribution
        p_my = w_my / tf.reshape(K.sum(w_my, axis=1), [-1, 1])

        # print(f'psi_my = {psi_my.shape}')
        # print(f'p_my = {p_my.shape}')

        if return_var == 'w':
            return w_my
        elif return_var == 'psi':
            return psi_my
        else:
            return p_my


class AffineLayer(Layer):
    """Represents an affine map from phi to yhat."""

    @ handle_errors
    def __init__(self,
                 eta,
                 monotonic,
                 **kwargs):
        """Construct layer."""
        # Whether to make monotonic function
        self.monotonic = monotonic

        # Create function that returns a kernel constraint
        # based on self.monotonic
        self.constraint = lambda: non_neg() if self.monotonic else None

        # Set regularizer
        self.eta = eta
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        # Call superclass constructor
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a = self.add_weight(name='a',
                                 shape=(1,),
                                 initializer=Constant(0.),
                                 trainable=True,
                                 regularizer=self.regularizer)

        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer=Constant(0.),
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    # Just an alias for phi_to_yhat, for tensors only
    def call(self, phi):
        """Transform layer inputs to outputs."""
        return self.phi_to_yhat(phi)

    # yhat as function of phi
    @ handle_arrays
    def phi_to_yhat(self, phi):
        """Compute yhat from phi."""
        yhat = self.a + self.b * phi
        return yhat


class MultiLatentGlobalEpistasisLayer(Layer):
    """Represents a global epistasis layer."""

    @ handle_errors
    def __init__(self,
                 K,
                 eta,
                 number_latent_nodes,
                 monotonic,
                 **kwargs):
        """Construct layer."""
        # Whether to make monotonic function
        self.monotonic = monotonic

        # Create function that returns a kernel constraint
        # based on self.monotonic
        self.constraint = lambda: non_neg() if self.monotonic else None

        # Set number of hidden nodes
        self.K = K

        # represents total number of latent nodes.
        self.number_latent_nodes = number_latent_nodes

        # Set regularizer
        self.eta = eta
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        # Call superclass constructor
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a_0 = self.add_weight(name='a_0',
                                   shape=(1,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        #b_k_dist = expon(scale=1 / self.K)
        self.b_k = self.add_weight(name='b_k',
                                   shape=(self.K,),
                                   #initializer=Constant(b_k_dist.rvs(self.K)),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.c_kl = self.add_weight(name='c_k',
                                   shape=(self.K, self.number_latent_nodes),
                                   #initializer=Constant(1.),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    # Just an alias for phi_to_yhat, for tensors only
    def call(self, phi):
        """Transform layer inputs to outputs."""
        return self.phi_to_yhat(phi)

    # yhat as function of phi

    @handle_arrays
    def phi_to_yhat(self, phi):
        """Compute yhat from phi."""

        phi = tf.reshape(phi, [-1, 1, self.number_latent_nodes])
        b_k = tf.reshape(self.b_k, [-1, self.K])
        c_kl = tf.reshape(self.c_kl, [-1, self.K, self.number_latent_nodes])
        d_k = tf.reshape(self.d_k, [-1, self.K])
        yhat = self.a_0 + tf.reshape(K.sum(b_k * tanh(K.sum(c_kl * phi, axis=2) + d_k), axis=1), shape=[-1, 1])

        return yhat


class GlobalEpistasisLayer(Layer):
    """Represents a global epistasis layer."""

    @ handle_errors
    def __init__(self,
                 K,
                 eta,
                 monotonic,
                 **kwargs):
        """Construct layer."""
        # Whether to make monotonic function
        self.monotonic = monotonic

        # Create function that returns a kernel constraint
        # based on self.monotonic
        self.constraint = lambda: non_neg() if self.monotonic else None

        # Set number of hidden nodes
        self.K = K

        # Set regularizer
        self.eta = eta
        self.regularizer = tf.keras.regularizers.L2(self.eta)

        # Call superclass constructor
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a_0 = self.add_weight(name='a_0',
                                   shape=(1,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)

        # Need to randomly initialize b_k
        b_k_dist = expon(scale=1 / self.K)
        self.b_k = self.add_weight(name='b_k',
                                   shape=(self.K,),
                                   initializer=Constant(b_k_dist.rvs(self.K)),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.c_k = self.add_weight(name='c_k',
                                   shape=(self.K,),
                                   initializer=Constant(1.),
                                   trainable=True,
                                   constraint=self.constraint(),
                                   regularizer=self.regularizer)

        self.d_k = self.add_weight(name='d_k',
                                   shape=(self.K,),
                                   initializer=Constant(0.),
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    # Just an alias for phi_to_yhat, for tensors only
    def call(self, phi):
        """Transform layer inputs to outputs."""
        return self.phi_to_yhat(phi)

    # yhat as function of phi
    @ handle_arrays
    def phi_to_yhat(self, phi):
        """Compute yhat from phi."""
        b_k = tf.reshape(self.b_k, [1, -1])
        c_k = tf.reshape(self.c_k, [1, -1])
        d_k = tf.reshape(self.d_k, [1, -1])
        phi = tf.reshape(phi, [-1, 1])
        yhat = self.a_0 + tf.reshape(
            K.sum(b_k * tanh(c_k * phi + d_k), axis=1), shape=[-1, 1])
        return yhat

class NoiseModelLayer(Layer):
    """Generic class representing the noise model of a GE model."""

    def __init__(self,
                 info_for_layers_dict,
                 polynomial_order=2,
                 eta_regularization=0.01,
                 is_noise_model_empirical=False,
                 **kwargs):
        """Construct class instance."""
        # order of polynomial which defines log_sigma's dependence on y_hat
        self.K = polynomial_order
        self.eta = eta_regularization
        self.info_for_layers_dict = info_for_layers_dict
        self.regularizer = tf.keras.regularizers.L2(self.eta)
        self.is_noise_model_empirical = is_noise_model_empirical
        super().__init__(**kwargs)

    def get_config(self):
        """Return configuration dictionary."""
        base_config = super().get_config()
        return {**base_config,
                'info_for_layers_dict': self.info_for_layers_dict,
                'polynomial_order': self.K,
                'eta_regularization': self.eta}

    def build(self, input_shape):
        """Build layer."""
        super().build(input_shape)

    def call(self, inputs):
        """
        Transform layer inputs to outputs.

        Parameters
        ----------
        inputs: (tf.Tensor)
            A (B,2) tensor containing both yhat and ytrue, where B is batch
            size.

        Returns
        -------
            A (B,) tensor containing negative log likelihood contributions
        """

        # variable represent user supplied error bars, will get
        # instantiated if noise model is empirical
        self.dy = None

        # if noise model is not empirical, do everything as before
        # i.e., do not supply the additional argument of dy to compute nll
        if self.is_noise_model_empirical == False:

            # this is yhat
            yhat = inputs[:, 0:1]

            # these are the labels
            ytrue = inputs[:, 1:]

            # TODO replace NANs with something else (y.mean) and don't let these contribute to ll
            # replace the tensors where nans in ytrue occur with zeros, so that likelihood for
            # that yhat, ypred pair is also zero.
            # yhat = tf.where(tf.is_nan(ytrue), tf.zeros_like(yhat), yhat)
            # ytrue = tf.where(tf.is_nan(ytrue), tf.zeros_like(ytrue), ytrue)

            # Compute negative log likelihood
            nlls = self.compute_nlls(yhat=yhat,
                                     ytrue=ytrue)

        # if noise model is empirical, supply argument dy to compute nll
        # method in the Gaussian noise model derived class.
        else:
            # this is yhat
            yhat = inputs[:, 0:1]

            # these are the labels
            ytrue = inputs[:, 1:2]

            # these are user supplied error bars
            self.dy = inputs[:, 2:3]

            # Compute negative log likelihood
            nlls = self.compute_nlls(yhat=yhat,
                                     ytrue=ytrue,
                                     dy=self.dy)

        # Compute I_var metric from nlls
        H_y = self.info_for_layers_dict.get('H_y_norm', np.nan)
        H_y_given_phi = K.mean(np.log2(e) * nlls)
        I_var = H_y - H_y_given_phi
        self.add_metric(I_var, name="I_var", aggregation="mean")

        return nlls

    @ handle_arrays
    def p_of_y_given_yhat(self,
                          y,
                          yhat):
        """Compute p(y|yhat)."""
        # Compute negative log likeliihood
        nll_arr = self.compute_nlls(yhat=yhat,
                                    ytrue=y)

        # Convert to p_y_given_yhat and return
        p_y_given_yhat = Exp(-nll_arr)

        return p_y_given_yhat

    @ handle_arrays
    def sample_y_given_yhat(self, yhat):
        """Sample y values from p(y|yhat)."""
        # Draw random quantiles
        q = tf.constant(np.random.rand(*yhat.shape), dtype=tf.float32)

        # Compute corresponding quantiles and return
        y = self.yhat_to_yq(yhat, q)
        return y

    # The following functions must be overridden
    def compute_params(self, yhat, ytrue):
        """Compute the parameters governing p(y|yhat)."""
        assert False, 'Function must be overridden'

    def compute_nlls(self, yhat, ytrue):
        """Compute negative log likelihoods."""
        assert False, 'Function must be overridden'

    def yhat_to_yq(self, yhat, q):
        """Compute quantiles of p(y|yhat)."""
        assert False, 'Function must be overridden'

    def yhat_to_ymean(self, yhat):
        """Compute the mean of p(y|yhat)."""
        assert False, 'Function must be overridden'

    def yhat_to_ystd(self, yhat):
        """Compute the standard deviation of p(y|yhat)."""
        assert False, 'Function must be overridden'


class GaussianNoiseModelLayer(NoiseModelLayer):
    """Represents a Gaussian noise model for GE regression."""

    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a = self.add_weight(name='a',
                                 shape=(self.K + 1, 1),
                                 initializer="TruncatedNormal",
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    @ handle_arrays
    def compute_params(self, yhat):
        """Compute layer parameters governing p(y|yhat)."""
        # Have to treat 0'th order term separately because of NaN bug.
        logsigma = self.a[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            logsigma += self.a[k] * K.pow(yhat, k)

        return logsigma

    @ handle_arrays
    def compute_nlls(self, yhat, ytrue, use_arrays=False):
        """Compute negative log likelihood contributions for each datum."""
        # Compute logsigma and sigma
        logsigma = self.compute_params(yhat)
        sigma = Exp(logsigma)

        # Compute nlls
        nlls = 0.5 * K.square((ytrue - yhat) / sigma) \
            + logsigma \
            + 0.5 * np.log(2 * pi)

        return nlls

    @ handle_arrays
    def yhat_to_yq(self, yhat, q):
        """Compute quantiles for p(y|yhat)."""
        sigma = Exp(self.compute_params(yhat))
        yq = yhat + sigma * np.sqrt(2) * erfinv(2 * q.numpy() - 1)
        return yq

    @ handle_arrays
    def yhat_to_ymean(self, yhat):
        """Compute mean of p(y|yhat)."""
        return yhat

    @ handle_arrays
    def yhat_to_ystd(self, yhat):
        """Compute standard deviation of p(y|yhat)."""
        sigma = Exp(self.compute_params(yhat))
        return sigma


class EmpiricalGaussianNoiseModelLayer(NoiseModelLayer):
    """Represents an Empirical Gaussian noise model for GE regression."""

    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a = self.add_weight(name='a',
                                 shape=(self.K + 1, 1),
                                 initializer="TruncatedNormal",
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    @ handle_arrays
    def compute_params(self, yhat):
        """Compute layer parameters governing p(y|yhat)."""

        # Have to treat 0'th order term separately because of NaN bug.
        logsigma = self.a[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            logsigma += self.a[k] * K.pow(yhat, k)

        return logsigma

    @ handle_arrays
    def compute_nlls(self, yhat, ytrue, dy, use_arrays=False):
        """
        Compute negative log likelihood contributions for each datum.
        Note: User must pass dy, since noise model is 'Empirical'
        """
        # Compute logsigma and sigma

        # Set sigma equal to dy (user supplied error bars. )
        logsigma = Log(dy)

        # Compute nlls
        nlls = 0.5 * K.square((ytrue - yhat) / dy) \
            + logsigma \
            + 0.5 * np.log(2 * pi)

        return nlls

    # @handle_arrays
    # def yhat_to_yq(self, yhat, q, dy):
    #     """
    #     Compute quantiles for p(y|yhat).
    #     Note: User must pass dy, since noise model is 'Empirical'
    #     """
    #
    #     # Set sigma equal to dy (user supplied error bars. )
    #     sigma = dy.numpy().reshape(-1, 1)
    #
    #     yq = yhat + sigma * np.sqrt(2) * erfinv(2 * q.numpy() - 1)
    #     return yq

    @ handle_arrays
    def yhat_to_ymean(self, yhat):
        """Compute mean of p(y|yhat)."""
        return yhat

    # @handle_arrays
    # def yhat_to_ystd(self, yhat, dy):
    #     """This method is dumb because the user supplies dy, which is ystd."""
    #     return dy


class CauchyNoiseModelLayer(NoiseModelLayer):
    """Represents a Cauchy noise model for GE regression."""

    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.a = self.add_weight(name='a',
                                 shape=(self.K + 1, 1),
                                 initializer="TruncatedNormal",
                                 trainable=True,
                                 regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config, "a": self.a}

    # Overloading functions
    @ handle_arrays
    def compute_params(self, yhat):
        """Compute layer parameters governing p(y|yhat)."""
        # Have to treat 0'th order term separately because of NaN bug.
        loggamma = self.a[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            loggamma += self.a[k] * K.pow(yhat, k)

        return loggamma

    @ handle_arrays
    def compute_nlls(self, yhat, ytrue):
        """Compute negative log likelihood contributions for each datum."""
        # Compute loggamma
        loggamma = self.compute_params(yhat)

        # Compute nlls
        nlls = Log(Exp(2 * loggamma) + K.square(ytrue - yhat)) \
            - loggamma \
            + np.log(pi)

        return nlls

    @ handle_arrays
    def yhat_to_yq(self, yhat, q):
        """Compute quantiles of p(y|yhat)."""
        gamma = Exp(self.compute_params(yhat))
        yq = yhat + gamma * tf.math.tan(np.pi * (q - 0.5))
        return yq

    @ handle_arrays
    def yhat_to_ymean(self, yhat):
        """Compute mean of p(y|yhat)."""
        return yhat

    @ handle_arrays
    def yhat_to_ystd(self, yhat):
        """Compute standard devaition of p(y|yhat)."""
        sigma = Exp(self.compute_params(yhat))
        return sigma


class SkewedTNoiseModelLayer(NoiseModelLayer):
    """Represents a skewed-t noise model for GE regression."""

    def __init__(self, *args, **kwargs):
        """Construct layer instance."""
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        """Build layer."""
        self.w_a = self.add_weight(name='w_a',
                                   shape=(self.K + 1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.w_b = self.add_weight(name='w_b',
                                   shape=(self.K + 1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)

        self.w_s = self.add_weight(name='w_s',
                                   shape=(self.K + 1, 1),
                                   initializer="random_normal",
                                   trainable=True,
                                   regularizer=self.regularizer)
        super().build(input_shape)

    def get_config(self):
        """Get configuration dictionary."""
        base_config = super().get_config()
        return {**base_config, "w_a": self.w_a,
                "w_b": self.w_b, "w_s": self.w_s}

    # Compute the mode as a function of a and b
    @ handle_arrays
    def t_mode(self, a, b):
        """Compute mode of p(t|a,b)."""
        t_mode = (a - b) * K.sqrt(a + b) \
            / (K.sqrt(2 * a + 1) * K.sqrt(2 * b + 1))
        return t_mode

    # Compute mean
    @ handle_arrays
    def t_mean(self, a, b):
        """Compute mean of p(t|a,b)."""
        return 0.5 * (a - b) * np.sqrt(a + b) * np.exp(
            LogGamma(a - 0.5)
            + LogGamma(b - 0.5)
            - LogGamma(a)
            - LogGamma(b)
        )

    @ handle_arrays
    def t_std(self, a, b):
        """Compute standard devaition of p(t|a,b)."""
        t_expected = self.t_mean(a, b)
        tsq_expected = 0.25 * (a + b) * \
            ((a - b) ** 2 + (a - 1) + (b - 1)) / ((a - 1) * (b - 1))
        return K.sqrt(tsq_expected - t_expected ** 2)

    @ handle_arrays
    def t_quantile(self, q, a, b):
        """Compute quantiles of p(t|a,b)."""
        x_q = tf.constant(betaincinv(a.numpy(), b.numpy(), q.numpy()),
                          dtype=tf.float32)
        t_q = (2 * x_q - 1) * K.sqrt(a + b) / K.sqrt(1 - (2 * x_q - 1) ** 2)
        return t_q

    # Overloading functions
    @ handle_arrays
    def compute_params(self, yhat):
        """Compute layer parameters governing p(y|yhat)."""
        # Have to treat 0'th order term separately because of NaN bug.
        log_a = self.w_a[0]
        log_b = self.w_b[0]
        log_s = self.w_s[0]

        # Add higher order terms and return
        for k in range(1, self.K + 1):
            log_a += self.w_a[k] * K.pow(yhat, k)
            log_b += self.w_b[k] * K.pow(yhat, k)
            log_s += self.w_s[k] * K.pow(yhat, k)

        # Compute a, b, s in terms of trainable parameters
        a = Exp(log_a)
        b = Exp(log_b)
        s = Exp(log_s)

        # Clip values to keep a and b in safe ranges
        a = tf.clip_by_value(a, 0.01, np.inf)
        b = tf.clip_by_value(b, 0.01, np.inf)

        return a, b, s

    @ handle_arrays
    def compute_nlls(self, yhat, ytrue):
        """Compute negative log likelihood contributions for each datum."""
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute t_mode
        t_mode = self.t_mode(a, b)

        # Compute t values
        t = t_mode + (ytrue - yhat) / s

        # Compute useful argument
        arg = t / K.sqrt(a + b + K.square(t))

        # Compute negative log likelihood contributions
        nlls = -(
            (a + 0.5) * Log(1 + arg) +
            (b + 0.5) * Log(1 - arg) +
            -(a + b - 1) * Log(np.float32(2.0)) +
            -0.5 * Log(a + b) +
            LogGamma(a + b) +
            -LogGamma(a) +
            -LogGamma(b) +
            -Log(s)
        )

        # nlls =  -(a + 0.5) * Log(1.0 + arg)
        # nlls += -(b + 0.5) * Log(1.0 - arg)
        # nlls += (a + b - 1.0) * Log(np.float32(2.0))
        # nlls += 0.5 * Log(a + b)
        # nlls += -LogGamma(a + b)
        # nlls += LogGamma(a)
        # nlls += LogGamma(b)
        # nlls += Log(s)

        return nlls

    @ handle_arrays
    def yhat_to_yq(self, yhat, q):
        """Compute quantiles for p(y|yhat)."""
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute t_mode
        t_mode = self.t_mode(a, b)

        # Compute random t values
        t_q = self.t_quantile(q, a, b)

        # Compute and return y
        yq = (t_q - t_mode) * s + yhat
        return yq

    @ handle_arrays
    def yhat_to_ymean(self, yhat):
        """Compute mean of p(y|yhat)."""
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute mean and mode of t distribution
        t_mean = self.t_mean(a, b)
        t_mode = self.t_mode(a, b)

        # Compute ymean
        ymean = s * (t_mean - t_mode) + yhat

        return ymean

    @ handle_arrays
    def yhat_to_ystd(self, yhat):
        """Compute standard deviation of p(y|yhat)."""
        # Compute distribution parameters at yhat values
        a, b, s = self.compute_params(yhat)

        # Compute tstd
        tstd = self.t_std(a, b)

        # Compute and return ystd
        ystd = s * tstd

        return ystd
