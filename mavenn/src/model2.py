"""
2022.02.04
----------
model2.py: Defines the Model() class, which represents all MAVE-NN2 models.
Unlike version, this model class contains new features such as custom
measurement processes, new measurement processes such as tite-seq,
multi-latent phenotype models, and more.

Pseudocode showing some of the updated workflow and new features

# Define GP map (sets dimensionality of latent phenotype phi)
gpmap = ThermodynamicGPMap(...)

# Define measurement processes (specify dimensions of phi and form of y)
mp_ge = GEMeasurementProcess(...)
mp_mpa = MPAMeasurementProcess(...)

# Define model
model = Model(gpmap = gpmap,
              mplist = [mp_ge, mp_mpa])

# Set data
model.set_data(x = x,
               y_list = [y_ge, y_mpa],
               validation_flags = validation_flags)

# Fit model
model.fit(...)
"""

# TODO: need to define helper function's for model in keynote (e.g. mp_ge.phi_to_yhat ... )
# TODO: need to finish implementation of mp_list (using Ammardev branch)
# TODO: need to finish updating various gpmap implementations (e.g., pairwise, custom, and refactor x to x_ohe)

# Standard imports
import numpy as np
import pandas as pd
import pdb
import pickle
import time
import numbers

# Tensorflow imports
# note model import has to be imported using an alias to avoid
# conflict with the Model class.
# Tensorflow imports
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model as TF_Functional_Model
from tensorflow.keras.layers import Input, Lambda, Concatenate

# MAVE-NN imports
from mavenn import TINY
from mavenn.src.utils import check, handle_errors
from mavenn.src.layers.input_layer import InputLayer
from mavenn.src.layers.measurement_process_layers \
    import GlobalEpsistasisMP, \
           GaussianNoiseModelLayer, \
           EmpiricalGaussianNoiseModelLayer, \
           CauchyNoiseModelLayer, \
           SkewedTNoiseModelLayer, \
           DiscreteAgnosticMP

from mavenn.src.validate import validate_seqs, \
    validate_1d_array, \
    validate_alphabet

from mavenn.src.utils import mat_data_to_vec_data, \
    vec_data_to_mat_data, \
    x_to_stats, \
    p_lc_to_x, _x_to_mat, \
    only_single_mutants

from mavenn.src.entropy import mi_continuous, mi_mixed, entropy_continuous
from mavenn.src.reshape import _shape_for_output, \
    _get_shape_and_return_1d_array, \
    _broadcast_arrays

# Scipy imports
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsmr
from scipy.stats import spearmanr

class Model:

    """
     Represents a MAVE-NN (version 2) model, which includes a genotype-phenotype (G-P) map
     as well as a list of measurement processes.

     Parameters
     ----------
     gpmap: (MAVE-NN gpmap)
         MAVE-NN's Genotype-phenotype object.

     mp_list: (list)
        List of measurement processes. 

    """

    @handle_errors
    def __init__(self,
                 gpmap,
                 mp_list,
                 additional_input_nodes=0):

        # set attributes required for defining a model
        self.gpmap = gpmap

        # Variable that contains number of additional nodes to add to input, if any.
        # e.g., for inputting categorical information like experiment id, or condition under
        # which sequence was assayed.
        self.additional_input_nodes = additional_input_nodes

        self.L = self.gpmap.L
        self.alphabet = self.gpmap.alphabet
        self.C = len(self.alphabet)

        self.mp_list = mp_list

        # define model
        self.model = self.define_model()

    def define_model(self):

        """
        Method that defines the neural network
        model using the gpmap and mp_list passed to
        the constructor
        """

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

        # Add additional/categorical nodes
        number_x_nodes += self.additional_input_nodes

        # get list of measurement processes
        measurement_processes_list = self.mp_list

        # determine number of target nodes for entire model
        number_of_targets = 0
        for output_layer_index in range(len(measurement_processes_list)):

            # if current measurement process object has yhat attribute
            # then output shape is 1, otherwise it is Y. This is a required
            # number of bins parameter for output layers
            # that have multiple nodes.
            current_mp = measurement_processes_list[output_layer_index]
            if hasattr(current_mp, 'yhat'):
                current_output_shape = 1
            else:
                current_output_shape = measurement_processes_list[output_layer_index].Y
            number_of_targets += current_output_shape

        # Get input layer tensor, the sequence input, and the labels input
        input_tensor, sequence_input = InputLayer(number_x_nodes,
                                                  number_of_targets).get_input_layer()

        # assign phi to gpmap input into constructor
        phi = self.gpmap(sequence_input)

        # loop over measurement processes and create a list of output tensors
        # that phi will be connected to. Currently output tensors get assigned
        # to the list, they are invoked separately based on whether they
        # output a regression value (yhat) or counts (like discrete agnostic MPA)
        # but this is subject to change

        # list that contains target labels which are initially passed in concantenated to input
        # sequences.
        labels_input = []

        # Here we build up lambda layers, on step at a time, which will be
        # fed to each of the measurement layers
        # We need two variables to map the target labels to their corresponding measurement processes

        start_pointer_define_model = number_x_nodes
        #end_pointer_define_model = None

        # Note the following issue need to fix the labels lambda layers below.
        # https://stackoverflow.com/questions/58965227/keras-custom-layer-issue-for-loop
        def get_labels_nodes(x, start, end):
            return x[:, start:end]

        for output_layer_index in range(len(measurement_processes_list)):

            # if current measurement process object has yhat attribute
            # then output shape is 1, otherwise it is Y. We must require
            # this number of bins (or Y) parameter for output layers
            # that have multiple nodes.
            current_mp = measurement_processes_list[output_layer_index]
            if hasattr(current_mp, 'yhat'):
                current_output_shape = 1
            else:
                current_output_shape = measurement_processes_list[output_layer_index].Y

            # assign end point, which is determined by the current the current mp's output shape
            end_pointer_define_model = start_pointer_define_model+current_output_shape

            temp_output_layer = Lambda(get_labels_nodes,
                                       arguments={'start': start_pointer_define_model, 'end': end_pointer_define_model},
                                       output_shape=((1,)), trainable=False,
                                       name='Labels_input_' + str(output_layer_index)
                                       )(input_tensor)

            # This snippet doesn't work, see stack overflow issue above for details on why.
            # temp_output_layer = Lambda(lambda x:
            #                               x[:, start_pointer_define_model:end_pointer_define_model],
            #                               #x[:, 1100:1101],
            #                               # the output shape below may need to be changed for MPA, to
            #                               # end_pointer - start_pointer
            #                               output_shape=((1,)), trainable=False,
            #                               name='Labels_input_' + str(output_layer_index))(input_tensor)

            labels_input.append(temp_output_layer)

            # update start pointer to be current end pointer for next iteration
            start_pointer_define_model = end_pointer_define_model

        output_tensor_list = []
        # index to keep track of names of different y_yhat_likelihood layers
        idx_y_and_yhat_ll = 0
        for current_mp in measurement_processes_list:

            # if measurement process object has yhat attribute
            # note prediction in GE is yhat, but MPA it would be phi
            if hasattr(current_mp, 'yhat'):

                yhat = current_mp.yhat(phi)

                # concatenate y_hat and y to pass into likelihood computation layer
                prediction_y_concat = Concatenate(name=f'yhat_and_y_to_ll_{idx_y_and_yhat_ll}')(
                    [yhat, labels_input[idx_y_and_yhat_ll]])

                idx_y_and_yhat_ll += 1
                output_tensor = current_mp.mp_layer(prediction_y_concat)
                output_tensor_list.append(output_tensor)
            else:
                # concatenate phi and counts to pass into likelihood computation layer
                prediction_y_concat = Concatenate()([phi, labels_input[idx_y_and_yhat_ll]])
                output_tensor = current_mp(prediction_y_concat)
                output_tensor_list.append(output_tensor)
                idx_y_and_yhat_ll += 1

        model = TF_Functional_Model(input_tensor, output_tensor_list)

        self.model = model

        return model

    @handle_errors
    def set_data(self,
                 x,
                 y_list,
                 additional_x=None,
                 dy=None,
                 ct=None,
                 validation_frac=.2,
                 validation_flags=None,
                 shuffle=True,
                 knn_fuzz=0.01,
                 verbose=True):
        """
        Set training data.

        Prepares data for use during training, e.g. by shuffling and one-hot
        encoding training data sequences. Must be called before ``Model.fit()``.

        Parameters
        ----------
        x: (np.ndarray)
            1D array of ``N`` sequences, each of length ``L``.

        y: (np.ndarray)
            Array of measurements.
            For GE models, ``y`` must be a 1D array of ``N`` floats.
            For MPA models, ``y`` must be either a 1D or 2D array
            of nonnegative ints. If 1D, ``y`` must be of length ``N``, and
            will be interpreted as listing bin numbers, i.e. ``0`` , ``1`` ,
            ..., ``Y-1``. If 2D, ``y`` must be of shape ``(N,Y)``, and will be   gb
            interpreted as listing the observed counts for each of the ``N``
            sequences in each of the ``Y`` bins.

        additional_x: (np.ndarray)
            Array of categorical variables append to each sequences to indicate
            a conidtion or experiment in which the sequence was assayed.

        dy : (np.ndarray)
            User supplied error bars associated with continuous measurements
            to be used as sigma in the Gaussian noise model.

        ct: (np.ndarray, None)
            Only used for MPA models when ``y`` is 1D. In this case, ``ct``
            must be a 1D array, length ``N``, of nonnegative integers, and
            represents the number  of observations of each sequence in each bin.
            Use ``y=None`` for GE models, as well as for MPA models when
            ``y`` is 2D.

        validation_frac (float):
            Fraction of observations to use for the validation set. Is
            overridden when setting ``validation_flags``. Must be in the range
            [0,1].

        validation_flags (np.ndarray, None):
            1D array of ``N`` boolean numbers, with ``True`` indicating which
            observations should be reserved for the validation set. If ``None``,
            the training and validation sets will be randomly assigned based on
            the value of ``validation_frac``.

        shuffle: (bool)
            Whether to shuffle the observations, e.g., to ensure similar
            composition of the training and validation sets when
            ``validation_flags`` is not set.

        knn_fuzz: (float>0)
            Amount of noise to add to ``y`` values before passing them to the
            KNN estimator (for computing I_var during training). Specifically,
            Gaussian noise with standard deviation ``knn_fuzz * np.std(y)`` is
            added to ``y`` values. This is needed to mitigate errors caused by
            multiple observations of the same sequence. Only used for GE
            regression.

        verbose: (bool)
            Whether to provide printed feedback.

        Returns
        -------
        None
        """

        # bind attributes to self so they can be used in other methods
        # like compute_parameter_uncertainties
        self.set_data_args = locals()
        self.set_data_args.pop('self')

        # Start timer
        set_data_start = time.time()

        # Validate x and set x
        x = validate_1d_array(x)
        x = validate_seqs(x, alphabet=self.alphabet)
        check(len(x) > 0, f'len(x)=={len(x)}; must be > 0')

        # TODO: validate for mavenn 2
        # Validate y, note that this doesn't
        # apply for MPA regression since y
        # is not a 1-d array in MPAR.
        # if self.regression_type == 'GE':
        #     y = validate_1d_array(y)
        #     check(len(x) == len(y), 'length of inputs (x, y) must be equal')
        #
        # elif self.regression_type == 'MPA':
        #     if y.ndim == 1:
        #         y, x = vec_data_to_mat_data(y_n=y, ct_n=ct, x_n=x)
        #     else:
        #         if isinstance(y, pd.DataFrame):
        #             y = y.values
        #         check(y.ndim == 2,
        #               f'y.ndim={y.ndim}; must be 1 or 2.')
        #
        # # Ensure empirical noise model conditions are set.
        # if self.ge_noise_model_type == 'Empirical':
        #
        #     # ensure the regression type is GE if noise model is empirical
        #     check(self.regression_type == 'GE',
        #           'Regression type must be "GE" for Empirical noise model.')
        #
        #     # if noise model is empirical ensure that dy is not None.
        #     check(dy is not None,
        #           'dy must not be None if noise model is Empirical and must be supplied.')
        #
        #     dy = validate_1d_array(dy)
        #
        #     check(len(y) == len(dy),
        #           'length of targets and error-bar array (y, dy) must be equal')
        #
        #     # set error bars.
        #     self.dy = dy.copy()

        # Set N
        self.N = len(x)
        # This list will get populated with target label data
        # in the loop a bit below. All elements of list will
        # get horizontally stacked afterwards.
        y_list_internal = []

        # length of y_list must be equal to length of measurement processes
        check(len(self.mp_list) == len(y_list), 'Length of y_list must be equal to mp_list ')

        # Use the following loop to building a dataframe containing all the labels data.
        # This will be passed to fit later.
        measurement_processes_list = self.mp_list
        # need to variables to map the target labels to their corresponding measurement processes

        number_x_nodes = int(self.L * self.C)
        start_pointer = number_x_nodes
        end_pointer = None

        # List which will contain y_stats for each measurement process
        self.y_stats_list = []

        for output_layer_index in range(len(measurement_processes_list)):

            current_y_stats = {}

            # if current measurement process object has yhat attribute
            # then mean normalize the output
            current_mp = measurement_processes_list[output_layer_index]
            if hasattr(current_mp, 'yhat'):

                # get current y
                current_y = y_list[output_layer_index]

                y_unique = np.unique(current_y)
                check(len(y_unique),
                      f'Only {len(y_unique)} unique y-values provided;'
                      f'At least 2 are requied')

                current_y_stats['y_mean'] = current_y.mean()
                current_y_stats['y_std'] = current_y.std()

                #current_y_norm = (current_y - current_y_stats['y_mean']) / current_y_stats['y_std']
                current_y_norm = current_y

                current_y_norm_reshaped = np.array(current_y_norm).reshape(-1, 1)

                # Subsample y_norm for entropy estimation if necessary
                N_max = int(1E4)
                if self.N > N_max:
                    z = np.random.choice(a=current_y_norm_reshaped.squeeze(),
                                         size=N_max,
                                         replace=False)
                else:
                    z = current_y_norm_reshaped.squeeze()

                # Add some noise to aid in entropy estimation
                z += knn_fuzz * z.std(ddof=1) * np.random.randn(z.size)

                # Compute entropy
                H_y_norm, dH_y = entropy_continuous(z, knn=7, resolution=0)
                H_y = H_y_norm + np.log2(current_y_stats['y_std'] + TINY)

                self.mp_list[output_layer_index].info_for_layers_dict['H_y'] = H_y
                self.mp_list[output_layer_index].info_for_layers_dict['H_y_norm'] = H_y_norm
                self.mp_list[output_layer_index].info_for_layers_dict['dH_y'] = dH_y

                self.y_stats = {}

                self.y_stats['y_mean'] = current_y.mean()
                self.y_stats['y_std'] = current_y.std()

                self.y_std = current_y.std()
                self.y_mean = current_y.mean()
            else:
                # binned data here
                current_y_stats['y_mean'] = 1
                current_y_stats['y_std'] = 0

                current_y_norm = y_list[output_layer_index]

                # Check that none of the y-rows sum to zero
                # Throw an error if there are.
                num_zero_ct_rows = sum(current_y_norm.sum(axis=1) == 0)

                # TODO need to fix this for multi-headed discrete agnostic regression
                # check(num_zero_ct_rows == 0,
                #       f'Found {num_zero_ct_rows} sequences that have no counts.'
                #       f'There cannot be any such sequences.')

                # Compute naive entropy estimate
                # Should probably be OK in most cases
                # Ideally we'd use the NSB estimator
                c_y = current_y_norm.sum(axis=0).squeeze()
                p_y = c_y / c_y.sum()
                ix = p_y > 0
                H_y_norm = -np.sum(p_y[ix] * np.log2(p_y[ix] + TINY))
                H_y = H_y_norm + np.log2(current_y_stats['y_std']+ TINY)
                dH_y = 0  # Need NSB to estimate this well

                self.mp_list[output_layer_index].info_for_layers_dict['H_y'] = H_y
                self.mp_list[output_layer_index].info_for_layers_dict['H_y_norm'] = H_y_norm
                self.mp_list[output_layer_index].info_for_layers_dict['dH_y'] = dH_y

            # update y dictionary with current y
            y_list_internal.append(current_y_norm)

            # update y stats list
            self.y_stats_list.append(current_y_stats)

        # horizontally stack all targets together. These will be further hstacked
        # with x_ohe in fit later
        self.y = np.hstack(y_list_internal)

        # Set validation flags
        if validation_flags is None:
            self.validation_flags = (np.random.rand(self.N) < validation_frac)
        else:
            self.validation_flags = validation_flags
        self.validation_frac = self.validation_flags.sum()/self.N

        # Make sure x is valid
        x = validate_seqs(x, alphabet=self.alphabet)

        # Set training and validation x
        self.x = x.copy()
        print(f'In set data, x.shape = {self.x.shape}')
        # Provide feedback
        if verbose:
            print(f'N = {self.N:,} observations set as training data.')
            print(f'Using {100*self.validation_frac:.1f}% for validation.')

        # Shuffle data if requested
        check(isinstance(shuffle, bool),
              f"type(shuffle)={type(shuffle)}; must be bool.")
        if shuffle:
            ix = np.arange(self.N).astype(int)
            np.random.shuffle(ix)
            self.x = self.x[ix]
            self.validation_flags = self.validation_flags[ix]
            #if self.regression_type == 'GE':
            #    self.y = self.y[ix]
            #else:
            self.y = self.y[ix, :]
            if verbose:
                print('Data shuffled.')

        # Compute sequence statistics (only on training set)
        self.x_stats = x_to_stats(self.x, self.alphabet)

        # Extract one-hot encoding of sequences
        # This is what is passed to the network.
        self.x_ohe = self.x_stats.pop('x_ohe')

        # Extract consensus sequence
        self.x_consensus = self.x_stats['consensus_seq']

        # Instantiate this key as false, update if more than
        # single mutants founds (see lines below).
        self.x_stats['only_single_mutants'] = False

        # if categorical variables provided, append to one-hot encoding of sequences
        if additional_x is not None:
            self.x_ohe = np.hstack([self.x_ohe, additional_x])

        # Check if only single mutants found in training data.
        only_single_mutants_found = only_single_mutants(training_sequences=self.x,
                                                        consensus_sequence=self.x_consensus,
                                                        alphabet=self.alphabet)

        # If only single mutants found in training data, check conditions below.
        if only_single_mutants_found:

            check(self.ge_nonlinearity_type == 'linear',
                  f'Only single mutants found in training data, this condition requires '
                  f'"model.ge_nonlinearity_type == linear", '
                  f' value set for ge_nonlinearity_type = {self.ge_nonlinearity_type}')

            check(self.gpmap_type == 'additive',
                  f'Only single mutants found in training data, this condition requires '
                  f'"model.gpmap_type == additive", '
                  f' value set for gpmap_type = {self.gpmap_type}')

            self.x_stats['only_single_mutants'] = True

        if verbose:
            print(f'Time to set data: {time.time() - set_data_start:.3} sec.')

    @handle_errors
    def fit(self,
            epochs=50,
            learning_rate=0.005,
            validation_split=0.2,
            verbose=True,
            early_stopping=True,
            early_stopping_patience=20,
            batch_size=50,
            linear_initialization=True,
            freeze_theta=False,
            callbacks=[],
            try_tqdm=True,
            optimizer='Adam',
            optimizer_kwargs={},
            fit_kwargs={}):
        """
        Infer values for model parameters.

        Uses training algorithms from TensorFlow to learn model parameters.
        Before this is run, the training data must be set using
        ``Model.set_data()``.

        Parameters
        ----------
        epochs: (int)
            Maximum number of epochs to complete during model training.
            Must be ``>= 0``.

        learning_rate: (float)
            Learning rate. Must be ``> 0.``

        validation_split: (float in [0,1])
            Fraction of training data to reserve for validation.

        verbose: (boolean)
            Whether to show progress during training.

        early_stopping: (bool)
            Whether to use early stopping.

        early_stopping_patience: (int)
            Number of epochs to wait, after a minimum value of validation loss is
            observed, before terminating the model training process.

        batch_size: (None, int)
            Batch size to use for stochastic gradient descent and related
            algorithms. If None, a full-sized batch is used.
            Note that the negative log likelihood loss function used by MAVE-NN
            is extrinsic in batch_size.

        linear_initialization: (bool)
            Whether to initialize the results of a linear regression
            computation. Has no effect when ``gpmap_type='blackbox'``.

        freeze_theta: (bool)
            Whether to set the weights of the G-P map layer to be
            non-trainable. Note that setting ``linear_initialization=True``
            and ``freeze_theta=True`` will set theta to be initialized at the
            linear regression solution and then become frozen during training.

        callbacks: (list)
            Optional list of ``tf.keras.callbacks.Callback`` objects to use
            during training.

        try_tqdm: (bool)
            If true, mavenn will attempt to load the package `tqdm` and append
            `TqdmCallback(verbose=0)` to the `callbacks` list in order to
            improve the visual display of training progress. If
            users do not have tqdm installed, this will do nothing.

        optimizer: (str)
            Optimizer to use for training. Valid options include:
            ``'SGD'``, ``'RMSprop'``, ``'Adam'``, ``'Adadelta'``,
            ``'Adagrad'``, ``'Adamax'``, ``'Nadam'``, ``'Ftrl'``.

        optimizer_kwargs: (dict)
            Additional keyword arguments to pass to the
            ``tf.keras.optimizers.Optimizer`` constructor.

        fit_kwargs: (dict):
            Additional keyword arguments to pass to ``tf.keras.Model.fit()``

        Returns
        -------
        history: (tf.keras.callbacks.History)
            Standard TensorFlow record of the training session.
        """

        # bind attributes to self so they can be used in other methods
        # like compute_parameter_uncertainties
        self.fit_args = locals()
        self.fit_args.pop('self')

        # this is due to some tensorflow bug, if this key is not popped then pickling
        # fails during model save. The reason for this bug could be that callbacks is
        # passed in as an empty list but is appended to later ... or something else.
        self.fit_args.pop('callbacks', None)

        # Start timer
        start_time = time.time()

        # Check epochs
        check(isinstance(epochs, int),
              f'type(epochs)={type(epochs)}; must be int.')
        check(epochs > 0,
              f'epochs={epochs}; must be > 0.')

        # Check learning rate & set
        check(isinstance(learning_rate, float),
              f'type(learning_rate)={type(learning_rate)}; must be float.')
        check(learning_rate > 0,
              f'learning_rate={learning_rate}; must be > 0.')
        self.learning_rate = learning_rate

        # Check epochs
        check(isinstance(validation_split, float),
              f'type(validation_split)={type(validation_split)}; '
              f'must be float.')
        check(0 < validation_split < 1,
              f'validation_split={validation_split}; must be in (0,1).')

        # Check verbose
        check(isinstance(verbose, bool),
              f'type(verbose)={type(verbose)}; must be bool.')

        # Check early_stopping
        check(isinstance(early_stopping, bool),
              f'type(early_stopping)={type(early_stopping)}; must be bool.')

        # Check early_stopping_patience
        check(isinstance(early_stopping_patience, int),
              f'type(early_stopping_patience)={type(early_stopping_patience)};'
              f' must be int.')
        check(early_stopping_patience > 0,
              f'early_stopping_patience={early_stopping_patience};'
              f'must be > 0.')

        # Check/set batch size
        check(isinstance(batch_size, (int, None)),
              f'type(batch_size)={type(batch_size)}; must be int or None.')
        if batch_size is None:
            batch_size = len(self.x)
        else:
            check(batch_size > 0,
                  f'batch_size={batch_size}; must be > 0.')

        # Check linear_initialization
        check(isinstance(linear_initialization, bool),
              f'type(linear_initialization)={type(linear_initialization)};'
              f'must be bool.')
        self.linear_initialization = linear_initialization

        # Check freeze_theta
        check(isinstance(freeze_theta, bool),
              f'type(freeze_theta)={type(freeze_theta)};'
              f'must be bool.')
        self.freeze_theta = freeze_theta

        # Check callbacks
        check(isinstance(callbacks, list),
              f'type(callbacks)={type(callbacks)}; must be list.')

        # Add tdm if possible
        if try_tqdm:
            try:
                from tqdm.keras import TqdmCallback
                callbacks.append(TqdmCallback(verbose=0))
            #except ModuleNotFoundError:
            except:
                pass

        # Check optimizer
        check(isinstance(optimizer, str),
              f'type(optimizer)={type(optimizer)}; must be str')

        # Check optimizer_kwargs
        check(isinstance(optimizer_kwargs, dict),
              f'type(optimizer_kwargs)={type(optimizer_kwargs)}; must be dict.')

        # Make Optimizer instance with specified name and learning rate
        optimizer_kwargs['learning_rate'] = learning_rate
        optimizer = tf.keras.optimizers.get({"class_name": optimizer,
                                             "config": optimizer_kwargs})

        # Check optimizer_kwargs
        check(isinstance(fit_kwargs, dict),
              f'type(fit_kwargs)={type(fit_kwargs)}; must be dict.')

        # set the theta/weights of the G-P map to be non-trainable
        # if requested.
        if self.freeze_theta:

            self.layer_gpmap.trainable = False

        # Returns the sum of negative log likelihood contributions
        # from each sequence, which is provided as y_pred
        def likelihood_loss(y_true, y_pred):
            return K.sum(y_pred)

        self.model.compile(loss=likelihood_loss,
                           optimizer=optimizer)

        # Set early stopping callback if requested
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss',
                                           mode='auto',
                                           patience=early_stopping_patience))

        # Set parameters that affect models
        # self.y_mean = self.y_stats['y_mean']
        # self.y_std = self.y_stats['y_std']

        # Note: this is only true for GE regression
        y_targets = self.y

        # Set parameters that affect models
        # if self.y.shape[1] == 1:
        #     self.y_mean = self.y_stats['y_mean']
        #     self.y_std = self.y_stats['y_std']
        # else:
        #     self.y_mean = 0
        #     self.y_std = 1

        # Set y targets for linear regression and sign assignment

        # # If MPA regression, use mean bin number
        # elif self.regression_type == 'MPA':
        #     bin_nums = np.arange(self.Y)
        #     y_targets = (self.y_norm
        #                  * bin_nums[np.newaxis, :]).sum(axis=1) / \
        #         self.y_norm.sum(axis=1)
        #
        # else:
        #     assert False, "This should never happen."

        # Do linear regression if requested
        if self.linear_initialization:

            # Extract training data
            ix_val = self.validation_flags
            x_sparse_train = csc_matrix(self.x_ohe[~ix_val])
            y_targets_train = y_targets[~ix_val]

            # Do linear regression if gpmap_type is not custom.
            if self.gpmap_type != 'custom' and self.gpmap_type != 'thermodynamic':
                t = time.time()
                self.theta_lc_init = lsmr(x_sparse_train,
                                          y_targets_train,
                                          show=verbose)[0]

                linear_regression_time = time.time() - t
                if verbose:
                    print(f'Linear regression time: '
                          f'{linear_regression_time:.4f} sec')

            # Set weights from linear regression result
            if self.gpmap_type == 'additive':
                self.model.x_to_phi_layer.set_params(
                    theta_0=0.,
                    theta_lc=self.theta_lc_init)
            elif self.gpmap_type in ['neighbor', 'pairwise']:
                self.model.x_to_phi_layer.set_params(
                    theta_0=0.,
                    theta_lc=self.theta_lc_init,
                    theta_lclc=np.zeros([self.L, self.C, self.L, self.C]))
            elif self.gpmap_type == 'blackbox' or self.gpmap_type == 'custom':
                print(
                    f'Warning: linear initialization has no effect when gpmap_type={self.gpmap_type}.')
            else:
                assert False, "This should not happen."

        # Concatenate seqs and ys if noise model is not empirical
        #if self.ge_noise_model_type != 'Empirical':
        if True:
            train_sequences = np.hstack([self.x_ohe,
                                         self.y])
        # Concatenate seqs, ys, and dys if noise model is  empirical
        if False:
            train_sequences = np.hstack([self.x_ohe,
                                         self.y,
                                         self.dy.reshape(-1, 1)])

        # Get training and validation sets
        ix_val = self.validation_flags
        x_train = train_sequences[~ix_val, :]
        x_val = train_sequences[ix_val, :]

        # Todo: the following two work for continuous targets. The ones below are for MPA regression.
        # generalize for all types of regression.
        #y_train = self.y[~ix_val]
        #y_val = self.y[ix_val]

        y_train = self.y[~ix_val, :]
        y_val = self.y[ix_val, :]

        #if self.regression_type == 'GE':
        # if True:
        #     y_train = self.y_norm[~ix_val]
        #     y_val = self.y_norm[ix_val]

            # if noise model is empirical, then input to the model
            # will be x, y, dy, which will get split into
            # x_train, y_trian, dy_trian, and x_val, y_val, dy_val

            # TODO: need to test this implementation.
            # if self.ge_noise_model_type == 'Empirical':
            #     dy_train = self.dy[~ix_val]
            #     dy_val = self.dy[ix_val]
            #
            #     y_train = np.hstack([y_train, dy_train.reshape(-1, 1)])
            #     y_val = np.hstack([y_val, dy_val.reshape(-1, 1)])

        # elif self.regression_type == 'MPA':
        #     y_train = self.y_norm[~ix_val, :]
        #     y_val = self.y_norm[ix_val, :]

        # Using tqdm progress bar for training.
        # if tqdm_bar == True:
        #     callbacks.append(TqdmCallback(verbose=0))
        #     verbose = False

        # Train neural network using TensorFlow
        history = self.model.fit(x_train,
                                 y_train,
                                 validation_data=(x_val, y_val),
                                 epochs=epochs,
                                 verbose=verbose,
                                 callbacks=callbacks,
                                 batch_size=batch_size,
                                 **fit_kwargs)

        # Get function representing the raw gp_map
        self._unfixed_gpmap = K.function(
            [self.model.layers[1].input],
            [self.model.layers[2].output])

        # compute unfixed phi using the function unfixed_gpmap with
        # training sequences.
        # Hot-fix related to TF 2.4, 2020.12.18
        #unfixed_phi = self._unfixed_gpmap(self.x_ohe)[0].ravel()
        # Commented due to memory overflow for higher order GP maps.
        # unfixed_phi = self._unfixed_gpmap(train_sequences)[0].ravel()

        # Set stats
        #if self.normalize_phi:
        if False:
            self.unfixed_phi_mean = np.mean(unfixed_phi)
            self.unfixed_phi_std = np.std(unfixed_phi)

            # Flip sign if correlation of phi with y_targets is negative
            r, p_val = spearmanr(unfixed_phi, y_targets)
            if r < 0:
                self.unfixed_phi_std *= -1.

        else:
            self.unfixed_phi_mean = 0.0
            self.unfixed_phi_std = 1.0

        # update history attribute
        self.history = history.history

        # Compute training time
        self.training_time = time.time() - start_time

        #if verbose:
        print(f'Training time: {self.training_time:.1f} seconds')

        return history
