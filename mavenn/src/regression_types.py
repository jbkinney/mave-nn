"""regression_types.py: Specialized classes for GE and MPA regression."""

# Standard imports
import numpy as np
import numbers

# Tensorflow imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Concatenate

# MAVE-NN imports
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_alphabet
from mavenn.src.layers.gpmap \
    import AdditiveGPMapLayer, PairwiseGPMapLayer, MultilayerPerceptronGPMap
from mavenn.src.layers.measurement_process_layers \
    import GlobalEpistasisLayer, \
        AffineLayer, \
        GaussianNoiseModelLayer, \
        EmpiricalGaussianNoiseModelLayer, \
        CauchyNoiseModelLayer, \
        SkewedTNoiseModelLayer, \
        DiscreteAgnosticMP


@handle_errors
class GlobalEpistasisModel:
    """
    Represents a global epistatsis model.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox']

    ge_nonlinearity_type: (str)
        Specifies the form of the GE nonlinearity. Options:
        "linear": An affine transformation from phi to yhat.
        "nonlinear": Allow and arbitrary nonlinear map from phi to yhat.

    ge_nonlinearity_monotonic: (boolean)
        Whether to use a monotonicity constraint in GE regression.
        This variable has no effect for MPA regression.

    ge_heteroskedasticity_order: (int)
        Order of the exponentiated polynomials used to make noise model
        parameters dependent on y_hat, and thus render the noise model
        heteroskedastic. Set to zero for a homoskedastic noise model.
        (Only used for GE regression).

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    initial_weights: (np.array)
        Numpy array of weights that gets set as initial weights of a model
        if not set to None.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 alphabet,
                 gpmap_type,
                 gpmap_kwargs,
                 ge_nonlinearity_monotonic,
                 ge_nonlinearity_type,
                 ohe_batch_size,
                 ge_heteroskedasticity_order,
                 theta_regularization,
                 eta_regularization,
                 custom_gpmap,
                 initial_weights):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.ge_nonlinearity_monotonic = ge_nonlinearity_monotonic
        self.ge_heteroskedasticity_order = ge_heteroskedasticity_order
        self.ohe_batch_size = ohe_batch_size
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ge_nonlinearity_type = ge_nonlinearity_type
        self.custom_gpmap = custom_gpmap
        self.initial_weights = initial_weights

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        #self.num_nodes_hidden_measurement_layer = None
        self.theta_gf = None
        self.ge_model = None

        # check that ge_nonlinearity_monotonic is a boolean.
        check(isinstance(self.ge_nonlinearity_monotonic,
                         (bool, np.bool_)),
              'ge_nonlinearity_monotonic must be a boolean')

        # check that ge_heteroskedasticity_order is an number
        check(isinstance(self.ge_heteroskedasticity_order, numbers.Integral),
              'ge_heteroskedasticity_order must be an integers')

        check(self.ge_heteroskedasticity_order >= 0,
              'ge_heteroskedasticity_order must be >= 0')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox', 'custom']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that ge_nonlinearity_type valid
        allowed_types = ['linear', 'nonlinear']
        check(self.ge_nonlinearity_type in allowed_types,
              f'gpmap_type = {self.gpmap_type};'
              f'must be in {allowed_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that eta regularization is a number
        check(isinstance(self.eta_regularization, numbers.Real),
              'eta must be a number')

        # check that theta regularization is greater than 0
        check(self.eta_regularization >= 0,
              'eta must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

    @handle_errors
    def define_model(self,
                     ge_noise_model_type,
                     ge_nonlinearity_hidden_nodes=50):
        """
        Establish model architecture.

        Defines the architecture of the global epistasis regression model.
        using the tensorflow.keras functional API.

        Parameters
        ----------
        ge_nonlinearity_hidden_nodes: (int)
            Number of hidden nodes (i.e. sigmoidal contributions) to use in the
            definition of the GE nonlinearity.

        ge_noise_model_type: (str)
            Specifies the type of noise model the user wants to infer.
            The possible choices allowed: ['Gaussian','Cauchy','SkewedT', 'Empirical']

        Returns
        -------
        model: (tf.model)
            TensorFlow model that can be compiled and subsequently fit to data.
        """
        # check that p_of_all_y_given_phi valid
        check(ge_noise_model_type in {'Gaussian', 'Cauchy', 'SkewedT', 'Empirical'},
              f'p_of_all_y_given_phi = {ge_noise_model_type};' 
              f'must be "Gaussian", "Cauchy", or "SkewedT"')

        check(isinstance(ge_nonlinearity_hidden_nodes, numbers.Integral),
              'ge_nonlinearity_hidden_nodes must be an integer.')

        check(ge_nonlinearity_hidden_nodes > 0,
              'ge_nonlinearity_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)

        # need to add one additional node for targets
        # but 2 additional nodes if noise model is empirical; 1 for y
        # and another for dy
        if ge_noise_model_type != 'Empirical':
            number_input_layer_nodes = number_x_nodes + 1
        else:
            number_input_layer_nodes = number_x_nodes + 2

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + 1],
            output_shape=((1,)),
            trainable=False, name='Labels_input')(inputTensor)

        # User supplied error bars
        if ge_noise_model_type == 'Empirical':

            dlabels_input = Lambda(
                lambda x: x[:, number_x_nodes+1:number_x_nodes + 2],
                output_shape=((1,)),
                trainable=False, name='dLabels_input')(inputTensor)


        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = AdditiveGPMapLayer(
                L=self.L,
		alphabet=self.alphabet,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = PairwiseGPMapLayer(
                L=self.L,
                alphabet=self.alphabet,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        elif self.gpmap_type == 'blackbox':
            self.x_to_phi_layer = MultilayerPerceptronGPMap(
                L=self.L,
                alphabet=self.alphabet,
                theta_regularization=self.theta_regularization,
                **self.gpmap_kwargs)
        elif self.gpmap_type == 'custom':
            self.x_to_phi_layer = self.custom_gpmap(**self.gpmap_kwargs)
        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Make global epistasis layer
        if self.ge_nonlinearity_type=='linear':
            self.phi_to_yhat_layer = \
                AffineLayer(eta=self.eta_regularization,
                            monotonic=self.ge_nonlinearity_monotonic)
        elif self.ge_nonlinearity_type=='nonlinear':
            self.phi_to_yhat_layer = \
                GlobalEpistasisLayer(K=ge_nonlinearity_hidden_nodes,
                                     eta=self.eta_regularization,
                                     monotonic=self.ge_nonlinearity_monotonic)
        else:
            assert False, 'This shouldnt happen'

        yhat = self.phi_to_yhat_layer(phi)

        # Concatenate yhat and training labels only if
        # noise model is not empirical, other concatenate
        # user error bars also.
        if ge_noise_model_type != 'Empirical':

            yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
                [yhat, labels_input])
        else:
            yhat_y_concat = Concatenate(name='yhat_and_y_to_ll')(
                [yhat, labels_input, dlabels_input])

        # Create noise model layer
        if ge_noise_model_type == 'Gaussian':
            self.noise_model_layer = GaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'Cauchy':
            self.noise_model_layer = CauchyNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'SkewedT':
            self.noise_model_layer = SkewedTNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization)

        elif ge_noise_model_type == 'Empirical':
            self.noise_model_layer = EmpiricalGaussianNoiseModelLayer(
                info_for_layers_dict=self.info_for_layers_dict,
                polynomial_order=self.ge_heteroskedasticity_order,
                eta_regularization=self.eta_regularization,
                is_noise_model_empirical=True)
        else:
            assert False, 'This should not happen.'

        outputTensor = self.noise_model_layer(yhat_y_concat)

        # create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.ge_nonlinearity_hidden_nodes = ge_nonlinearity_hidden_nodes

        # set initial weights if provided.
        if self.initial_weights is not None:

            # check(isinstance(self.initial_weights, np.ndarray),
            #       f'type(initial_weights)={type(self.initial_weights)}; must be a numpy array.')

            model.set_weights(self.initial_weights)

        return model


@handle_errors
class MeasurementProcessAgnosticModel:
    """
    Represents a measurement process agnostic model.

    Parameters
    ----------
    sequence_length: (int)
        Integer specifying the length of a single training sequence.

    number_of_bins: (int)
        Integer specifying the number of bins. (Only used for MPA regression).

    alphabet: (str)
        Specifies the type of input sequences. Three possible choices
        allowed: ['dna','rna','protein', 'protein*'].

    gpmap_type: (str)
        Specifies the type of G-P model the user wants to infer.
        Possible choices: ['additive','neighbor','pairwise','blackbox', 'custom']

    ohe_batch_size: (int)
        Integer specifying how many sequences to one-hot encode at a time.
        The larger this number number, the quicker the encoding will happen,
        but this may also take up a lot of memory and throw an exception
        if its too large. Currently for additive models only.

    theta_regularization: (float >= 0)
        Regularization strength for G-P map parameters theta.

    eta_regularization: (float >= 0)
        Regularization strength for measurement process parameters eta.

    custom_gpmap: (GPMapLayer sub-class)
        Defines custom gpmap, provided by user. Inherited class of GP-MAP layer,
        which defines the functionality for x_to_phi_layer.

    initial_weights: (np.array)
        Numpy array of weights that gets set as initial weights of a model
        if not set to None.
    """

    @handle_errors
    def __init__(self,
                 info_for_layers_dict,
                 sequence_length,
                 number_of_bins,
                 gpmap_type,
                 gpmap_kwargs,
                 alphabet,
                 theta_regularization,
                 eta_regularization,
                 ohe_batch_size,
                 custom_gpmap,
                 initial_weights):
        """Construct class instance."""
        # set class attributes
        self.info_for_layers_dict = info_for_layers_dict
        self.gpmap_type = gpmap_type
        self.gpmap_kwargs = gpmap_kwargs
        self.alphabet = validate_alphabet(alphabet)
        self.C = len(self.alphabet)
        self.theta_regularization = theta_regularization
        self.eta_regularization = eta_regularization
        self.ohe_batch_size = ohe_batch_size
        self.number_of_bins = number_of_bins
        self.custom_gpmap = custom_gpmap
        self.initial_weights = initial_weights

        # class attributes that are not parameters
        # but are useful for using trained models
        self.history = None
        self.model = None

        # the following set of attributes are used for
        # gauge fixing the neural network model (x_to_phi and measurement)
        # and are set after the model has been fit to data.
        self.mpa_hidden_nodes = None
        self.theta_gf = None
        self.na_model = None

        # check that L is an number
        check(isinstance(sequence_length, numbers.Integral),
              'L must be an integer')

        # check that number_of_bins is an number
        check(isinstance(self.number_of_bins, numbers.Integral),
              'number_of_bins must be an integer')

        # check that model type valid
        valid_gpmap_types = ['additive', 'neighbor', 'pairwise', 'blackbox', 'custom']
        check(self.gpmap_type in valid_gpmap_types,
              f'model_type = {self.gpmap_type}; must be in {valid_gpmap_types}')

        # check that theta regularization is a number
        check(isinstance(self.theta_regularization, numbers.Real),
              'theta_regularization must be a number')

        # check that theta regularization is greater than 0
        check(self.theta_regularization >= 0,
              'theta_regularization must be >= 0')

        # check that ohe_batch_size is an number
        check(isinstance(self.ohe_batch_size, numbers.Integral),
              'ohe_batch_size must be an integer')

        # check that ohe_batch_size is > 0
        check(self.ohe_batch_size > 0,
              'ohe_batch_size must be > 0')

        # record sequence length for convenience
        self.L = sequence_length

        # Record number of bins
        self.Y = number_of_bins
        self.all_y = np.arange(self.Y).astype(int)

    def define_model(self,
                     mpa_hidden_nodes=10):
        """
        Define the neural network architecture of the MPA model.

        Uses the tensorflow.keras functional API. If custom_architecture is
        not None, this is used instead as the model architecture.

        Parameters
        ----------
        mpa_hidden_nodes: (int)
            Number of nodes to use in the hidden layer of the measurement
            network of the GE model architecture.

        Returns
        -------
        model: (tf.model)
            A tensorflow model that can be compiled and subsequently
            fit to data.
        """
        check(isinstance(mpa_hidden_nodes, numbers.Integral),
              'mpa_hidden_nodes must be a number.')

        check(mpa_hidden_nodes > 0,
              'mpa_hidden_nodes must be greater than 0.')

        # Compute number of sequence nodes. Useful for model construction below.
        number_x_nodes = int(self.L*self.C)
        number_input_layer_nodes = number_x_nodes+self.number_of_bins

        inputTensor = Input((number_input_layer_nodes,),
                            name='Sequence_labels_input')

        sequence_input = Lambda(lambda x: x[:, 0:number_x_nodes],
                                output_shape=((number_x_nodes,)),
                                name='Sequence_only')(inputTensor)
        labels_input = Lambda(
            lambda x: x[:, number_x_nodes:number_x_nodes + self.number_of_bins],
            output_shape=((1,)),
            trainable=False,
            name='Labels_input')(inputTensor)

        # Create G-P map layer
        if self.gpmap_type == 'additive':
            self.x_to_phi_layer = AdditiveGPMapLayer(
                L=self.L,
                alphabet=self.alphabet,
                theta_regularization=self.theta_regularization)
        elif self.gpmap_type in ['pairwise', 'neighbor']:
            self.x_to_phi_layer = PairwiseGPMapLayer(
                L=self.L,
                alphabet=self.alphabet,
                theta_regularization=self.theta_regularization,
                mask_type=self.gpmap_type)
        elif self.gpmap_type == 'blackbox':
            self.x_to_phi_layer = MultilayerPerceptronGPMap(
                L=self.L,
                alphabet=self.alphabet,
                theta_regularization=self.theta_regularization,
                **self.gpmap_kwargs)
        elif self.gpmap_type == 'custom':
            self.x_to_phi_layer = self.custom_gpmap(**self.gpmap_kwargs)

        else:
            assert False, "This should not happen."
        phi = self.x_to_phi_layer(sequence_input)

        # Create concatenation layer
        self.layer_concatenate_phi_ct = Concatenate(name='phi_and_ct')
        phi_ct = self.layer_concatenate_phi_ct([phi, labels_input])

        # Create measurement process layer
        self.layer_measurement_process = DiscreteAgnosticMP(
            info_for_layers_dict=self.info_for_layers_dict,
            Y=self.number_of_bins,
            K=mpa_hidden_nodes,
            eta=self.eta_regularization
            )
        outputTensor = self.layer_measurement_process(phi_ct)

        #create the model:
        model = Model(inputTensor, outputTensor)
        self.model = model
        self.mpa_hidden_nodes = mpa_hidden_nodes

        # set initial weights if provided.
        if self.initial_weights is not None:

            # check(isinstance(self.initial_weights, np.ndarray),
            #      f'type(initial_weights)={type(self.initial_weights)}; must be a numpy array.')

            model.set_weights(self.initial_weights)

        return model
