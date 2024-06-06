# Standard imports
import numpy as np
import pandas as pd
import glob
import pdb

# MAVE-NN imports
import mavenn
from mavenn.src.examples import load_example_dataset, load_example_model
from mavenn.src.validate import validate_alphabet
from mavenn.src.utils import load
from mavenn.src.error_handling import check, handle_errors
from mavenn.tests.testing_utils import test_parameter_values

# Need to incorporate into model before testing.
def test_validate_alphabet():
    """ 20.09.10 JBK """

    # Tests that should pass
    success_list = [
        'dna',
        'rna',
        'protein',
        'protein*',
        np.array(['A', 'B', 'C']),
        {'A', 'B', 'C'},
        ['A', 'B', 'C'],
        pd.Series(['A', 'B', 'C'])
    ]

    # Tests that should fail
    fail_list = [
        'xna',
        'protein-',
        ['A','B','A'],
        [],
        {'A':5},
        np.array([['A','B'],['C','D']]),
        np.arange(5),
        pd.Series([])
    ]

    # Run tests of validate_alphabet
    test_parameter_values(func=validate_alphabet,
                          var_name='alphabet',
                          fail_list=fail_list,
                          success_list=success_list)


# def test_get_1pt_variants():
#     """20.09.01 JBK"""
#
#     # Tests with alphabet='protein'
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['QYKL'],
#                           fail_list=['ACGU', 'QYKL*', '',
#                                      0, ['A', 'C', 'G', 'T']],
#                           alphabet='protein')
#
#     # Tests with wt_seq='QYKL'
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='alphabet',
#                           success_list=['protein','protein*',
#                                         ['Q', 'Y', 'K', 'L']],
#                           fail_list=['dna','rna','ACGU','',0,
#                                      ['Q', 'Y', 'K'], ['A', 'C', 'G', 'T']],
#                           wt_seq='QYKL')
#
#     # Test include_wt
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='include_wt',
#                           success_list=[True, False],
#                           fail_list=[0, None],
#                           wt_seq='QYKL', alphabet='protein')
#
#     # Singleton tests
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['ACGT'],
#                           fail_list=['ACGU'],
#                           alphabet='dna')
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['ACGU'],
#                           fail_list=['ACGT'],
#                           alphabet='rna')
#     test_parameter_values(func=get_1pt_variants,
#                           var_name='wt_seq',
#                           success_list=['QYKL*', 'QYKL'],
#                           fail_list=['ACGU'],
#                           alphabet='protein*')


def test_GlobalEpistasisModel():

    # load MPSA dataset for testing
    data_df = load_example_dataset('mpsa')
    #x, y = load_example_dataset()
    x = data_df['x'].values
    y = data_df['y'].values

    # test on subset of sequences
    x = x[0:1000].copy()
    y = y[0:1000].copy()

    L = len(x[0])

    # sequences arrays that fail when entered into mavenn.
    bad_x = 'x'
    bad_y = [1, 3, -2, 4.5]

    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['GE'],
                          gpmap_type='additive', alphabet='rna', L=L)

    # test labels parameter ge_nonlinearity_monotonic
    test_parameter_values(func=mavenn.Model, var_name='ge_nonlinearity_monotonic', fail_list=['True', -1],
                          success_list=[True,False], regression_type='GE',
                          gpmap_type='additive', alphabet='rna', L=L)

    # test labels parameter ge_nonlinearity_hidden_nodes
    test_parameter_values(func=mavenn.Model, var_name='ge_nonlinearity_hidden_nodes', fail_list=[0.6,-1,0],
                          success_list=[1,10,100], regression_type='GE',
                          gpmap_type='additive', alphabet='rna', L=L)

    # test parameter gpmap_type
    test_parameter_values(func=mavenn.Model, var_name='gpmap_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='GE', alphabet='rna', L=L)

    # test parameter ge_heteroskedasticity_order
    test_parameter_values(func=mavenn.Model, var_name='ge_heteroskedasticity_order', fail_list=['0', 0.1, -1],
                          success_list=[0, 1, 10], gpmap_type='additive',
                          regression_type='GE', alphabet='rna', L=L)

    # test parameter theta_regularization
    test_parameter_values(func=mavenn.Model, var_name='theta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive',
                          regression_type='GE', alphabet='rna', L=L)

    # test parameter eta
    test_parameter_values(func=mavenn.Model, var_name='eta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive',
                          regression_type='GE',alphabet='rna', L=L)

    # test parameter ohe_batch_size
    test_parameter_values(func=mavenn.Model, var_name='ohe_batch_size', fail_list=['0', -1, -0.1, 0],
                          success_list=[20000], gpmap_type='additive',
                          regression_type='GE', alphabet='rna', L=L)

    # Prep model to test mavenn.Model child methods
    model = mavenn.Model(regression_type='GE', L=L, alphabet='rna')
    model.set_data(x=x, y=y, verbose=False)
    model.fit(epochs=1, verbose=False)

    # test model.simulate_method parameter N
    #test_parameter_values(func=model.simulate_dataset, var_name='N', fail_list=['0', -1, -0.1, 0],
    #                      success_list=[10, 1000])
                          
    # test model.simulate_method parameter x
    #test_parameter_values(func=model.simulate_dataset, var_name='x', fail_list=['0', -1, -0.1, 0],
    #                      success_list=[x[0:5]])                          

    # test model.simulate_method parameter validation_frac
    #test_parameter_values(func=model.simulate_dataset, var_name='validation_frac', fail_list=['0', -1, -0.1, 0],
    #                      success_list=[0.5], N=10)

    # test model.simulate_method parameter test_frac
    #test_parameter_values(func=model.simulate_dataset, var_name='test_frac', fail_list=['0', -1, -0.1, 0],
    #                      success_list=[0.5], N=10)

    # TODO: using gauge='user' breaks, need to test with p_lc, and x_wt
    # test model.get_theta
    test_parameter_values(func=model.get_theta, var_name='gauge', fail_list=[0, 'lorentz'],
                          success_list=["none", "uniform", "empirical", "consensus"])


def test_NoiseAgnosticModel():

    # load MPSA dataset for testing
    #x, y, ct_n = load_example_dataset(name='Sort-Seq')
    data_df = load_example_dataset('sortseq')
    x = data_df['x'].values
    y = data_df.filter(regex='ct_*').values

    # test on subset of sequences
    x = x[0:1000].copy()
    y = y[0:1000].copy()

    L = len(x[0])
    Y = 10

    # sequences arrays that fail when entered into mavenn.
    bad_X = 'x'

    # could possibly check if all elements are numeric
    # but that could slow things down
    bad_y = [[1, 3, -2, 4.5]]
    # Need to check for nans in y

    # test sequences parameter X
    # test_parameter_values(func=mavenn.Model, var_name='x', fail_list=[bad_X], success_list=[x],
    #                       gpmap_type='additive', y=y, regression_type='MPA', alphabet='dna', ct_n=ct_n)


    # test labels parameter regression_type
    test_parameter_values(func=mavenn.Model, var_name='regression_type', fail_list=['polynomial'], success_list=['MPA'],
                          gpmap_type='additive', alphabet='dna', L=L,
                          Y=Y)

    # test labels parameter gpmap_type
    test_parameter_values(func=mavenn.Model, var_name='gpmap_type', fail_list=['standard'],
                          success_list=['additive', 'neighbor', 'pairwise'],
                          regression_type='MPA', alphabet='dna', L=L,
                          Y=Y)

    # test parameter mpa_hidden_nodes
    test_parameter_values(func=mavenn.Model, var_name='mpa_hidden_nodes', fail_list=['0', 0.1, -1, 0],
                          success_list=[1, 10], gpmap_type='additive',
                          regression_type='MPA', alphabet='dna', L=L,
                          Y=Y)

    # test parameter theta_regularization
    test_parameter_values(func=mavenn.Model, var_name='theta_regularization', fail_list=['0', -1, -0.1],
                          success_list=[0, 0.1, 10], gpmap_type='additive',
                          regression_type='MPA', alphabet='dna', L=L,
                          Y=Y)

    # test parameter ohe_batch_size
    test_parameter_values(func=mavenn.Model, var_name='ohe_batch_size', fail_list=['0', -1, -0.1, 0],
                          success_list=[20000], gpmap_type='additive',
                          regression_type='MPA', alphabet='dna', L=L,
                          Y=Y)


def test_load():

    """
    Method that tests the load method.
    """

    mavenn_dir = mavenn.__path__[0]
    model_dir = f'{mavenn_dir}/examples/models/'

    good_MPA_model_1 = model_dir + 'sortseq_full-wt_mpa_additive'
    good_GE_model_1 = model_dir + 'gb1_ge_additive'

    # Good GE model file
    #good_GE_model_1 = mavenn.__path__[0] +'/tests/model_files/test_GE_model_good'
    #good_MPA_model_1 = mavenn.__path__[0] + '/tests/model_files/test_MPA_model_good'

    test_parameter_values(func=load,
                          var_name='filename',
                          fail_list=[],
                          success_list=[good_GE_model_1,
                                        good_MPA_model_1])


@handle_errors
def test_x_to_phi(model, seq):
    x = seq
    phi = model.x_to_phi(x)
    check(isinstance(phi, float), f'phi is {type(phi)}, not a float')
    check(np.isfinite(phi), f'phi={phi} is not finite.')

    x = np.array(seq)
    phi = model.x_to_phi(x)
    check(isinstance(phi, float), f'phi is {type(phi)}, not a float')

    x = [seq,
         seq,
         seq]
    shape = (3,)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[seq, seq, seq]]
    shape = (1, 3)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[seq],
         [seq],
         [seq]]
    shape = (3, 1)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = [[[seq],
          [seq],
          [seq]]]
    shape = (1, 3, 1)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')

    x = np.array([seq, seq, seq])
    shape = (3,)
    phi = model.x_to_phi(x)
    check(isinstance(phi, np.ndarray), f'phi is {type(phi)}, not a np.ndarray')
    check(phi.shape == shape,
          f'phi={phi} does not have the expected shape={shape}')


@handle_errors
def test_x_to_yhat(model, seq):
    x = seq
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, float), f'yhat is {type(yhat)}, not a float')
    check(np.isfinite(yhat), f'yhat={yhat} is not finite.')

    x = np.array(seq)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, float), f'yhat is {type(yhat)}, not a float')

    x = [seq,
         seq,
         seq]
    shape = (3,)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[seq, seq, seq]]
    shape = (1, 3)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[seq],
         [seq],
         [seq]]
    shape = (3, 1)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = [[[seq],
          [seq],
          [seq]]]
    shape = (1, 3, 1)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')

    x = np.array([seq, seq, seq])
    shape = (3,)
    yhat = model.x_to_yhat(x)
    check(isinstance(yhat, np.ndarray),
          f'yhat is {type(yhat)}, not a np.ndarray')
    check(yhat.shape == shape,
          f'yhat={yhat} does not have the expected shape={shape}')


def test_x_to_phi_or_yhat():
    mavenn_dir = mavenn.__path__[0]
    model_dir = f'{mavenn_dir}/examples/models/'

    mpa_model = load(model_dir + 'sortseq_full-wt_mpa_additive')
    mpa_seq = 'AAAAAAAGTGAGTTAGCCAACTAATTAGGCACCGTACGCTTTATAGTTTTTGCTTCCGGCTCGTATGTTGTGTGG'
    mpa_seq_gap = 'AAAAAAAGT-AGTTAGCCAACTAATTAGGCACCGTACGCTTTATAGTTTTTGCTTCCGGCTCGTATGTTGTGTGG'
    ge_model = load(model_dir + 'gb1_ge_additive')
    ge_seq = 'QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    ge_seq_gap = 'QYKLILNGKTLK-ETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

    test_parameter_values(test_x_to_phi,
                          var_name='seq',
                          success_list=[ge_seq],
                          fail_list=[mpa_seq, ge_seq_gap],
                          model=ge_model)

    test_parameter_values(test_x_to_phi,
                          var_name='seq',
                          success_list=[mpa_seq],
                          fail_list=[ge_seq, mpa_seq_gap],
                          model=mpa_model)

    test_parameter_values(test_x_to_yhat,
                          var_name='seq',
                          success_list=[ge_seq],
                          fail_list=[mpa_seq, ge_seq_gap],
                          model=ge_model)

    test_parameter_values(test_x_to_yhat,
                          var_name='seq',
                          success_list=[],
                          fail_list=[mpa_seq],
                          model=mpa_model)

@handle_errors
def test_for_nan_in_model_methods(model, seqs, y, regression_type):

    """
    Method that evaluates model methods and checks
    if are any NANs in the output.

    parameters
    ----------
    model: (mavenn model)
        Mavenn model object whose methods will be used
        to compute various outputs.

    seqs: (array-like of strings)
        Sequences which will be input to the methods x_to_*.

    y: (array-like of floats)
        Observations/y-values corresponding to the seqs parameter.

    regression_type: (str)
        String specifying 'GE' or 'MPA' regression.

    returns
    -------
    None.
    """

    # sum the arrays produced by the following Model
    # methods and then use np.isnan together with check.
    check(np.isnan(np.sum(model.x_to_phi(seqs))) == False,
          'x_to_phi produced a NAN')

    check(np.isnan(np.sum(model.p_of_y_given_phi(y=y,
                                                 phi=model.x_to_phi(seqs)).ravel())) == False,
          'p_of_y_given_phi produce a NAN')

    #I, dI = model.I_predictive(x=seqs, y=y)
    #check(np.isnan(I) == False, 'Predictive information computed to NAN')
    #check(np.isnan(dI) == False, 'Error predictive information computed to NAN')

    if regression_type == 'MPA':

        # TODO: this method's broadcasting doesn't work as in GE.
        # This should be updated to work with the new MPA format
        # For now, pick a specific bin_number to test for NANs.
        bin_number = 5
        check(np.isnan(np.sum(model.p_of_y_given_x(y=bin_number, x=seqs).ravel())) == False,
              'p_of_y_given_x produce a NAN')

    # method applicable only for GE regression.
    elif regression_type == 'GE':
        check(np.isnan(np.sum(model.x_to_yhat(seqs))) == False,
              'x_to_yhat produced a NAN')

        check(np.isnan(np.sum(model.yhat_to_yq(model.x_to_yhat(seqs)).ravel())) == False,
              'yhat_to_yq produce a NAN')

        check(np.isnan(np.sum(model.phi_to_yhat(model.x_to_phi(seqs)).ravel())) == False,
              'phi to yhat produced a NAN')

        check(np.isnan(np.sum(model.p_of_y_given_yhat(y=y, yhat=model.x_to_yhat(seqs)).ravel())) == False,
              'p_of_y_given_yhat produced a NAN')

        check(np.isnan(np.sum(model.p_of_y_given_x(y=y, x=seqs).ravel())) == False,
              'p_of_y_given_x produce a NAN')




@handle_errors
def test_GE_fit():

    """
    Method that tests the fit method of the Model class
    for GE regression. Small subsets of data are used for
    training, for all combinations of gpmap_type(s) and
    GE noise models. Models are trained for one epoch
    and all Model method outputs are subsequently checked
    for NANs.

    parameters
    ----------
    None

    returns
    -------
    None

    """

    # Turn off warnings/retracing just for testing,
    # however need to figure out how to properly
    # handle these warnings.
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    GE_datasets = ['mpsa', 'gb1']

    GE_noise_model_types = ['Gaussian', 'Cauchy', 'SkewedT']
    gpmap_types = ['additive', 'neighbor', 'pairwise']

    # loop over GE datasets for testing fit.
    for ge_dataset in GE_datasets:

        data_df = mavenn.load_example_dataset(ge_dataset)

        # use small subset of data for quick training
        data_df = data_df.loc[0:100].copy()

        # get training set from data_df
        ix = (data_df['set']!='test')
        L = len(data_df['x'][0])
        train_df = data_df[ix]
        test_df = data_df[~ix]

        # set alpbabet according to dataset.
        if ge_dataset == 'mpsa':
            alphabet = 'rna'
        elif ge_dataset == 'gb1':
            alphabet = 'protein'

        # loop over different gpmap_types
        for gpmap_type in gpmap_types:

            # loop over different GE noise model types
            for GE_noise_model_type in GE_noise_model_types:
                #print(f'======= {gpmap_type} : {GE_noise_model_type} : {dataset} ========')

                # Define model
                model = mavenn.Model(regression_type='GE',
                                     L=L,
                                     alphabet=alphabet,
                                     gpmap_type=gpmap_type,
                                     ge_noise_model_type=GE_noise_model_type,
                                     ge_heteroskedasticity_order=2)

                # Set training data
                model.set_data(x=train_df['x'],
                               y=train_df['y'],
                               shuffle=True,
                               verbose=False)

                # Fit model to data
                _history = model.fit(epochs=1,
                                     linear_initialization=True,
                                     batch_size=200,
                                     verbose=False)

                # check model methods for NANs
                #print('Check for NANs in the output of model methods')
                print(f'Testing model inference with: \n'
                      f'gpmap_type={repr(gpmap_type)}, \n'
                      f'dataset={repr(ge_dataset)}, \n'
                      f'noise_model={repr(GE_noise_model_type)}')

                test_parameter_values(test_for_nan_in_model_methods,
                                      var_name='seqs',
                                      success_list=[test_df['x'].values],
                                      fail_list=[[np.nan]],
                                      model=model,
                                      y=test_df['y'],
                                      regression_type='GE')


@handle_errors
def test_MPA_fit():

    """
    Method that tests the fit method of the Model class
    for MPA regression. Small subsets of data are used for
    training, for all combinations of gpmap_type(s).
    Models are trained for one epoch and all Model
    method outputs are subsequently checked for NANs.

    parameters
    ----------
    None

    returns
    -------
    None

    """

    # turn off warnings/retracing just for testing
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    gpmap_types = ['additive', 'neighbor', 'pairwise']

    data_df = mavenn.load_example_dataset('sortseq')

    # use small subset of data for quick training
    data_df = data_df.loc[0:100].copy()

    # Comptue sequence length and number of bins
    L = len(data_df['x'][0])
    y_cols = [c for c in data_df.columns if 'ct_' in c]
    Y = len(y_cols)
    print(f'L={L}, Y={Y}')

    # Split into trianing and test data
    ix = (data_df['set'] != 'test')
    L = len(data_df['x'][0])
    train_df = data_df[ix]
    test_df = data_df[~ix]

    # loop over different gpmap_types
    for gpmap_type in gpmap_types:

        # Define model
        model = mavenn.Model(regression_type='MPA',
                             L=L,
                             Y=Y,
                             alphabet='dna',
                             gpmap_type=gpmap_type)

        model.set_data(x=train_df['x'].values,
                       y=train_df[y_cols].values,
                       verbose=False)

        # Fit model to data
        history = model.fit(epochs=1,
                            batch_size=250,
                            verbose=False)
        # check model methods for NANs
        #print('Check for NANs in the output of model methods')
        print(f'gpmap_type = {gpmap_type}, dataset = sortseq')
        print(f'Testing model inference with: \n'
              f'gpmap_type={repr(gpmap_type)}, \n'
              f'dataset="sortseq"')

        test_parameter_values(test_for_nan_in_model_methods,
                              var_name='seqs',
                              success_list=[test_df['x'].values],
                              fail_list=[[np.nan]],
                              model=model,
                              y=test_df[y_cols].values,
                              regression_type='MPA')

# @handle_errors
# def _test_phi_calculation(model_file):
#     # Load model (assumes .h5 extension)
#     model = mavenn.load(model_file[:-3])
#
#     # Get sequence
#     seq = model.x_stats['consensus_seq']
#
#     # Get alphabet
#     alphabet = model.model.alphabet
#     alphabet = validate_alphabet(alphabet)
#
#     # Explain test to user
#     print(
# f"""\nTesting phi calcluation
# model     : {model_file}
# gpmap_type: {model.gpmap_type}
# alphabet  : {model.alphabet}
# seq       : {seq}""")
#
#     # Get MPA model parameters
#     tmp_df = model.get_gpmap_parameters()
#
#     # Create theta_df
#     theta_df = pd.DataFrame()
#     theta_df['id'] = [name.split('_')[1] for name in tmp_df['name']]
#     theta_df['theta'] = tmp_df['value']
#     theta_df.set_index('id', inplace=True)
#     theta_df.head()
#
#     # Get model type
#     if model.gpmap_type == 'additive':
#         f = additive_model_features
#     elif model.gpmap_type in ['pairwise', 'neighbor']:
#         f = pairwise_model_features
#     else:
#         check(model.gpmap_type in ['additive', 'neighbor', 'pairwise'],
#               'Unrecognized model.gpmap_type: {model.gpmap_type}')
#
#     # Encode sequence features
#     x, names = f([seq], alphabet=alphabet)
#
#     # Create dataframe
#     x_df = pd.DataFrame()
#     x_df['id'] = [name.split('_')[1] for name in names]
#     x_df['x'] = x[0, :]
#     x_df.set_index('id', inplace=True)
#     x_df.head()
#
#     # Make sure theta_df and x_df have the same indices
#     x_ids = set(x_df.index)
#     theta_ids = set(theta_df.index)
#     check(x_ids >= theta_ids, f"theta features are not contained within x features.")
#
#     # Merge theta_df and x_df into one dataframe
#     df = pd.merge(left=theta_df, right=x_df, left_index=True, right_index=True,
#                   how='left')
#
#     # Make sure there are no nan entries
#     num_null_entries = df.isnull().sum().sum()
#     check(num_null_entries == 0,
#           f'x_df and theta_df do not agree; found {num_null_entries} null entries.')
#
#     # Compute phi from manual calculation
#     phi_check = np.sum(df['theta'] * df['x'])
#
#     # Compute phi using model method
#     phi_model = model.x_to_phi(seq)
#
#     check(np.isclose(phi_check, phi_model, atol=1E-5),
#           f'phi_check: {phi_check} != phi_model: {phi_model} for gpmap_type: {model.gpmap_type}')
#     print(
# f"""phi_model : {phi_model}
# phi_check : {phi_check}""")


# def test_phi_calculations():
#     mavenn_dir = mavenn.__path__[0]
#     model_dir = f'{mavenn_dir}/examples/models/'
#
#     # Get list of models in directory
#     model_files = glob.glob(model_dir + '*.h5')
#
#     test_parameter_values(func=_test_phi_calculation,
#                           var_name='model_file',
#                           success_list=model_files,
#                           fail_list=[])


# def test_load_example():
#
#     successful_which_list = [None, 'model', 'training_data', 'test_data']
#     fail_which_list = [0, 'xxx', True]
#
#     successful_dataset_names_list = [None, 'mpsa', 'sortseq', 'gb1']
#     incorrect_dataset_names_list = [0, 'xxx']
#
#     successful_model_names_list = ["gb1_ge_additive",
#                                    "mpsa_ge_pairwise",
#                                    "sortseq_mpa_additive"]
#
#     incorrect_model_names_list = [0, "gb1", 'xxx']
#
#     # test parameter which
#     test_parameter_values(func=load_example,
#                           var_name='which',
#                           success_list=successful_which_list,
#                           fail_list=fail_which_list)
#
#     # test parameter name, with which='test_data'
#     test_parameter_values(func=load_example,
#                           var_name='name',
#                           which='test_data',
#                           success_list=successful_dataset_names_list,
#                           fail_list=incorrect_dataset_names_list)
#
#     # test parameter name, with which='model'
#     test_parameter_values(func=load_example,
#                           var_name='name',
#                           which='model',
#                           success_list=successful_model_names_list,
#                           fail_list=incorrect_model_names_list)
#
