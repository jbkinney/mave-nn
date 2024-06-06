"""utils.py: Utility functions for MAVE-NN."""

# Standard imports
import numpy as np
import pandas as pd
import mavenn
import random
import pdb
import pickle
import time
import os

# Import Tensorflow
import tensorflow as tf

# Imports from MAVE-NN
from mavenn.src.error_handling import handle_errors, check
from mavenn.src.validate import validate_1d_array, validate_nd_array, \
    validate_alphabet, validate_seqs

@handle_errors
def load(filename, verbose=True):
        """
        Load a previously saved model.

        Saved models are represented by two files having the same root
        and two different extensions, ``.pickle`` and ``.h5``. The ``.pickle``
        file contains model metadata, including all information needed to
        reconstruct the model's architecture. The ``.h5`` file contains the
        values of the trained neural network weights.

        Parameters
        ----------
        filename: (str)
            File directory and root. Do not include extensions.

        verbose: (bool)
            Whether to print feedback.

        Returns
        -------
        loaded_model: (mavenn.Model)
            MAVE-NN model object.
        """
        # Load model
        print(filename)
        filename_pickle = filename + '.pickle'
        with open(filename_pickle, 'rb') as f:
            config_dict = pickle.load(f)

        # Create model object
        loaded_model = mavenn.Model(**config_dict['model_kwargs'])

        # Add in diffeomorphic mode fixing and standardization params
        loaded_model.unfixed_phi_mean = config_dict.get('unfixed_phi_mean', 0)
        loaded_model.unfixed_phi_std = config_dict.get('unfixed_phi_std', 1)
        loaded_model.y_mean = config_dict.get('y_mean', 0)
        loaded_model.y_std = config_dict.get('y_std', 1)
        loaded_model.x_stats = config_dict.get('x_stats', {})
        loaded_model.history = config_dict.get('history', None)
        loaded_model.info_for_layers_dict = \
            config_dict.get('info_for_layers_dict', {})

        #loaded_model.set_data_args = config_dict.get('set_data_args')
        loaded_model.fit_args = config_dict.get('fit_args')

        # Load and set weights
        filename_h5 = filename + '.h5'
        loaded_model.get_nn().load_weights(filename_h5)

        # Provide feedback
        if verbose:
            print(f'Model loaded from these files:\n'
                  f'\t{filename_pickle}\n'
                  f'\t{filename_h5}')

        # Return model
        return loaded_model


@handle_errors
def vec_data_to_mat_data(y_n,
                         ct_n=None,
                         x_n=None):
    """
    Transform from vector data format to matrix data format.

    Parameters
    ----------
    y_n: (np.ndarray)
        Array of N bin numbers y. Must be set by user.

    ct_n: (np.ndarray)
        Array N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (np.ndarray)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.

    Returns
    -------
    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.
    """
    # Note: this use of validate_1d_array is needed to avoid a subtle
    # bug that occurs when inputs are pandas series with non-continguous
    # indices
    y_n = validate_1d_array(y_n).astype(int)
    N = len(y_n)
    if x_n is not None:
        x_n = validate_1d_array(x_n)
    else:
        x_n = np.arange(N)

    if ct_n is not None:
        ct_n = validate_1d_array(ct_n).astype(int)
    else:
        ct_n = np.ones(N).astype(int)

    # Cast y as array of ints
    y_n = np.array(y_n).astype(int)
    N = len(x_n)

    # This case is only for loading data. Should be tested/made more robust
    if N == 1:
        # should do check like check(mavenn.load_model==True,...

        return y_n.reshape(-1, y_n.shape[0]), x_n

    # Create dataframe
    data_df = pd.DataFrame()
    data_df['x'] = x_n
    data_df['y'] = y_n
    data_df['ct'] = ct_n

    # Sum over repeats
    data_df = data_df.groupby(['x', 'y']).sum().reset_index()

    # Pivot dataframe
    data_df = data_df.pivot(index='x', columns='y', values='ct')
    data_df = data_df.fillna(0).astype(int)

    # Clean dataframe
    data_df.reset_index(inplace=True)
    data_df.columns.name = None

    # Get ct_my values
    cols = [c for c in data_df.columns if not c in ['x']]

    ct_my = data_df[cols].values.astype(int)

    # Get x_m values
    x_m = data_df['x'].values

    return ct_my, x_m

@handle_errors
def mat_data_to_vec_data(ct_my,
                         x_m=None):
    """
    Transform from matrix data format to vector data format.

    Parameters
    ----------
    ct_my: (2D array of ints)
        Matrix of counts.

    x_m: (array)
        Corresponding list of x-values.

    Returns
    -------
    y_n: (np.ndarray)
        Array of N bin numbers y. Must be set by user.

    ct_n: (np.ndarray)
        Array N counts, one for each (sequence,bin) pair.
        If None, a value of 1 will be assumed for all observations

    x_n: (np.ndarray)
        List of N sequences. If None, each y_n will be
        assumed to come from a unique sequence.
    """
    # Note: this use of validate_1d_array is needed to avoid a subtle
    # bug that occurs when inputs are pandas series with non-continguous
    # indices
    ct_my = validate_nd_array(ct_my).astype(int)
    check(ct_my.ndim == 2,
          f'ct_my.ndim={ct_my.ndim}; must be 2.')
    M, Y = ct_my.shape

    if x_m is not None:
        x_m = validate_1d_array(x_m)
    else:
        x_m = np.arange(M)

    # Create dataframe
    data_df = pd.DataFrame()
    y_cols = list(range(Y))
    data_df.loc[:, 'x'] = x_m
    data_df.loc[:, y_cols] = ct_my

    # Melt dataframe
    data_df = data_df.melt(id_vars='x',
                           value_vars=y_cols,
                           value_name='ct',
                           var_name='y')

    # Remove zero count rows
    ix = data_df['ct'] > 0
    data_df = data_df[ix].reset_index()
    data_df.sort_values(by=['ct','y'], ascending=False, inplace=True)

    # Get return values values
    x_n = data_df['x'].values
    y_n = data_df['y'].values
    ct_n = data_df['ct'].values

    return y_n, ct_n, x_n


@handle_errors
def x_to_alphabet(x, return_name=False):
    """
    Return the alphabet from which a set of sequences are drawn.

    Parameters
    ----------
    x: (array of str)
        Array of sequences (equal-length strings).

    return_name: (bool)
        Whether to return the name of the alphabet, as opposed to an np.array
        of characters.

    Returns
    -------
    alphabet: (np.array or str)
        If return_name=False, is an np.array of the characters in the identified
        alphabet. If no match is found with a pre-defined alphabet, returns
        a sorted array of unique characters in x. If return_name=True,
        is a str representing the name of the alphabet. If no pre-defined
        alphabet match exists, will be set to 'unknown'.
    """
    # Validate set of sequences with unknown alphabet
    x = validate_seqs(x, alphabet=None)

    # Get set of unique characters as the empirical alphabet
    empirical_alphabet = np.array(list(set(''.join(x))))
    empirical_alphabet.sort()

    # Find smallest matching alphabet if it exists
    alphabet = empirical_alphabet
    loss = np.inf
    name = "unknown"
    for n, a in alphabet_dict.items():
        if set(a) >= set(empirical_alphabet) and len(a) < loss:
            alphabet = a
            loss = len(a)
            name = n

    # Return the alphabet
    if return_name:
        return name
    else:
        return alphabet


@handle_errors
def p_lc_to_x(N, p_lc, alphabet):
    """
    Generate an array of N sequences given a probability matrix.

    Parameters
    ----------
    N: (int > 0)
        Number of sequences to generate

    p_lc: (np.array)
        An (L,C) array  listing the probability of each base (columns) for each
        position (rows).

    alphabet: (np.array)
        The alphabet, length C, from which sequences will be generated.

    Returns
    -------
    x: (np.array)
        An (N,) array of sequences drawn from p_lc
    """
    # Validate N
    check(isinstance(N, int),
          f'type(N)={type(N)}; must be int')
    check(N > 0,
          f'N={N}; must be > 0')

    # Validate p_lc
    check(isinstance(p_lc, np.ndarray),
          f'type(p_lc)={type(p_lc)}; must be np.ndarray.')
    check(p_lc.ndim == 2,
          f'p_lc.ndim={p_lc.ndim}; must be 2.')
    check(np.all(p_lc.ravel() >= 0),
          f'some elements of p_lc are negative.')
    check(np.all(p_lc.ravel() <= 1),
          f'some elements of p_lc are greater than 1.')
    L, C = p_lc.shape

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)
    check(len(alphabet) == C,
          f'len(alphabet)={len(alphabet)} does not match p_lc.shape[1]={C}')

    # Create function to fill in a set of characters given p_l
    def fill_x_arr_col(p_l):
        return np.random.choice(a=alphabet,
                                size=N,
                                replace=True,
                                p=p_l)

    # Create seqs
    x_arr = np.apply_along_axis(fill_x_arr_col, axis=1, arr=p_lc).T
    x = np.apply_along_axis(''.join, axis=1, arr=x_arr)

    return x


@handle_errors
def x_to_stats(x, alphabet, weights=None, verbose=False):
    """
    Identify the consensus sequence from a sequence alignment.

    Parameters
    ----------
    x: (np.ndarray)
        List of sequences. Sequences must all be the same length.

    alphabet: (np.ndarray)
        Alphabet from which sequences are drawn.

    weights: (None, np.ndarray)
        Weights for each sequence. E.g., count values, or numerical y values.
        If None, a value of 1 will be assumed for each sequence.

    verbose: (bool)
        Whether to print computation time.

    Returns
    -------
    consensus_seq: (str)
        Consensus sequence.
    """
    # Start timer
    start_time = time.time()

    # Validate alphabet
    alphabet = validate_alphabet(alphabet)

    # Validate sequences
    x = validate_seqs(x=x,
                      alphabet=alphabet,
                      restrict_seqs_to_alphabet=True)

    # Check weights and set if not provided
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = validate_1d_array(weights)
        weights = weights.astype(float)
        check(len(weights) == len(x),
              f"len(weights)={len(weights)} does not match len(x)={len(x)}")

    # Do one-hot encoding of sequences
    t = time.time()
    x_nlc = x_to_ohe(x,
                     alphabet,
                     check_seqs=False,
                     check_alphabet=False,
                     ravel_seqs=False)
    #print(f'Time for x_to_ohe: {time.time()-t:.3f} sec.')
    N, L, C = x_nlc.shape

    # Dictionary to hold results
    stats = {}

    # Compute x_ohe
    stats['x_ohe'] = x_nlc.reshape([N, L*C]).astype(np.int8)

    # Multiply by weights
    x_nlc = x_nlc.astype(float) * weights[:, np.newaxis, np.newaxis]

    # Compute lc encoding of consensus sequence
    x_sum_lc = x_nlc.sum(axis=0)
    x_sum_lc = x_sum_lc.reshape([L, C])
    x_support_lc = (x_sum_lc != 0)

    # Set number of sequences
    stats['N'] = N

    # Set sequence length
    stats['L'] = L

    # Set number of characters
    stats['C'] = C

    # Set alphabet
    stats['alphabet'] = alphabet

    # Compute probability matrix
    p_lc = x_sum_lc / x_sum_lc.sum(axis=1)[:, np.newaxis]
    stats['probability_df'] = pd.DataFrame(index=range(L),
                                           columns=alphabet,
                                           data=p_lc)

    # Compute sparsity factor
    stats['sparsity_factor'] = (x_nlc != 0).sum().sum() / x_nlc.size

    # Compute the consensus sequence and corresponding matrix.
    # Adding noise prevents ties
    x_sum_lc += 1E-1 * np.random.rand(*x_sum_lc.shape)
    stats['consensus_seq'] = \
        ''.join([alphabet[np.argmax(x_sum_lc[l, :])] for l in range(L)])

    # Compute mask dict
    missing_dict = {}
    for l in range(L):
        if any(~x_support_lc[l, :]):
            missing_dict[l] = ''.join(alphabet[~x_support_lc[l, :]])
    stats['missing_char_dict'] = missing_dict

    # Provide feedback if requested
    duration_time = time.time() - start_time
    if verbose:
        print(f'Stats computation time: {duration_time:.5f} sec.')

    return stats


@handle_errors
def set_seed(seed=1234):
    """
    Set random number generator seed; use to make training reproducible.

    Parameters
    ----------
    seed: (int)
        The value provided is used in np.random.seed(),
        tf.random.set_seed(), random.seed() and
        environ['PYTHONHASHSEED']. Default is seed=1234.

    Returns
    -------
    None.
    """
    # Check seed
    check(isinstance(seed, int),
          f'type(seed)={type(seed)}; must be int')

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@handle_errors
def x_to_ohe(x,
             alphabet,
             check_seqs=True,
             check_alphabet=True,
             ravel_seqs=True):
    """
    Convert a sequence array to a one-hot encoded matrix.

    Parameters
    ----------
    x: (np.ndarray)
        (N,) array of input sequences, each of length L

    alphabet: (np.ndarray)
        (C,) array describing the alphabet sequences are drawn from.

    check_seqs: (bool)
        Whether to validate the sequences

    check_alphabet: (bool)
        Whether to validate the alphabet

    ravel_seqs: (bool)
        Whether to return an (N, L*C) array, as opposed to an (N, L, C) array.

    Returns
    -------
    x_ohe: (np.ndarray)
        Array of one-hot encoded sequences, stored as np.int8 values.
    """
    # Validate alphabet as (C,) array
    if check_alphabet:
        alphabet = validate_alphabet(alphabet)

    # Validate sequences as (N,) array
    if check_seqs:
        x = validate_seqs(x, alphabet=alphabet)

    # Get dimensions
    L = len(x[0])
    N = len(x)
    C = len(alphabet)

    # Shape sequences as array of int8s
    x_arr = np.frombuffer(bytes(''.join(x), 'utf-8'),
                          np.int8, N * L).reshape([N, L])

    # Create alphabet as array of int8s
    alphabet_arr = np.frombuffer(bytes(''.join(alphabet), 'utf-8'),
                                 np.int8, C)

    # Compute (N,L,C) grid of one-hot encoded values
    x_nlc = (x_arr[:, :, np.newaxis] ==
             alphabet_arr[np.newaxis, np.newaxis, :]).astype(np.int8)

    # Ravel if requested
    if ravel_seqs:
        x_ohe = x_nlc.reshape([N, L * C])
    else:
        x_ohe = x_nlc

    return x_ohe


def only_single_mutants(training_sequences, consensus_sequence, alphabet):
    """
    Method that checks if training data contains only single mutants or not.

    training_sequences: (array)
        Array of sequences used for training models.

    consensus_sequence:
        Consensus sequence, evaluated using training data.
        Available as model.x_to_stats['consensus_seq'].

    alphabet: (np.array)
        The alphabet, length C, from which sequences will be generated.

    returns
    -------
    only_single_mutants_in_training_data: (bool)
        True if only single mutants founds compared to consensus
        sequence, false otherwise.
    """

    # Convert the consensus and training sequences to one-hot encoded.
    consensus_seq_ohe = x_to_ohe(consensus_sequence, alphabet)
    training_seqs_ohe = x_to_ohe(training_sequences, alphabet)

    # Calculate the hamming distance between all the training
    # sequences and the consensus sequence and store it in
    # hamming_dist array.
    hamming_dist = np.sum(np.bitwise_xor(
        training_seqs_ohe, consensus_seq_ohe), axis=1)/2.0

    # If all the elements of hamming_dist array are less than
    # and equal to one, then training set consists of
    # only single mutants.
    only_single_mutants_in_training_data = np.all(hamming_dist <= 1)

    return only_single_mutants_in_training_data


# Converts sequences to matrices
def _x_to_mat(x, alphabet):
    return (np.array(list(x))[:, np.newaxis] ==
            alphabet[np.newaxis, :]).astype(float)

@handle_errors
def split_dataset(data_df,
                  set_col='set',
                  train_set_name='training',
                  val_set_name='validation',
                  test_set_name='test',
                  verbose=True):
    """
    Splits dataset into
        (1) `trainval_df`: training + validation set
        (2) `train_df`: test set
    based on the value of the column `set_col`, which is then dropped. Also
    adds a `val_set_name` column to trainval_df indicating which data is to be
    reserved for validation (as opposed to gradient descent).

    Parameters
    ----------
    data_df: (pd.DataFrame)
        Dataset to split

    set_col: (str)
        Column of data_df indicating training, validation, or test set

    train_set_name: (str)
        Value in data_df[set_col] indicating allocation to training set

    val_set_name: (str)
        Value in data_df[set_col] indicating allocation to validation set

    test_set_name: (str)
        Value in data_df[set_col] indicating allocation to test set

    verbose: (bool)
        Whether to print out the size of the resulting datasets

    Returns
    -------
    trainval_df: (pd.DataFrame)
        Training + validation dataset. Contains a column named `val_set_name`
        indicating whether a row is allocated to the training or validation
        set.

    test_df: (pd.DataFrame)
        Test dataset.
    """

    # Specify training + validation sets
    trainval_ix = data_df[set_col].isin([train_set_name, val_set_name])
    trainval_df = data_df[trainval_ix].copy().reset_index(drop=True)
    trainval_df.insert(loc=0,
                       column=val_set_name,
                       value=trainval_df[set_col].eq(val_set_name))
    trainval_df.drop(columns=set_col, inplace=True)

    # Specify test set
    test_ix = data_df[set_col].eq(test_set_name)
    test_df = data_df[test_ix].copy().reset_index(drop=True)
    test_df.drop(columns=set_col, inplace=True)

    # Show dataset sizes
    if verbose:
        # Print number of data points
        num_data = len(data_df)
        num_train = sum(~trainval_df["validation"])
        pct_train = 100 * num_train / num_data
        num_val = sum(trainval_df["validation"])
        pct_val = 100 * num_val / num_data
        num_test = len(test_df)
        pct_test = 100 * num_test / num_data

        print(f'Training set   : {num_train:8,d} observations ({pct_train:7.2f}%)')
        print(f'Validation set : {num_val:8,d} observations ({pct_val:7.2f}%)')
        print(f'Test set       : {num_test:8,d} observations ({pct_test:7.2f}%)')
        print('-------------------------------------------------')
        print(f'Total dataset  : {num_data:8,d} observations ({100:7.2f}%)')
        print('')

    # return
    return trainval_df, test_df
