# -*- coding: utf-8 -*-
"""
Binary Logit Model with limited Mixed Logit capabilities:
    This file can be used to estimate 'standard' binary logit models as
    well as mixed logit models with alternative specific variance for one
    of the classes (i.e. the class with the least amount of people, which will
    be assumed to be the class with the label "1" as opposed to the label "0").

Created on Fri Jun 12 12:24:03 2015
@author: Timothy Brathwaite
"""

import pickle
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats

# Define boundary values for the calculations
integration_precision = 1e-9
max_comp_value = 1e300
min_comp_value = 1e-30
max_iterations = 1e5

def create_draws(mixing_distribution, num_draws, num_obs, halton_draws):    
    """
    mixing_distribution:    scipy.stats distribution. Draws are taken from this
                            distribution for the mixed logit model.
                            
    num_draws:              int. Number of draws from mixing_distribution to
                            use in estmating the mixed logit model.
                            
    num_obs:                int. The number of observations in the dataset.
    
    halton_draws:           bool. If True, the given number of draws are taken
                            using a Halton sequence. This should permit a much
                            smaller number of draws to be used per given level
                            of desired accuracy from the mixed logit model.
    ====================
    Returns:                numpy array of shape (num_obs, num_draws). This
                            array contains the num_draws draws from the passed
                            mixing distribution for each observation.
    """    
    if not halton_draws:
        # Take the draws directly from the mixing distribution    
        return mixing_distribution.rvs(size=(num_obs, num_draws))
    else:
        error="Asking for Halton draws and this feature isn't implemented yet"
        raise Exception(error)

def create_mixed_v(standard_v, current_std, draws):
    """
    standard_v:     numpy array of shape num_observations by 1. Elements of 
                    this array should be the systematic utility for each 
                    observation.
                    
    current_std:    float or int. Denotes the current coefficient being for
                    the normal distribution in the random error component of
                    the systematic utility.
                    
    draws:          numpy array of shape num_observations by num_draws.
                    Elements of this array should be individual draws from the
                    mixing distribution, for each observation.
    ====================
    Returns:        numpy array of shape num_observations by num_draws. Each
                    element of the array represents the utility of 
                    "alternative 1" for each observation, for each draw of the
                    mixing distribution, minus the type I extreme value error
                    term. 
    """        
    # Broadcast the standard systematic utilities and add them to the
    # Random error component of the systematic utility    
    all_v = standard_v + current_std * draws
    # As a precaution when v becomes too large, set it to a maximum value    
#    all_v[np.isinf(all_v)] = max_comp_value
    return all_v
    
def calc_drawn_probs(standard_v, current_std, draws):
    """
    standard_v:     numpy array of shape num_observations by 1. Elements of 
                    this array should be the systematic utility for each 
                    observation.
                    
    current_std:    float or int. Denotes the current coefficient being for
                    the normal distribution in the random error component of
                    the systematic utility.
                    
    draws:          numpy array of shape num_observations by num_draws.
                    Elements of this array should be individual draws from the
                    mixing distribution, for each observation.
    ====================
    Returns:        numpy array of shape num_observations by num_draws. Each
                    element of the array represents the probability of 
                    "alternative 1" being associated with each observation, 
                    for each draw of the mixing distribution.
    """        
    # Calculate the probability for every draw
    # Note that the complete systematic utilities for every draw    
    drawn_probabilities = 1.0 / (1.0 + np.exp(-1.0 * create_mixed_v(standard_v,
                                                                   current_std,
                                                                    draws)))
    # As a precaution when the probabilities are too large,
    # Set the simulated probabilitiese to a maximum value                                                                
    drawn_probabilities[drawn_probabilities >= 1.0] = 1 - min_comp_value
    
    # As a precaution when the drawn probabilities are too small
    # Set them to a minimum value
    drawn_probabilities[drawn_probabilities < min_comp_value] = min_comp_value
        
    return drawn_probabilities
    
def calc_simulated_probs(drawn_probabilities):
    """
    drawn_probabilities:    numpy array of shape num_observations by num_draws. 
                            Each element of the array represents the
                            probability of "alternative 1" being associated
                            with each observation, for each draw of the mixing
                            distribution.
    ====================
    Returns:                numpy array of shape num_observations by 1. EAch
                            element in the array is the average of the 
                            simulated probabilites for each draw for a given
                            individual.
    """    
    # Calculate the simulated probability by averaging over all of the
    # probabilities
    simulated_probs = drawn_probabilities.mean(axis=1,
                                               dtype=np.float64)[:, np.newaxis]
    
    # As a precaution when the simulated probabilities are too small
    simulated_probs[simulated_probs < min_comp_value] = min_comp_value
    
    # As a precaution when the simulated probabilities are too large
    simulated_probs[simulated_probs >= 1.0] = (1.0 - min_comp_value)
    
    # Make sure the simulated probabilities are of the correct magnitude.    
    assert np.all(simulated_probs > 0.0)
    try:
        assert np.all(simulated_probs < 1.0)
    except AssertionError as e:
        print "The maximum simulated probability is", simulated_probs.max()
        num_probs_above_1 = (simulated_probs >= 1.0).sum()
        probs_above_1_msg = "There are {:,} probabilities >= to 1.0"
        print probs_above_1_msg.format(num_probs_above_1)
        print "The simulated probabilities are"        
        print simulated_probs
        raise e
    
    return simulated_probs
    
def calc_prob_draw_derivatives(drawn_probabilities, x_matrix, 
                               pos_indicator, neg_indicator,
                               draws):   
    """
    drawn_probabilities:    numpy array of shape num_observations by num_draws. 
                            Each element of the array represents the
                            probability of "alternative 1" being associated
                            with each observation, for each draw of the mixing
                            distribution.
                            
    x_matrix:               A numpy array of shape (nObs, nPredictors). This
                            array contains all of the explanatory variables
                            being used to predict the probability that an
                            observation is of class 1.
                        
    pos_indicator:          A numpy array of shape (nObs, 1). The array 
                            indicates if a given observation is of class 1.
                    
    neg_indicator:          A numpy array of shape (nObs, 1). The array
                            indicates if a given observation is of class 0,
                            i.e. the negative class.
                    
    draws:                  numpy array of shape num_observations by num_draws.
                            Elements of this array should be individual draws
                            from the mixing distribution, for each observation.
    ====================
    Returns:                numpy array of shape num_observations by the
                            number of estimated coefficients. Each row in the
                            array represents the gradient of each 
                            observation's average simulated probability with 
                            respect to the individual coefficients being
                            estimated
    """    
    # Calculate the probability of a person making the choice that they chose
    # Should have shape nObs x nDraws
    prob_of_choice = drawn_probabilities * pos_indicator +\
                     (1.0 - drawn_probabilities) * neg_indicator
    # Make sure none of the predicted probabilities are zero
    assert np.all(prob_of_choice > 0)
    
    # Create "R" x-matrices, one for each probability draw.
    num_obs = x_matrix.shape[0]
    # Create "R" x-matrices which are populated for the first k-independent
    # variables with a column of zeros appended at the end
    x_part_1 = np.hstack((x_matrix, np.zeros((num_obs, 1))))[:, np.newaxis, :]
    # Create "R" x-matrices with all zeros for the first k-columns of
    # independent variables and a final column full of ones.
    # Multiply these matrices by the 
    x_part_2 = (draws[:, :, np.newaxis] * 
                np.hstack((np.zeros((num_obs, x_matrix.shape[1])),
                          np.ones((num_obs, 1))))[:,np.newaxis, :])
    # The full x_matrix has dimensions n_obs x n_draws x (n_ind_vars + 1)
    full_x_matrix = x_part_1 + x_part_2
    
    # Calculate the x_matrix where the x's corresponde to the x's of each
    # person's choice
    x_matrix_of_choice = full_x_matrix  * pos_indicator[:, :, np.newaxis]

    # Implement the derivative of the calculated probability for each 
    # independent variable for each draw
    prob_derivatives = (prob_of_choice[:, :, np.newaxis] * 
                        (x_matrix_of_choice - 
                         full_x_matrix * 
                         drawn_probabilities[:, :, np.newaxis]))
    
    return prob_derivatives.mean(axis=1, dtype=np.float64)
    
def calc_simulated_gradient(simulated_probs, drawn_probabilities,
                            x_matrix, beta,
                            pos_indicator, neg_indicator,
                            estimable_pos, ridging, draws):
    """
    simulated_probs:        numpy array of shape num_observations by 1. EAch
                            element in the array is the average of the 
                            simulated probabilites for each draw for a given
                            individual.
                            
    drawn_probabilities:    numpy array of shape num_observations by num_draws. 
                            Each element of the array represents the
                            probability of "alternative 1" being associated
                            with each observation, for each draw of the mixing
                            distribution.
                            
    x_matrix:               A numpy array of shape (nObs, nPredictors). This
                            array contains all of the explanatory variables
                            being used to predict the probability that an
                            observation is of class 1.
                            
    beta:                   A numpy array of shape (nPredictors,). It contains
                            the most recent guess about the value of the 
                            coefficients being used to predict the probability
                            that an observation is of class 1.
                        
    pos_indicator:          A numpy array of shape (nObs, 1). The array 
                            indicates if a given observation is of class 1.
                    
    neg_indicator:          A numpy array of shape (nObs, 1). The array
                            indicates if a given observation is of class 0,
                            i.e. the negative class.
                            
    estimable_pos:          A list of positions which indicate the columns of
                            the x_matrix whose parameters can or are actually
                            being estimated.
    
    ridging:                A float or the boolean "False" indicating whether
                            or not the ridge regression estimates of the
                            parameters should be calculated. If the ridge
                            estimates are to be calculated, then "ridging"
                            should be a positive float.
                            
    draws:                  numpy array of shape num_observations by num_draws.
                            Elements of this array should be individual draws
                            from the mixing distribution, for each observation.
    ====================
    """
    prob_draw_derivatives = calc_prob_draw_derivatives(drawn_probabilities,
                                                       x_matrix, 
                                                       pos_indicator,
                                                       neg_indicator,
                                                       draws)
                                                       
#    print "Minimum probability draw derivative is", prob_draw_derivatives.min()
                                                       
    score_matrix = 1.0/simulated_probs * prob_draw_derivatives
    if not ridging:
        overall_gradient = score_matrix.take(estimable_pos, axis=1).sum(axis=0)
        assert beta.shape[0] == overall_gradient.shape[0]
    else:
        # When performing ridge regression, all betas are estimable, so don't
        # bother to "extract" estimable betas. However, we must add the
        # derivative of the ridge regression penalty
        overall_gradient = score_matrix.sum(axis=0).ravel() +\
                           2 * ridging * beta
                           
    return overall_gradient
    
def calc_vector_prob(v):   
    """
    v:      numpy array of shape nObs by 1. Should contain the systematic
            systematic utility of class "1" being chosen, assuming that class
            "0" has a utility of 0. The systematic utility is beta * x.
    ==========
    Returns: numpy array of nObs by 1. The returned array contains the
             probability of each observation being associated with the outcome
             y = 1.
    """
    # Calculate the probabilities based on the systematic utilities
    probabilities = 1.0 / (1.0 + np.exp(-1.0 * v))
    
    # As a precaution when the probabilities are too small,
    # set them to a minimum value
    probabilities[probabilities < min_comp_value] = min_comp_value
    
    return probabilities
                                  
def calc_overall_loss(probabilities, beta,
                      pos_indicator, neg_indicator,
                      ridging, stratified_results = False):
    """
    probabilities:      numpy array of nObs by 1. The returned array contains
                        the probability of each observation being associated
                        with the outcome y = 1.
    
    beta:               A numpy array of shape (nPredictors,). It contains the
                        most recent guess about the value of the betas being
                        used to predict the probability that an observation is
                        of class 1.

    pos_indicator:      A numpy array of shape (nObs, 1). The array indicates
                        if a given observation is of class 1.
                    
    neg_indicator:      A numpy array of shape (nObs, 1). The array indicates
                        if a given observation is of class 0, i.e. the
                        negative class.
                    
    ridging:            A float or the boolean "False" indicating whether or
                        not the ridge regression estimates of the parameters
                        should be calculated. If the ridge estimates are to be
                        calculated, then "ridging" should be a positive float.
                    
    stratified_results: bool. Indicates whether or not simply the overall loss
                        for the dataset should be returned or whether a tuple
                        of the overall loss, as well as the loss on the
                        'positive' and 'negative' class should be returned.
    ====================
    Returns:            float or tuple. If stratified_results == False, then
                        a float is returned which denotes the overall loss
                        across the dataset. If stratified_results == True,
                        then a tuple of floats is returned where the first
                        element is the overall loss, the second element is the
                        loss on the total loss across members of the positive
                        class, and the third element is the loss on the
                        'negative' class.
    """
    assert np.all(probabilities > 0.0)
    assert np.all(probabilities < 1.0)
    # Calculate the negative log-likelihood for the entire dataset
    # Note that this is -1 * sum(log(probability of the CHOSEN alternative))
    losses = -1.0 * (np.log(probabilities) * pos_indicator +\
                     np.log(1 - probabilities) * neg_indicator)
    
    # Calculate the overall loss for the dataset
    overall_loss = losses.ravel().sum()
    
    assert not np.isnan(overall_loss)
    
    # If performing ridge regression, add the ridge penalty to the standard
    # loss function.
    if ridging is not False:
        ridge_loss = ridging * np.square(beta.ravel()).sum()
        overall_loss = overall_loss + ridge_loss
    
    # Use this condition to check whether or not stratified results are to be
    # returned. This feature is mainly to compare different models using the
    # particular loss function used to estimate this model.    
    if not stratified_results:
        return overall_loss
    else:
        pos_loss = losses[np.where(pos_indicator)].ravel().sum()
        neg_loss = losses[np.where(neg_indicator)].ravel().sum()
        return overall_loss, pos_loss, neg_loss
    
def calc_gradient(probabilities, x_matrix, beta,
                  pos_indicator, neg_indicator,
                  estimable_pos, ridging):
    """
    probabilities:      numpy array of nObs by 1. The returned array contains
                        the probability of each observation being associated
                        with the outcome y = 1.
        
    x_matrix:           A numpy array of shape (nObs, nPredictors). This array
                        contains all of the explanatory variables being used
                        to predict the probability that an observation is of
                        class 1.
    
    beta:               A numpy array of shape (nPredictors,). It contains the
                        most recent guess about the value of the betas being
                        used to predict the probability that an observation is
                        of class 1.

    pos_indicator:      A numpy array of shape (nObs, 1). The array indicates
                        if a given observation is of class 1.
                    
    neg_indicator:      A numpy array of shape (nObs, 1). The array indicates
                        if a given observation is of class 0, i.e. the
                        negative class.
    
    estimable_pos:      A list of positions which indicate the columns of the
                        x_matrix whose parameters can or are actually being
                        estimated.
                    
    ridging:            A float or the boolean "False" indicating whether or 
                        not the ridge regression estimates of the parameters 
                        should be calculated. If the ridge estimates are to be
                        calculated, then "ridging" should be a positive float.
    ====================
    Returns:            a numpy array of shape (len(estimable_pos), 1). The
                        elements of this array are the derivative of the
                        overall loss with respect to the corresponding
                        coefficients being estimated.
    """    
    # The gradient of proper losses is the predicted P(y=1) * x for
    # negative examples and (predicted P(y=1) - 1) * x for positive examples    
    gradient_pieces = (probabilities - 1) * pos_indicator * x_matrix +\
                      probabilities * neg_indicator * x_matrix
    # The overall gradient for the dataset is the sum of the gradients on each
    # example.
    if not ridging:
        overall_gradient = gradient_pieces.take(estimable_pos,
                                                axis=1).sum(axis=0)
    else:
        # When performing ridge regression, all betas are estimable, so don't
        # bother to "extract" estimable betas. However, we must add the
        # derivative of the ridge regression penalty
        overall_gradient = gradient_pieces.sum(axis=0).ravel() +\
                           2 * ridging * beta
    return overall_gradient
    
def calc_loss_and_gradient(beta, x_matrix, constraint_vec, estimable_pos,
                           pos_indicator, neg_indicator, ridging, draws):
    """
    beta:           A numpy array of shape (nPredictors,). It contains the most
                    recent guess about the value of the betas being used to
                    predict the probability that an observation is of class 1.
                    
    x_matrix:       A numpy array of shape (nObs, nPredictors). This array
                    contains all of the explanatory variables being used to
                    predict the probability that an observation is of class 1.

    constraint_vec: An array of shape (nPredictors,). It's used, if necessary,
                    to constrain certain parameters to specified values.
                    
    estimable_pos:  A list of positions which indicate the columns of the
                    x_matrix whose parameters can or are actually being
                    estimated.
    
    pos_indicator:  A numpy array of shape (nObs, 1). The array indicates if a
                    given observation is of class 1.
                    
    neg_indicator:  A numpy array of shape (nObs, 1). The array indicates if a
                    given observation is of class 0, i.e. the negative class.

    ridging:        A float or the boolean "False" indicating whether or not
                    the ridge regression estimates of the parameters should be
                    calculated. If the ridge estimates are to be calculated,
                    then "ridging" should be a positive float.
                    
    draws:          numpy array of shape num_observations by num_draws.
                    Elements of this array should be individual draws rom the
                    mixing distribution, for each observation.
    ===================
    Returns:        (scalar, a numpy array of shape (len(estimable_pos), 1))
    
                    The scalar is the overall uneven logistic loss that was
                    obtained using the betas passed into the function as an
                    argument.
    
                    The array is the gradient of the standard or simulated
                    logistic loss function (i.e. the negative log-likelihood).
    """
    # Create a vector the same length of the columns of x_matrix where the
    # Values of the vector take into account that not all of the values being
    # multiplied by the x_matrix are necessarily being estiamted.
    if draws is None:    
        constraint_vec[estimable_pos] = beta
    else:
        constraint_vec[estimable_pos[:-1]] = beta[:-1]

    # Calculate the systematic utilities for all observations    
    v = np.dot(x_matrix, constraint_vec).reshape((len(x_matrix), 1))
    
    if draws is None:
        # Calculate the probability of each observation choosing y = 1
        prob_of_y_equal_1 = calc_vector_prob(v)
        
        # Calculate the gradient for each beta being estimated
        gradient = calc_gradient(prob_of_y_equal_1, x_matrix, beta,
                                 pos_indicator, neg_indicator,
                                 estimable_pos, ridging)
    else:
        # Calculate the probabilities for each draw of the mixing distribution
        probability_draws = calc_drawn_probs(v, beta[-1], draws)
        
        # Calculate the simulated probability of each observation choosing y=1
        prob_of_y_equal_1 = calc_simulated_probs(probability_draws)
        
        # Calculate the gradient of the simulated negative log-likelihood
        gradient = calc_simulated_gradient(prob_of_y_equal_1,
                                           probability_draws,
                                           x_matrix, beta,
                                           pos_indicator, neg_indicator,
                                           estimable_pos, ridging, draws)
        
    # Calculate the loss overall
    overall_loss = calc_overall_loss(prob_of_y_equal_1, beta, 
                                     pos_indicator, neg_indicator,
                                     ridging)
    
#    print "Overall Loss: {:.4f}".format(overall_loss)
#    print "Current Gradient: {}".format(gradient.ravel()) 
#    print "="*20
    return overall_loss, gradient.ravel()
    
def estimate(data_path, explanatory_vars, shape_param, choice_col="choice",
             in_data=None, ridge=False, print_results=True,
             num_draws=None, mixing_dist=None,
             halton_draws=False, init_estimates=None):
    """
    data_path:          string. A path to a file location with a csv file of 
                        one's dataset. The csv file should have headers for
                        each column.
                       
    explanatory_vars:   A list of explanatory variables for use in the model.
                        The list SHOULD include the string 'intercept', if one
                        wants to include an intercept in one's model. All
                        other list elements should be strings corresponding to
                        the column headings in one's csv file or dataframe.
    
    shape_param:        None. Does nothing in this file and is not used.
                        Accepted as an argument simply to allow compatibility
                        with other discrete choice model files which do accept
                        a shape parameter.
                       
    choice_col:         string. The column header in one's csv file or
                        dataframe which denotes the column of zeros and ones
                        that denote whether an observation is of the "negative"
                        or "positive" class, respectively.
                       
    in_data:            A pandas dataframe containing the dataset one wishes to
                        use for estimation.
                       
    ridge:              float or boolean (False). If one wishes to estimate a
                        penalized cloglog model using ridge regression, pass a
                        positive float as the ridge argument. If one wishes to
                        estimate an un-penalized model, pass the boolean False.
                       
    print_results:      boolean. Determines whether or not the results, in
                        terms of estimation time or log-likelihood will be
                        printed to the screen.
                        
    num_draws:          None or int. If None, then a standard, i.e. non-mixed
                        logit is estimated. If not None, a mixed logit model
                        is estimated with the given number of draws.
                        
    mixing_dist:        scipy.stats distribution. Draws are taken from this
                        distribution for the mixed logit model.
                        
    halton_draws:       bool. If True, the given number of draws are taken
                        using a Halton sequence. This should permit a much
                        smaller number of draws to be used per given level of
                        desired accuracy from the mixed logit model.
                        
    init_estimates:     numpy array of shape (len(explanatory_variables),).
                        Optional starting parameters for the mixed logit model.
    ==========================================                   
    Returns:           tuple. First item is scipy.minimize results. Second
                       item is the flattened array of predicted probabilities
                       of being in the "positive" class. The third, fourth,
                       and fifth items are the log-likelihood overall, on the
                       positive, and on the negative class, respectively.
    """    
    #Input parameters
    loss_tol = 1e-06
    gradient_tol = 1e-06
    
    # Read in the data
    if in_data is None:
        print "Reading " + data_path
        data = pd.read_csv(data_path)
    else:
        if isinstance(in_data, pd.DataFrame):
            data = in_data.copy()
        else:
            data = in_data
 
    # Make sure the data has an intercept column
    data['intercept'] = 1.0
   
    # Make sure the choice column is in the data
    assert choice_col in data.columns.tolist()    
    
    # Create a numpy array of the data that can be operated on by scipy.optimize
    x_matrix = np.array(data[explanatory_vars])
    
    # Create arrays indicating the class of each observation
    pos_indicator = (data[choice_col] == 1.0).astype(int).values.reshape((len(data), 1))
    neg_indicator = (data[choice_col] == 0.0).astype(int).values.reshape((len(data), 1))
    
    
    # Create a constraint vector that will be used to constrain values in case
    # of perfect separation of the data due to dummy variables.
    num_explanatory_vars = len(explanatory_vars)
    # Note the scale is automatically negative three. This should probably be
    # changed to make the value scale with whatever value is needed to give a
    # "strong" negative effect. The same is true below.
    constraint_vec = (-3.0) * np.ones(num_explanatory_vars) 
    
    # Check for only two values in the column (since dummy variables are 0/1)
    if num_draws is None:    
        estimable_positions = range(num_explanatory_vars)
    else:
        estimable_positions = range(num_explanatory_vars + 1)
    dummy_values = set([0.0, 1.0])
    
    if ridge is False:       
        for pos, col in enumerate(explanatory_vars):
            possible_values = data[col].unique()
            if not dummy_values.difference(possible_values):
                # Check for perfect separation
                values_at_one = data[data[col]==1][choice_col].unique()
                #Note the line below assumes only two possible classes
                if len(values_at_one) != 2: 
                    # Make sure this variable is considered un-estimable
                    estimable_positions.remove(pos)
                    # If having a one for this dummy means everyone is in class
                    # one, then change the value of the parameter to be fixed to
                    # positive (3.0) instead of being negative, -3.0
                    if 0.0 not in values_at_one and 1.0 in values_at_one:
                        constraint_vec[pos] = 3.0
    
    # Take initial guesses for the value of the betas
    betas = np.zeros(len(estimable_positions))
    
    if init_estimates is not None:
        assert init_estimates.shape[0] == num_explanatory_vars
        assert num_explanatory_vars == len(estimable_positions) - 1
        betas[:-1] += init_estimates
        betas[-1] += 1
    
    # Take the draws for the mixed logit if necessary
    if num_draws is not None:
        # Make sure a scipy.stats frozen random variable was passed to the
        # function        
        try:
            assert isinstance(mixing_dist,
                              scipy.stats._distn_infrastructure.rv_frozen)
        except AssertionError:
            print "The mixing_dist must be a scipy.stats frozen random variable"
            print "A {} was passed instead.".format(type(mixing_dist))
       
        draws = create_draws(mixing_dist, num_draws,
                             x_matrix.shape[0], halton_draws)
                             
        # Initialize the mixing distribution parameter to be 1.0
        betas[-1] = 1.0
                             
    else:
        draws = None
    
    # Start timing the estimation process
    start_time = time.time()    
    
    # Find the betas that minimize the loss function
    results = minimize(calc_loss_and_gradient, betas,
                   args = (x_matrix, constraint_vec, estimable_positions,
                           pos_indicator, neg_indicator, ridge, draws), 
                   method = 'BFGS', jac = True, tol = loss_tol,
                   options = {'gtol': gradient_tol})
   
    # Stop timing the estimation process and report the timing results
    end_time = time.time()
    if print_results:
        elapsed_time = (end_time - start_time)/60.0
        print "Estimation Time: {:.2f} minutes".format(elapsed_time)
                
    # Calculate final probability for regular logit
    if num_draws is None:    
        constraint_vec[estimable_positions] = results.x
        final_v = np.dot(x_matrix, constraint_vec)[:, np.newaxis]
        final_p_class1 = calc_vector_prob(final_v)
    else:
        # Calculate final probability for mixed logit,
        # starting with systematic utilities        
        constraint_vec[estimable_positions[:-1]] = results.x[:-1]
        final_v = np.dot(x_matrix, constraint_vec)[:, np.newaxis]
        
        # Calculate the probabilities for each draw of the mixing distribution
        probability_draws = calc_drawn_probs(final_v, results.x[-1], draws)
        
        # Calculate the simulated probability of each observation choosing y=1
        final_p_class1 =  calc_simulated_probs(probability_draws)
        
    # Calculate the final probability of choosing y= 0.
    final_p_class0 = 1 - final_p_class1
    
    # Calculate the final log-likelihood overall and on each class    
    final_pos_ll = np.log(np.compress(pos_indicator.ravel(), final_p_class1)).sum()
    final_neg_ll = np.log(np.compress(neg_indicator.ravel(), final_p_class0)).sum()
    final_log_likelihood = final_pos_ll + final_neg_ll
    if print_results:    
        print "Log-likelihood of final model is:", final_log_likelihood
    # Return the estimation results and estimated p(y=1|x)
    return results, final_p_class1.ravel(), final_log_likelihood, final_pos_ll, final_neg_ll
    
# Create a model object which I can save, use for prediction, etc.
class DCModel(object):
    def __init__(self, data_path, data=None,
                 shape_param = None,
                 choice_col = "choice",
                 print_results=True,
                 ridge=False,
                 num_draws=None,
                 mixing_dist=None,
                 halton_draws=False,
                 init_estimates=None):
        """
        data_path:      string. A path to a file location containing a csv file
                        of one's dataset. The csv file should have headers for
                        each column.
                        
        data:           A pandas dataframe containing the dataset to be used
                        for estimation.
                        
        shape_param:    None or tuple. If tuple, should contain two elements,
                        each of which is an int, float, or long. They should
                        be the first and second parameters, respectively, of
                        Stukel's generalized logistic model. They should both
                        be within the range (-1, 1), exclusive. They control
                        the shape of the inverse link function in the region
                        of positive (or negative) "v", i.e. positive or 
                        negative differences in systematic utilities,
                        respectively. 
                        
        choice_col:     string. The column header in one's csv file or
                        dataframe which denotes the column of zeros and ones
                        that denote whether an observation is of the "negative"
                        or "positive" class, respectively.

        print_results:  boolean. Determines whether or not the results, in
                        terms of estimation time or log-likelihood will be
                        printed to the screen.
    
        ridge:          float or boolean (False). If one wishes to estimate a
                        penalized cloglog model using ridge regression, pass a
                        positive float as the ridge argument. If one wishes to
                        estimate an un-penalized model, pass the boolean False.
        ====================
        Returns:        None. Initializing the model class will populate the
                        following attributes: 'data' (which stores the data
                        that will be used for model estimation), 'shape_param',
                        'ridge_param', 'choice_col', 'print_results', and
                        'model_type'.
        """
        
        # Load the data for the model whether it is coming from a csv file
        # or whether it is being directly passed in as a dataframe
        if isinstance(data_path, str):
            # If a filepath to the data is passed, make sure it ends in .csv            
            try:
                assert data_path[-4:] == ".csv"
            except AssertionError as e:
                msg = "The passed filepath: {} is not csv file."
                print msg.format(data_path)
                raise e

            print "Reading " + data_path
            self.data = pd.read_csv(data_path)
        
        elif isinstance(data, pd.DataFrame):
            self.data = data

        else:
            print "Neither a valid filepath nor a dataframe was passed."

        # Save the shape parameter, ridge_parameter, choice column, model type,
        # and whether or not we want the results of the model estimation to
        # be printed.
        self.shape_param = shape_param        
        self.print_results = print_results
        self.ridge_param = ridge
        self.choice_col = choice_col
        self.model_type = "Binary Logit"
        self.num_draws = num_draws
        self.mixing_dist = mixing_dist
        self.halton_draws = halton_draws
        self.init_estimates = init_estimates
    
    # Create a function which will be exponsed to the user for model estimation
    def fit(self, explanatory_variables, shape_param):
        """
        explanatory_variables:  A list of explanatory variables to be used in
                                the model. The list SHOULD include the string
                                'intercept', if one wants an intercept to be
                                included in one's model. All other list
                                elements should be strings corresponding to
                                the column headings in one's csv file or
                                dataframe.
        
        shape_param:            None or tuple. If tuple, should contain two
                                elements, each of which is an int, float, or
                                long. They should be the first and second
                                parameters, respectively, of Stukel's
                                generalized logistic model. They should both
                                be within the range (-1, 1), exclusive. They
                                control the shape of the inverse link function
                                in the region of positive (or negative) "v", 
                                i.e. positive or negative differences in 
                                systematic utilities, respectively.     
        ====================
        Returns:                None. After this function is executed, the
                                following attributes will be populated: 
                                'coefs', 'fitted_probs', 
                                'in_sample_log_likelihoods',
                                'explanatory_vars', and 'shape_param'.
        """
        if shape_param is not None:
            self.shape_param = shape_param
        
        try:
            assert self.shape_param is not None
        except AssertionError as e:
            print "The model cannot be fit because there is no shape parameter"
            raise e
        
        estimation_results = estimate(None,
                                      explanatory_variables,
                                      self.shape_param,
                                      choice_col = self.choice_col,
                                      in_data = self.data,
                                      ridge = self.ridge_param,
                                      print_results=self.print_results,
                                      num_draws=self.num_draws,
                                      mixing_dist=self.mixing_dist,
                                      halton_draws=self.halton_draws,
                                      init_estimates=self.init_estimates)
                                      
        self.estimation_results = estimation_results[0]

        if self.num_draws is None:        
            self.coefs = pd.Series(dict(zip(explanatory_variables,
                                             estimation_results[0].x)),
                                    name="estimated_coefs",
                                    index = explanatory_variables)
        else:
            self.coefs = pd.Series(dict(zip(explanatory_variables + ['sigma'],
                                             estimation_results[0].x)),
                                    name="estimated_coefs",
                                    index = explanatory_variables + ['sigma'])
                                
        self.fitted_probs = estimation_results[1]
        
        self.in_sample_log_likelihoods =\
             pd.Series({"log_likelihood_overall": estimation_results[2],
                        "log_likelihood_pos": estimation_results[3],
                        "log_likelihood_neg": estimation_results[4]},
                        index = ['log_likelihood_overall',
                                 'log_likelihood_pos',
                                 'log_likelihood_neg'],
                        name = "Log-Likelihood Results")
                        
        self.explanatory_vars = explanatory_variables
        
        self.estimation_success = estimation_results[0]["success"]
        
    def predict_probabilities(self, data_path, data=None):
        """
        data_path:      string. A path to a file location containing a csv file
                        of one's dataset. The csv file should have headers for
                        each column.
                        
        data:           A pandas dataframe containing the dataset to be used
                        for prediction. Any transformations applied to the
                        dataset used for prediction should have already been
                        performed on the dataset being passed to this function.      
        ====================
        Returns:        numpy array of shape (nObs,). The elements of the
                        array are the predictions that an observation will be
                        of the "positive" class, whatever that was defined to
                        be in the dataset originally used to estimate the
                        model.
        """
        # Read in the data for which we want predictions based on the filepath        
        if isinstance(data_path, str):
            try:
                assert data_path[-4:] == ".csv"
            except AssertionError as e:
                msg = "The passed filepath: {} is not csv file."
                print msg.format(data_path)
                raise e
            print "Reading " + data_path
            data = pd.read_csv(data_path)
        else:
            # If we've passed a dataframe, then ensure it's valid
            try:            
                assert isinstance(data, pd.DataFrame)
            except AssertionError as e:
                print "The object passed as 'data' is not a Pandas dataframe."
                raise e
        
        # Make sure there is an intercept column
        if "intercept" in self.coefs.index and "intercept" not in data.columns:
            data["intercept"] = 1.0
        
        # Make sure that the variables used in the original model are all
        # present in the new dataframe on which we want to make predictions
        try:
            variables_in_columns = [x in data.columns for x
                                    in self.explanatory_vars]        
            assert all(variables_in_columns)       
        except AssertionError as e:
            missing_vars = [x for pos, x in enumerate(self.explanatory_vars)
                            if not variables_in_columns[pos]]
            msg_1 = "The following explanatory variables for this model are"
            msg_2 = " missing from the passed dataframe:"
            print msg_1 + msg_2
            print missing_vars
            raise e
        
        # Calculate the systematic utilities for each of the individuals
        # Assume that all individuals have all alternatives available
        x_matrix = np.array(data[self.coefs.index])        
        sys_utilities = np.dot(x_matrix, self.coefs.values[:, np.newaxis])
        # Calculate the probabilities given the systematic utilities
        # Note that the returned predictions are a numpy of (nObs by 1).
        predictions = calc_vector_prob(sys_utilities, self.shape_param)
        return predictions.ravel()
        
    def to_pickle(self, filepath):
        """
        filepath:   str. Should end in .pkl. If it does not, ".pkl" will be
                    appended to the passed string.
        ====================
        Returns:    None. Saves the model object to the passed in filepath.
        """
        assert isinstance(filepath, str)        
        if filepath[-4:] != ".pkl":
            filepath = filepath + ".pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print "Model saved to {}".format(filepath)
        
    def calc_canonical_loss(self, probs=None,
                            pos_indicator=None, neg_indicator = None):
        if probs is None:
            probs = self.fitted_probs
            pos_indicator = self.data[self.choice_col].values
            neg_indicator = 1 - pos_indicator
        else:
            assert pos_indicator is not None and neg_indicator is not None
            assert pos_indicator.shape == probs.shape
            assert neg_indicator.shape == probs.shape
        overall_loss, pos_loss, neg_loss = calc_overall_loss(probs,
                                                     self.shape_param,
                                                     self.coefs.values,
                                                     pos_indicator,
                                                     neg_indicator,
                                                     self.ridge_param,
                                                     stratified_results=True)
                                                                  
        return pd.Series({"overall_loss": overall_loss,
                          "positive_loss": pos_loss,
                          "negative_loss": neg_loss},
                          index=["overall_loss",
                                 "positive_loss",
                                 "negative_loss"])
