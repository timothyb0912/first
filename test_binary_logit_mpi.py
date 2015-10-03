import os
import time
import json
import mpi4py
import numpy as np
import pandas as pd
import pymultinest as pmn             # Used to access the MultiNest software


# Specify the path to the output folder
relative_output_folder = "mpi_binary_logit_for_comparison/"

# Try to make the output folder
try:
    os.makedirs(relative_output_folder)
except OSError:
    # but if it exists, then just keep going
    # If the folder does not exist and we cannot make it, raise an error
    if not os.path.isdir(relative_output_folder):
        raise

# Define boundary values for the calculations
max_comp_value = 1e300
min_comp_value = 1e-300

# If we are going to use a uniform distribution, assign the upper and lower
# limits of the distribution
upper_uniform_limit = 20
lower_uniform_limit = -20

# Read in the necessary data for the model
path_to_pkl_of_data = "expanded_hybrid_specification_df.pkl"
raw_data = pd.read_pickle(path_to_pkl_of_data)

# Specify a choice column
choice_col = "bike_choice"

# Make sure the choice column is in the data
assert choice_col in raw_data.columns.tolist()

# Make sure the data has an intercept column
raw_data['intercept'] = 1.0

# Specify the explanatory variables, i.e. the number of dimensions to estimate
explanatory_vars = ['intercept',
                    'bike_time',
                    'num_bikes',
                    'num_cars',
                    'age',
                    'gender',
                    'education_status_2',
                    'education_status_3',
                    'education_status_4',
                    'education_status_5',
                    'education_status_6',
                    'standardized_income']

# Make sure all the explanatory variables are in the dataframe                    
assert all([x in raw_data.columns.tolist() for x in explanatory_vars])
                    
# Create a numpy array of the data for which we want estimated coefficients
model_data = np.array(raw_data[explanatory_vars])

# Create a numpy array containing the individual outcomes
choice_vector = raw_data[choice_col].values

# Record the number of parameters/dimensions being estimated in this model
num_dimensions = len(explanatory_vars)

# Define the number of live points to be used
num_live_points = 1000

# Define the sampling efficiency that is desired
desired_sampling_efficiency = 0.5

# Create function that converts the prior cube into the parameter cube
def uniform_prior(cube, n_dim, n_params):
    # Note that we are using a uniform distribution over all parameters    
    for i in range(n_dim):
        cube[i] = ((upper_uniform_limit - lower_uniform_limit) * cube[i]) +\
                  lower_uniform_limit
        
    return None
        

# Create a function that estimates the log likelihood
def binary_logit_log_likelihood(cube, n_dim, n_params):
    # Create an array of the coefficients
    betas = np.array([float(cube[i]) for i in range(n_dim)])
    
    # Calculate the systematic utility of choice y = 1
    # Resulting array should be of shape (n_obs,)
    sys_utility = model_data.dot(betas)
    
    # Calculate the total log-likelihood
    log_likelihood = (choice_vector * sys_utility -
                      np.log(1.0 + np.exp(sys_utility))).sum()
                      
    return log_likelihood
    

# Create a main function so that the script can be run
# from the command line as well as from within other
# scripts or from an ipython notebook
def main():
    # Begin timing the estimation process
    start_time = time.time()
    
    # Run the MultiNest software
    pmn.run(binary_logit_log_likelihood, uniform_prior, num_dimensions,
            outputfiles_basename=relative_output_folder,
            n_live_points=num_live_points,
            sampling_efficiency=desired_sampling_efficiency,
            log_zero=-1e200,
            mode_tolerance=-1e180,
            null_log_evidence=-1e180,
            resume=False, verbose=True, init_MPI=False)
            
    # Record the ending time of the estimation process
    end_time = time.time()
    tot_minutes = (end_time - start_time) / 60.0
            
    # Save the parameter names
    with open(relative_output_folder + "parameter_names.json", 'wb') as f:
        json.dump(explanatory_vars, f)
        
    # Save the number of live points used as the total estimation time
    model_run_params = {"n_live_points": num_live_points,
                        "sampling_efficiency": desired_sampling_efficiency,
                        "estimation_minutes": tot_minutes}
    with open(relative_output_folder + "model_run_parameters.json", "wb") as f:
        json.dump(model_run_params, f)
        
    # Print a report on how long the estimation process took
    print "Estimation process took {:.2f} minutes".format(tot_minutes)

if __name__ == "__main__":
    main()    

    
