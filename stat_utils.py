import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy


def make_exp_z_scores(resids):
    df = pd.DataFrame(resids, columns = ['residual']) #Create a "spreadsheet" with one column--that of the residuals.
    df.sort(columns = 'residual', inplace = True)  #Sort the residuals from smallest to largest.
    df['rank'] = range(1, len(resids.index)+1)        #Add a column to the "spreadsheet which goes from 1 to the final number of observations
    df['corrCDF'] = (df['rank'] - 0.5) / len(df['rank']) #Use a continuity correction to calculate the cdf observed in the column of residuals
    df['expZscore'] = scipy.stats.norm.ppf(df['corrCDF']) #Using the inverse cdf, find the z-score which would be generate the observed CDF. Create a column for these values
    return df['expZscore']

def makeNormProbPlot(ser):
    """ser = a series or numpy array or list of residuals from a regression model"""
    df = pd.DataFrame(ser, columns = ['residual']) #Create a "spreadsheet" with one column--that of the residuals.
    df.sort(columns = 'residual', inplace = True)  #Sort the residuals from smallest to largest.
    df['rank'] = range(1, len(ser.index)+1)        #Add a column to the "spreadsheet which goes from 1 to the final number of observations
    df['corrCDF'] = (df['rank'] - 0.5) / len(df['rank']) #Use a continuity correction to calculate the cdf observed in the column of residuals
    df['expZscore'] = scipy.stats.norm.ppf(df['corrCDF']) #Using the inverse cdf, find the z-score which would be generate the observed CDF. Create a column for these values
    plt.scatter(df['residual'], df['expZscore'])    #Create a scatter plot with the residuals on the x-axis and the expected z-scores on the y-axis.
    plt.xlabel('Residuals')    #Label the x-axis
    plt.ylabel('Expected Normal Scores')    #Label the y-axis
    h = pd.Series([df['residual'].min(), df['residual'].max()])    #Choose two points to anchor the regression line of the residuals vs. the expected z-score.
    res = smf.ols('expZscore ~ residual', data = df).fit()    #Estimate the regression line of the residuals vs. the expected z-score
    intercept, slope = res.params    #store the coefficients of the expected z-score vs. residuals regression as variables 'intercept' and 'slope'
    j = intercept + slope * h    #Create the regression line of expected z-score vs. residuals.
    plt.plot(h, j, 'r-')    #Plot the regression line of expected z-score vs. residuals
    plt.suptitle('Normal Probability Plot: r-squared = {}'.format(round(res.rsquared,3)))    #Add a title to the normal probability plot.
    plt.show()    #Display the plot.
    return
       
def makeResidPlot(xVar, residuals, x_label, title='Residuals vs. ', y_label='Residuals'):
    """xVar = a series or numpy array of the variable that you want the residuals to be plotted against
    residuals = a series or numpy array of the residuals"""
    plt.scatter(xVar, residuals)    #Create a scatter plot of the residuals versus the variable that is supposed to be on the x-axis
    small = xVar.min()-0.1 * xVar.min()    #Calculate a value for the lowest visible number on the x-axis.
    big = xVar.max()+ 0.1 * xVar.max()     #Calculate a value for the highest visisble number on the x-axis.
    plt.xlim(small, big)                   #Create the x-axis with the limits specified by the numbers above.
    plt.xlabel(x_label)                    #Create the x-label for the scatter plot.
    plt.ylabel(y_label)                    #Create the y-label for the scatter plot.
    plt.title(title+x_label)               #Create the title for the scatter plot.
    h = pd.Series([small, big])            #Create limits points for the line of x = 0
    y = 0 * h                              #Create the line x = 0
    plt.plot(h, y, 'r-')                   #Plot the line x = 0
    plt.show()                             #Show the plot
    return
    
def all_residual_plots(model, data):
    residuals = model.resid
    makeNormProbPlot(residuals)
    variables = model.params.index.tolist()
    for var in variables[:]:
        if "Intercept" in var or "I(" in var:
            variables.remove(var)
    makeResidPlot(model.fittedvalues, residuals, "y-hat")
    for var in variables:
        makeResidPlot(data[var], residuals, var)
    
    
def boxCox(dep, formula, raw_data):
    """dep = an n by 1 array of the values on the left-side of a regression equation.
       formula should be statsmodel formula string, starting from the first variable name after '~'.
       raw_data should be a pandas dataframe
       Returns a tuple, of the estimated lambda value and the dataframe of lambdas and their r-squared values."""
    dataset = raw_data.copy()
    best = 1    #Initialize a "best" value that will be returned later on. It will be updated with lambdas as I iterate over possible lamdas
    best_val = 0    #Initialize a "best" r-squared value that will be returned later on. It will be updated with lambdas as I iterate over possible lamdas
    values = []    #Initialize an empty list to store the lambda values that have been tried.
    r2_values = []    #Initialize an empty list to store the r-squared values associated with the lambda values above.
    gMean = scipy.stats.gmean(dep)    #calculate the geometric mean of the dependent variable for use later on.
    for guess in xrange(-30,31,1):   #since range only moves in steps of one, and only goes between integers, use -20 to 21 to represent -2 to 2.
        lam = guess * 0.1 #Use this step to scale the values from xrange down to single digits -2 to 2 and with a step value of 0.1 instead of 1
        if lam == 0 or lam == 0.0:    #Use this transformation when lambda equals zero.
            v = gMean * np.log(dep)
        else:
            v = ((dep ** lam) - 1)/(lam * (gMean ** (lam - 1)))   #Use this transformation when lambda does not equal 1.
        dataset["trial_v"] = v
        res = smf.ols('trial_v ~ ' + formula, data = dataset).fit()    #Fit the transformed data to the given right hand side of the equation.
        norm_correlation, norm_p_value = scipy.stats.pearsonr(np.sort(res.resid), make_exp_z_scores(res.resid))
        if norm_correlation > best_val:    #check if the normal probability correlation value is better than the best value so far
            best_val = norm_correlation    #If the current normal probability correlation value is better than the best value so far, change the best value to the curent normal probability correlation
            best = lam     #If the current r-squared value is better than the best r-squared value so far, change the best lambda value to the curent lambda value
        values.append(lam)  #Append the current lambda value to the end of the 'values' list
        r2_values.append(norm_correlation **2)    #Append the current lambda value to the end of the 'lambdas list'
    df = pd.DataFrame({'exponent':values, 'r2_values':r2_values}) #Create a "spreadsheet" of the lambda values and their associated r-squared values.
    df.sort(columns = 'r2_values', inplace = True, ascending = False)    #Sort the "spreadsheet" in order of decreasing r-squared values.
    return best, df
