# This is the assignment 2 for Computational Math
# In this assignment include 3 question

#%% import packages
import numpy as np
from scipy.stats import t
from scipy.stats import norm
import matplotlib as plt
#%%
# Monte-Carlo Simulation
def confidence_interval(x, confidence):
    """
    Function to estimate the confidence interval
    Arguments: 
        Input: 
            x : numpy array,
                input data
            confidence : demical
                confidence interval level of the dataset usually set up as 0.95
        Output:
            left_confidence_interval : the left value of the confidence interval
            right_confidence_interval : the right value of the confidence interval
    """
    # confidence interval calculation
    """
    Methodology:
    confidence interval

    (m - t * s / sqrt(N), m + t * s / sqrt(N))
    m : mean value
    s : standard deviation of sample
    N : sample size
    t : t-value that correspond to the confidence interval

    Reference website:
    https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    """
    # obtain the mean value 
    m = np.mean(x)
    # obtain the standard deviation
    std = np.std(x) # standard deviation for f(x)
    # calculate the degree of freedom (DOF)
    dof = len(x) - 1
    confidence = confidence
    # calculate the t value 
    t_crit = np.abs(t.ppf((1 - confidence) /2, dof)) # ppf calculates the inverse cumulative distribution function.
    # confidence interval
    # left 
    left_confidence_interval = m - std * t_crit / np.sqrt(len(x))
    # right
    right_confidence_interval = m + std * t_crit / np.sqrt(len(x))
    return left_confidence_interval, right_confidence_interval


def monte_carlo(miu, sigma2, n):
    """
    Using Monte-Carlo simulation method to approximating the E[f(x)]

    Input:
        miu: mean or expectation of the distribution
        sigma2: variance of distribution
        n: the number of sample paths

    Output:
        Sample_mean: the value that comes out of the monte carlo simulation
        Sample_variance: the variance that comes out of the monte carlo simulation
        Confidence_interval: the value of 95% confidence interval
    
    """

    # methodology explain
    """
    E[sin(x) * e^x] = 1/n * sum(sin(x_i) * e^x_i)
    Here x_i are samples of X ~ N(miu,sigma2)
    X = miu + signma * error
    error = N(0,1)
    """

    # generate the x_i
    """
    Note: here should generate N x 1 shape numpy array
    """
    x_i = np.sqrt(sigma2) * np.random.randn(n, 1) + miu # N ~ (0,1)
    # define the f(x)
    f_x = np.sin(x_i) * np.exp(x_i)
    # use monte-carlo simulation to estimate
    sample_mean = 1 / n * sum(f_x)
    # calculate the variance
    # reference website
    # https://www.calculatorsoup.com/calculators/statistics/variance-calculator.php
    sample_variance = sum((f_x - sample_mean) ** 2) / (n - 1)
    # calculate the confidence interval at 0.95
    left_confidence_interval, right_confidence_interval = confidence_interval(x= f_x, confidence= 0.95)
    print("the number of sample path is:", n)
    print("sample mean is:", sample_mean)
    print("sample variance is:", sample_variance)
    print("confidence interval at 0.95 is:", (left_confidence_interval, right_confidence_interval))
    return sample_mean, sample_variance, left_confidence_interval, right_confidence_interval
#%%
mean, variance, left_confidence_interval, right_confidence_interval = monte_carlo(miu = 5, sigma2 = 1, n = 500000)

# %%
# Black Scholes formula
# define the black scholes call function
def bs_call(S, K, T, r, sigma):
    """
    Here is the function that using the Black Scholes formula to price the European call

    Arguments:
    Input :
        S : current asset price 
        K : strike price of the option
        r : risk free interest rate 
        T : time until option expiration
        sigma : annualized volatility of the asset's return

    Output : 
        Paying stock
    """

    # methodology 
    """
    Methodology :
    Call option : 
        Call = S0 * N(d1) - N(d2) * K * e^ (-rT)
    
    Reference website : 
    https://www.codearmo.com/python-tutorial/options-trading-black-scholes-model

    """

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    call_option = S * norm.cdf(d1) - K * np.exp(- r * T) * norm.cdf(d2)

    return call_option



def bs_put(S, K, T, r, sigma):
    """
    Here is the function that using the Black Scholes formula to price the European put

    Arguments:
    Input :
        S : current asset price 
        K : strike price of the option
        r : risk free interest rate 
        T : time until option expiration
        sigma : annualized volatility of the asset's return

    Output : 
        Paying stock
    """

    # methodology 
    """
    Methodology :
    Put option : 
        Put = N(-d2) * K * exp(-r * T) - N(-d1) * S0
    
    Reference website : 
    https://www.codearmo.com/python-tutorial/options-trading-black-scholes-model

    """

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    put_option = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S

    return put_option
# %% Question 2
# Use the Black-Scholes formula to price European all and put options
maturity = 1 # T
interest_rate = 0.07 # r
volatility = 0.35 # sigma
strike_price = 100 # K
stock_price = np.array([80,85,90,95,98,100,102,105,110,115,120])  # S 





call_price = bs_call(S= 80,
                    K= 100,
                    T= 1,
                    r= 0.07,
                    sigma= 0.35)
put_price = bs_put(S= 80,
                    K= 100,
                    T= 1,
                    r= 0.07,
                    sigma= 0.35)                    
# %%
