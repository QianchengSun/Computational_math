# this is the code implement for Assignment 3 for computational math

#%% import package
from re import S
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
#%%
# Question 1 Pricing European call and put options by explicit finite difference method
# define the function
def EuPutExpl(S_0, K, r, T, sigma, S_max, dS, dt):
    # here using explicit finite difference method to solve the European call

    # set up grid and adjust increments if necessary
    M = np.round(S_max / dS)
    M = M.astype(int)
    dS = S_max / M
    N = np.round(T / dt)
    N = N.astype(int)
    dt = T / N

    matval = np.zeros(shape = (M + 1, N + 1))

    vetS = np.transpose(np.linspace(0, S_max, M + 1))
    veti = np.linspace(0, M, M + 1)
    vetj = np.linspace(0, N, N + 1)
    # set up boundary conditons
    # Note: 
    # here the matval shape is (51,101)
    # because python start to count with 0 
    matval[:, N] = np.maximum(K - vetS, 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = K * np.exp(-r * dt * (N - vetj))
    matval[M, :] = 0

    # set up coefficients
    a = 0.5 * dt * (sigma ** 2 * veti - r) * veti
    b = 1 - dt * (sigma ** 2 * veti ** 2 + r) 
    c = 0.5 * dt * (sigma ** 2 * veti + r) * veti 

    # solve the packward in time

    for j in reversed(range(N)):
        for i in range(1 , M):
            matval[i,j] = a[i] * matval[i - 1, j + 1] + b[i] * matval[i, j + 1] + c[i] * matval[i + 1, j + 1]

    # return price, possibly by linear interpolation outside the grid
    f_x = interp1d(vetS, matval[:,0])
    price = f_x(S_0)
    return price

#%%
p = EuPutExpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 2,
            dt= 5/1200)
print(p)

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

# %%
put_price = bs_put(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print(put_price)
# %%
