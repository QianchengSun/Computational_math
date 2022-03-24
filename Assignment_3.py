# this is the code implement for Assignment 3 for computational math
"""
The Implicit method has problem, need to fix and debug
"""
#%% import package
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import scipy
#%% Define the function for Problem 1
# define the function for the 
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


# define the function for the European call explicit method
def EuCallExpl(S_0, K, r, T, sigma, S_max, dS, dt):
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
    matval[:, N] = np.maximum(vetS - K, 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = 0
    matval[M, :] = S_max - K * np.exp(-r * dt * (N - vetj))

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

#%% problem 1 
# problem 1 question (a) 
# here use the explicit FD method to solve european put option
p = EuPutExpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 2,
            dt= 5/1200)
print("the explicit FD method to solve European put option is: ", p)
put_price = bs_put(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print("The Black-Scholes formula to solve European put option is:", put_price)

#  Problem 1 question (b)
# here use the explicit FD method to solve the European call option
call_price = EuCallExpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 80,
            dS= 2,
            dt= 5/1200)
print("the explicit FD method to solve European call option is: ", call_price)
# here use the Black-Scholes formula to solve European call option
call_bs_price = bs_call(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print("The Black-Scholes formula to solve European call option is:",call_bs_price)
#%%
# Problem 1 Question c
S_0_list = [95,100,105]
K = 100
r = 0.07
sigma = 0.35
T = 1
S_max = 190
dS = 2
dt = 1/1200
put_price_list = []
put_bs_price_list = []
call_price_list = []
call_bs_price_list = []
for i in range(0, len(S_0_list)):
    # explicit put price
    p = EuPutExpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    put_price_list.append(p)
    # black shole
    put_price = bs_put(S= S_0_list[i],
                    K= K,
                    T= T,
                    r= r,
                    sigma= sigma)
    put_bs_price_list.append(put_price)
    # explicit call price
    call_price = EuCallExpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    call_price_list.append(call_price)
# here use the Black-Scholes formula to solve European call option
    call_bs_price = bs_call(S= S_0_list[i],
                    K= K,
                    T= T,
                    r= r,
                    sigma= sigma)
    call_bs_price_list.append(call_bs_price)
print("the explicit FD method to solve European put option is: ", put_price_list)
print("The Black-Scholes formula to solve European put option is:", put_bs_price_list)
print("the explicit FD method to solve European call option is: ", call_price_list)
print("The Black-Scholes formula to solve European call option is:",call_bs_price_list)
# %%
# Question 2 
# Fully implicit method 
# define the function 
def EuPutImpl(S_0, K, r, T, sigma, S_max, dS, dt):
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

    # set up the tridiagonal coefficients matrix
    a = 0.5 * (r * dt * veti - sigma ** 2 * dt * veti ** 2)
    b = 1 + sigma ** 2 * dt * (veti ** 2) + r * dt
    c = -0.5 * (r * dt * veti + sigma ** 2 * dt * veti ** 2)

    coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
    P,L,U = scipy.linalg.lu(coeff) 

    # solve the sequence of linear system
    aux = np.zeros(shape = (M-1, 1))
    for j in reversed(range(N)):
        aux[0] = -a[1] * matval[0, j]
        matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))
    f_x = interp1d(vetS, matval[:,0])
    price = f_x(S_0)
    return price

def EuCallImpl(S_0, K, r, T, sigma, S_max, dS, dt):
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
    matval[:, N] = np.maximum(vetS - K, 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = 0
    matval[M, :] = S_max - K * np.exp(-r * dt * (N - vetj))

    # set up the tridiagonal coefficients matrix
    a = 0.5 * (r * dt * veti - sigma ** 2 * dt * veti ** 2)
    b = 1 + sigma ** 2 * dt * (veti ** 2) + r * dt
    c = -0.5 * (r * dt * veti + sigma ** 2 * dt * veti ** 2)

    coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
    P,L,U = scipy.linalg.lu(coeff) 

    # solve the sequence of linear system
    aux = np.zeros(shape = (M-1, 1))
    for j in reversed(range(N)):
        aux[0] = -a[1] * matval[0, j]
        matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))
    f_x = interp1d(vetS, matval[:,0])
    price = f_x(S_0)
    return price
#%% 
# Problem 2 Question (a)
# here use the fully implicit method to solve the European put option
put_implicit_price = EuPutImpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 0.5,
            dt= 5/2400)
print("The implicit method to solve the European put option is:", put_implicit_price)
# here use the Black-Scholes formula to solve the European put option
put_bs_price = bs_put(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print("The Black-Scholes formula to solve the European put option is:", put_bs_price)
# Problem 2 Question (b)
# here use the fully implicit method to solve the European put option
call_implicit_price = EuCallImpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 0.5,
            dt= 5/2400)
print("The implicit method to solve the European call option is:", call_implicit_price)
# here use the Black-Scholes formula to solve the European put option
call_bs_price = bs_call(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print("The Black-Scholes formula to solve the European call option is:", call_bs_price)
#%%
# Problem 2 Question c
S_0_list = [95,100,105]
K = 100
r = 0.07
sigma = 0.35
T = 1
S_max = 200
dS = 2
dt = 1/2400
put_price_list = []
put_bs_price_list = []
call_price_list = []
call_bs_price_list = []
for i in range(0, len(S_0_list)):
    # explicit put price
    p = EuPutImpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    put_price_list.append(p)
    # black shole
    put_price = bs_put(S= S_0_list[i],
                    K= K,
                    T= T,
                    r= r,
                    sigma= sigma)
    put_bs_price_list.append(put_price)
    # explicit call price
    call_price = EuCallImpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    call_price_list.append(call_price)
# here use the Black-Scholes formula to solve European call option
    call_bs_price = bs_call(S= S_0_list[i],
                    K= K,
                    T= T,
                    r= r,
                    sigma= sigma)
    call_bs_price_list.append(call_bs_price)
print("the implicit FD method to solve European put option is: ", put_price_list)
print("The Black-Scholes formula to solve European put option is:", put_bs_price_list)
print("the implicit FD method to solve European call option is: ", call_price_list)
print("The Black-Scholes formula to solve European call option is:",call_bs_price_list)
# %% Problem 2 Debug section
#  Debug section
"""
Debug the implicit method put / call method for Problem 2
Result: 
"""
S_0= 50
K= 50 
r= 0.1
T= 5/12
sigma= 0.4
S_max = 100
dS= 0.5
dt= 5/2400
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

    # set up the tridiagonal coefficients matrix
a = 0.5 * (r * dt * veti - sigma ** 2 * dt *(veti ** 2 ))
b = 1 + sigma ** 2 * dt * (veti ** 2) + r * dt
c = -0.5 * (r * dt * veti + sigma ** 2 * dt *(veti ** 2 ))

coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
P,L,U = scipy.linalg.lu(coeff) 

    # solve the sequence of linear system
aux = np.zeros(shape = (M-1, 1))

for j in reversed(range(N)):
    aux[0] = - a[1] * matval[0, j]
    matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))

f_x = interp1d(vetS, matval[:,0])
price = f_x(S_0)
print(price)
# %% Pricing European options by Crank-Nicolson method
# define the function 
def DOPutCK(S_0, K, r,T, sigma, S_b, S_max, dS, dt):
    # set up grid and adjust increments if necessary 
    M = np.round((S_max - S_b) / dS)
    M = M.astype(int)
    dS = (S_max - S_b) / M
    N = np.round(T / dt)
    N = N.astype(int)
    dt = T / N

    matval = np.zeros(shape = (M + 1, N + 1))
    vetS = np.transpose(np.linspace(S_b, S_max, M + 1))
    veti = vetS / dS
    vetj = np.linspace(0, N, N + 1)
    # set up boundary conditons
    # Note: 
    # here the matval shape is (51,101)
    # because python start to count with 0 
    matval[:, N] = np.maximum(K- vetS, 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = K * np.exp(-r * dt * (N - vetj))
    matval[M, :] = 0
    # set up the coefficients matrix
    alpha = 0.25 * dt * (sigma ** 2 * veti ** 2 - r * veti)
    beta = - dt * 0.5 * (sigma ** 2 * veti ** 2 + r)
    gamma = 0.25 * dt * (sigma ** 2 * veti ** 2 + r * veti)

    M_1 = -np.diag(alpha[2:M], -1) + np.diag((1 - beta[1:M])) - np.diag(gamma[1:M-1], 1)
    P,L,U = scipy.linalg.lu(M_1)
    M_2 = np.diag(alpha[2:M], -1) + np.diag((1 + beta[1:M])) + np.diag(gamma[1:M-1], 1)
    # solve the sequence of linear system
    for j in reversed(range(N)):
        matval[1:M,j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L), np.matmul(M_2, matval[1:M, j+1])))
    f_x = interp1d(vetS, matval[:,0])
    price = f_x(S_0)
    return price

# %% Problem 3 Question (a)
crank_nicolson_put_price = DOPutCK(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_b = 40,
            S_max = 100,
            dS= 0.5,
            dt= 1/1200)
print("The put option by using Crank Nicolson method is:",crank_nicolson_put_price)
#%%
# Problem 3 Question (b)
crank_nicolson_put_price = DOPutCK(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_b = 0,
            S_max = 100,
            dS= 0.5,
            dt= 1/1200)
print("The put option by using Crank Nicolson method with the boundary condition S=0 is:",crank_nicolson_put_price)
p = EuPutExpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 2,
            dt= 5/1200)
print("the explicit FD method to solve European put option is: ", p)
# here use the fully implicit method to solve the European put option
put_implicit_price = EuPutImpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 100,
            dS= 0.5,
            dt= 5/2400)
print("The implicit method to solve the European put option is:", put_implicit_price)
# here use the Black-Scholes formula to solve the European put option
put_bs_price = bs_put(S= 50,
                K= 50,
                T= 5/12,
                r= 0.1,
                sigma= 0.4)
print("The Black-Scholes formula to solve the European put option is:", put_bs_price)
#%% Problem debug section
"""
Section for problem 3 debug
Result : bug has been fixed
"""
S_0= 50
K= 50 
r= 0.1
T= 5/12
sigma= 0.4
S_b = 40
S_max = 100
dS= 0.5
dt= 1/1200

M = np.round((S_max - S_b) / dS)
M = M.astype(int)
dS = (S_max - S_b) / M
N = np.round(T / dt)
N = N.astype(int)
dt = T / N

matval = np.zeros(shape = (M + 1, N + 1))
vetS = np.transpose(np.linspace(S_b, S_max, M + 1))
veti = vetS / dS
vetj = np.linspace(0, N, N + 1)
    # set up boundary conditons
    # Note: 
    # here the matval shape is (51,101)
    # because python start to count with 0 
matval[:, N] = np.maximum(K- vetS, 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
matval[0, :] = K * np.exp(-r * dt * (N - vetj))
matval[M, :] = 0
    # set up the coefficients matrix
alpha = 0.25 * dt * (sigma ** 2 * veti ** 2 - r * veti)
beta = - dt * 0.5 * (sigma ** 2 * veti ** 2 + r)
gamma = 0.25 * dt * (sigma ** 2 * veti ** 2 + r * veti)

M_1 = -np.diag(alpha[2:M], -1) + np.diag((1 - beta[1:M])) - np.diag(gamma[1:M-1], 1)
P,L,U = scipy.linalg.lu(M_1)
M_2 = np.diag(alpha[2:M], -1) + np.diag((1 + beta[1:M])) + np.diag(gamma[1:M-1], 1)
    # solve the sequence of linear system
for j in reversed(range(N)):
    vf_x = interp1d(vetS, matval[:,0])
price = f_x(S_0)
print(price)


# %% Problem 4 Function
# define function
def AmPutExpl(S_0, K, r, T, sigma, S_max, dS, dt):
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
            matval[i,j] = np.maximum(matval[i,j], K - i * dS)
    # return price, possibly by linear interpolation outside the grid
    f_x = interp1d(vetS, matval[:,0])
    price = f_x(S_0)
    return price
#%% Problem 4 question (a)
explicit_am_put = AmPutExpl(S_0= 50,
            K= 50, 
            r= 0.1,
            T= 5/12,
            sigma= 0.4, 
            S_max = 80,
            dS= 2,
            dt= 5/1200)
print("the explicit FD method to solve American put option is: ", explicit_am_put)
#%% Problem 4 
# Question (b)
S_0_list = [95,100,105]
K = 100
r = 0.07
sigma = 0.35
T = 1
S_max = 200
dS = 2
dt = 1/2400

put_price_list = []
put_bs_price_list = []
put_am_price_list = []
for i in range(0, len(S_0_list)):
    # explicit put price
    p = EuPutExpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    put_price_list.append(p)
    # black shole
    put_price = bs_put(S= S_0_list[i],
                    K= K,
                    T= T,
                    r= r,
                    sigma= sigma)
    put_bs_price_list.append(put_price)
    # explicit call price
    put_am_price = AmPutExpl(S_0= S_0_list[i],
                K= K, 
                r= r,
                T= T,
                sigma= sigma, 
                S_max = S_max,
                dS= dS,
                dt= dt)
    put_am_price_list.append(put_am_price)

print("the explicit FD method to solve European put option is: ", put_price_list)
print("The Black-Scholes formula to solve European put option is:", put_bs_price_list)
print("the explicit FD method to solve American put is: ", call_price_list)











#%% Debug section for Problem 4
S_0= 50
K= 50
r= 0.1
T= 5/12
sigma= 0.4
S_max = 100
dS= 2
dt= 5/1200

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
        matval[i,j] = np.maximum(matval[i,j], K - i * dS)
    # return price, possibly by linear interpolation outside the grid
f_x = interp1d(vetS, matval[:,0])
price = f_x(S_0)
print(price)
# %% Problem 5
# implicit finite difference method for PDE
def Implicit_put_q5(x, K, r, T, sigma, a, b, x_min, x_max, dx, dt):

    S_0 = np.exp(x)
    S_min = np.exp(x_min)
    S_max = np.exp(x_max)
    M = np.round((x_max - x_min) / dx)
    M = M.astype(int)
    dx = (x_max - x_min) / M
    N = np.round(T / dt)
    N = N.astype(int)
    dt = T / N

    matval = np.zeros(shape = (M + 1, N + 1))
    vetx = np.transpose(np.linspace(x_min, x_max, M + 1))
    veti = np.linspace(0, M, M + 1)
    vetj = np.linspace(0, N, N + 1)
        # set up boundary conditons
        # Note: 
        # here the matval shape is (51,101)
        # because python start to count with 0 
    matval[:, N] = np.maximum(K - np.exp(vetx), 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = K * np.exp(-r * dt * (N - vetj))
    matval[M, :] = 0

        # set up the tridiagonal coefficients matrix
    # a = 0.5 * (r * dt * veti - sigma ** 2 * dt *(veti ** 2 ))
    w = ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

    y = 1 + sigma ** 2 * dt + r * dt * veti

    z = - ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

    coeff = np.diag(w[2:M], -1) + np.diag(y[1:M]) + np.diag(z[1:M-1], 1)
    P,L,U = scipy.linalg.lu(coeff) 

        # solve the sequence of linear system
    aux = np.zeros(shape = (M-1, 1))

    for j in reversed(range(N)):
        aux[0] = - w[1] * matval[0, j]
        matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))

    f_x = interp1d(vetx, matval[:,0])
    price = f_x(S_0)
    
    return price

def Implicit_call_q5(x, K, r, T, sigma, a, b, x_min, x_max, dx, dt):

    S_0 = np.exp(x)
    S_min = np.exp(x_min)
    S_max = np.exp(x_max)
    M = np.round((x_max - x_min) / dx)
    M = M.astype(int)
    dx = (x_max - x_min) / M
    N = np.round(T / dt)
    N = N.astype(int)
    dt = T / N

    matval = np.zeros(shape = (M + 1, N + 1))
    vetx = np.transpose(np.linspace(x_min, x_max, M + 1))
    veti = np.linspace(0, M, M + 1)
    vetj = np.linspace(0, N, N + 1)
        # set up boundary conditons
        # Note: 
        # here the matval shape is (51,101)
        # because python start to count with 0 
    matval[:, N] = np.maximum(np.exp(vetx) - K , 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
    matval[0, :] = K * np.exp(-r * dt * (N - vetj))
    matval[M, :] = 0

        # set up the tridiagonal coefficients matrix
    # a = 0.5 * (r * dt * veti - sigma ** 2 * dt *(veti ** 2 ))
    w = ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

    y = 1 + sigma ** 2 * dt + r * dt * veti

    z = - ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

    coeff = np.diag(w[2:M], -1) + np.diag(y[1:M]) + np.diag(z[1:M-1], 1)
    P,L,U = scipy.linalg.lu(coeff) 

        # solve the sequence of linear system
    aux = np.zeros(shape = (M-1, 1))

    for j in reversed(range(N)):
        aux[0] = - w[1] * matval[0, j]
        matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))

    f_x = interp1d(vetx, matval[:,0])
    price = f_x(S_0)
    
    return price
#%% Problem 5 & 6
x = np.log(100)
K= 100
r= 0.07
T= 1
sigma= 0.35
a = 1
b = 0.5
x_min = 1
x_max = 1000
dx= 2
dt= 5/2400

put_price = Implicit_put_q5(x= x, K= K, r= r, T=T, sigma= sigma, a= a, b= b, x_min= x_min, x_max= x_max, dx= dx, dt= dt)
print("The put price by using implicit method is:", put_price)
call_price = Implicit_call_q5(x= x, K= K, r= r, T=T, sigma= sigma, a= a, b= b, x_min= x_min, x_max= x_max, dx= dx, dt= dt)
print("The call price by using implicit method is:", call_price)
#%% Debug for Problem 5
"""
Debug for Problem 5
"""
x = np.log(100)
K= 100
r= 0.07
T= 1
sigma= 0.35
a = 1
b = 0.5
x_min = 1
x_max = 1000
dx= 2
dt= 5/2400



S_0 = np.exp(x)
S_min = np.exp(x_min)
S_max = np.exp(x_max)
M = np.round((x_max - x_min) / dx)
M = M.astype(int)
dx = (x_max - x_min) / M
N = np.round(T / dt)
N = N.astype(int)
dt = T / N

matval = np.zeros(shape = (M + 1, N + 1))
vetx = np.transpose(np.linspace(x_min, x_max, M + 1))
veti = np.linspace(0, M, M + 1)
vetj = np.linspace(0, N, N + 1)
    # set up boundary conditons
    # Note: 
    # here the matval shape is (51,101)
    # because python start to count with 0 
matval[:, N] = np.maximum(K - np.exp(vetx), 0) # here the boundary condition should be define as [:,N] instead of [:, N + 1] in python
matval[0, :] = K * np.exp(-r * dt * (N - vetj))
matval[M, :] = 0

    # set up the tridiagonal coefficients matrix
# a = 0.5 * (r * dt * veti - sigma ** 2 * dt *(veti ** 2 ))
w = ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

y = 1 + sigma ** 2 * dt + r * dt * veti

z = - ((a - b * (x_min + veti * np.log(vetx))) / (2 * np.log(vetx))) * dt - 0.5 * sigma ** 2 * dt

coeff = np.diag(w[2:M], -1) + np.diag(y[1:M]) + np.diag(z[1:M-1], 1)
P,L,U = scipy.linalg.lu(coeff) 

    # solve the sequence of linear system
aux = np.zeros(shape = (M-1, 1))

for j in reversed(range(N)):
    aux[0] = - w[1] * matval[0, j]
    matval[1:M, j] = np.matmul(np.linalg.inv(U), np.matmul(np.linalg.inv(L),(matval[1:M, j+1] + aux)[0]))

f_x = interp1d(vetx, matval[:,0])
price = f_x(S_0)
print(price)
# %%
