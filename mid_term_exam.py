# Code answer for Mid-term exam
# %%
# import packages
import numpy as np
from scipy.integrate import quad
from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt
#%%
# Question 1, develop a quadrature formula using five equally spaced notes in [0,1]
# so that it is exact for all polynomial of degree <= 4
# Part A:
# Find weights w_i

# Since y are five equally space notes in [0,1]
# it could be easily define as the following:
y = np.array([1, 1/2, 1/3, 1/4, 1/5]).reshape(5,1) # define y shape as (5,1)
x = np.array([0, 1/4, 1/2, 3/4, 1]).reshape(5,1) # define the input value for x

# define the input x
f_x_A = [x**0, x, x**2, x**3, x**4]
X = np.ndarray(shape = (5,5))

# generate a 5x5 matrix
for i in range(0,len(f_x_A)):
    X[i,:] = np.array(f_x_A[i]).reshape(len(f_x_A)) 
print(X)

# Here to generate vector w_i by using matrix X and result y
"""
Methodology :

W * X = y

Reference Website:

https://en.wikipedia.org/wiki/Linear_regression
"""
W = (np.matmul(np.matmul(np.linalg.inv((np.matmul(np.transpose(X), X))),np.transpose(X)),y)).reshape(1,len(f_x_A))
print("The weight for quadrature formula is:", W)
# %%
# Part B:
def f_1(x):
    return x**5
result_1, error_1 = quad(f_1, 0 , 1)
print("the exact value for x^5 is: ", result_1)
approximate_value_1 = np.matmul(W,f_1(x))
print("the error of approximate value - exact value is: ",abs(approximate_value_1 - result_1))

def f_2(x):
    return np.exp(x)
result_2, error_2 = quad(f_2, 0 , 1)
print("the exact value for e^x is: ", result_2)
approximate_value_2 = np.matmul(W,f_2(x))
print("the error of approximate value - exact value is: ",abs(approximate_value_2 - result_2))

def f_3(x):
    return np.sin(x)
result_3, error_3 = quad(f_3, 0 , 1)
print("the exact value for sin(x) is: ", result_3)
approximate_value_3 = np.matmul(W,f_3(x))
print("the error of approximate value - exact value is: ",abs(approximate_value_3 - result_3))

def f_4(x):
    return 1/(1 + x**2)
result_4, error_4 = quad(f_4, 0 , 1)
print("the exact value for 1/(1+x^2) is: ", result_4)
approximate_value_4 = np.matmul(W,f_4(x))
print("the error of approximate value - exact value is: ",abs(approximate_value_4 - result_4))


# %%
# Problem 2: Use the Black-Scholes formula to price European call and put options

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

#%% 
# Part A
# question part a 1)
# S = np.array([40,45,48,50,52,55,60])
# question part a 2)
S = np.linspace(30,70, 70-30+1)
T = 0.5
r = 0.05
sigma = 0.4
K = 50 
call_price_list = []
put_price_list = []
for i in range(0, len(S)):
    call_price = bs_call(S = S[i], K = K, T = T, r = r, sigma = sigma)
    call_price_list.append(call_price)

    put_price = bs_put(S = S[i], K = K, T = T, r = r, sigma = sigma)
    put_price_list.append(put_price)   

# plot the call price
x = S
y = call_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Call Price")
plt.xlabel("stock price")
plt.ylabel("call price")
plt.show()
# plot the call price
x = S
y = put_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Put Price")
plt.xlabel("stock price")
plt.ylabel("Put price")
plt.show()


# %%
# Part B
S = 49
# Question 1)
# T = np.linspace(0.3, 0.8, 6)
# Question 2) 
T = np.linspace(0.1, 1, 21)
r = 0.05
sigma = 0.4
K = 50 
call_price_list = []
put_price_list = []
for i in range(0, len(T)):
    call_price = bs_call(S = S, K = K, T = T[i], r = r, sigma = sigma)
    call_price_list.append(call_price)

    put_price = bs_put(S = S, K = K, T = T[i], r = r, sigma = sigma)
    put_price_list.append(put_price)  

 
# plot the call price
x = T
y = call_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Call Price vs Maturity")
plt.xlabel("Maturity")
plt.ylabel("call price")
plt.show()
# plot the call price
x = T
y = put_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Put Price vs Maturity")
plt.xlabel("Maturity")
plt.ylabel("Put price")
plt.show()
# %%
# Part C
S = 50
T = 1
r = 0.05
# question 1)
# sigma = np.linspace(0.2,0.5, 7)
# question 2)
sigma = np.linspace(0.1,0.8, int((0.8 - 0.1)/ 0.02 + 1))
K = 50 
call_price_list = []
put_price_list = []
for i in range(0, len(sigma)):
    call_price = bs_call(S = S, K = K, T = T, r = r, sigma = sigma[i])
    call_price_list.append(call_price)

    put_price = bs_put(S = S, K = K, T = T, r = r, sigma = sigma[i])
    put_price_list.append(put_price)  

 
# plot the call price
x = sigma
y = call_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Call Price vs Volatility")
plt.xlabel("Volatility")
plt.ylabel("call price")
plt.show()
# plot the call price
x = sigma
y = put_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Put Price vs Volatility")
plt.xlabel("Volatility")
plt.ylabel("Put price")
plt.show()

# %%
# Part D
S = 52
T = 1
# question 1)
# r = np.linspace(0.02, 0.07, int((0.07 - 0.02)/ 0.01 + 1))
# question 2)
r = np.linspace(0.01, 0.08, int((0.08 - 0.01)/ 0.005 + 1))
sigma = 0.35
K = 50 
call_price_list = []
put_price_list = []
for i in range(0, len(r)):
    call_price = bs_call(S = S, K = K, T = T, r = r[i], sigma = sigma)
    call_price_list.append(call_price)

    put_price = bs_put(S = S, K = K, T = T, r = r[i], sigma = sigma)
    put_price_list.append(put_price)  

 
# plot the call price
x = r
y = call_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Call Price vs Interest Rate")
plt.xlabel("Interest Rate")
plt.ylabel("call price")
plt.show()
# plot the call price
x = r
y = put_price_list
# figure define
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.title("Put Price vs Interest Rate")
plt.xlabel("Interest Rate")
plt.ylabel("Put price")
plt.show()
# %%
# Part E : In MatLab Code

# %%
# Question 3 slove heat equation by explicit finite difference method
# define heat explicit function
def heat_explicit(delta_x, delta_t, t_max, K):
    """
    function to solve heat equation by explicit finite difference method
    
    Reference :

    Example 5.3 on Page 305, Figure 5.14 on Page 306
    
    """
    N = np.round(2 / delta_x)
    N = int(N)
    M = np.round(t_max / delta_t)
    M = int(M)
    sol = np.zeros(shape = (N + 1, M + 1))
    rho = delta_t / (delta_x ** 2) * K
    rho2 = 1 - 2 * rho
    # devide 0 to 1 based on delta_x
    vetx = np.linspace(1,3, int(2/delta_x) + 1)

    for i in range(1, int(np.ceil((N + 1) / 2))): # from 2 to 6
        sol[i,0] = 2 * (vetx[i] - 1)
        sol[N - i, 0] = sol[i,0]
    
    for j in range(0, M):
        for k in range(1, N):
            sol[k, j + 1] = rho * sol[k - 1, j] + rho2 * sol[k,j] + rho * sol[k + 1, j]

    return sol

# define variables
delta_x = 0.2
delta_t = 0.002
t_max = delta_t * 1000

sol = heat_explicit(delta_x = delta_x,delta_t = delta_t,t_max = t_max, K = 3/2)
# plot the result
# t = 0
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,0]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()

# t = 0.01
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,10]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()
# t = 0.05
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,50]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()
# t = 0.1
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,100]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()
# t = 0.5
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,500]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()
# t = 1
x = np.linspace(1,3, int(2/delta_x) + 1)
y = sol[:,1000]
figure = plt.figure()
ax = plt.axes()
ax.plot(x,y)
plt.show()

# %%
