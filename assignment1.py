# Python code for MTH 563-01 Computation Finance Assignment 1, Spring 2022
# Author : Qiancheng Sun
# In this assignment include 3 questions 
# 1. Solve a linear system AX = b by using different method
# 2. Use the Gauss-Hermite quadrature formular to approximate E[f(x)]  where  X ~ N(mu, sigma^2)
# 3. Determining the weight for the Simpson's rule

#%% import packages
import numpy as np
import scipy.linalg  # load scipy.linalg in order to call the function scipy.linalg.lu()
#%% 1. Solve a linear system AX = b by using different method
"""
Note : 
    The code in Python is different from MATLAB, 
    For some case in MATLAB the for loop start with 1 and end with the length of the element
    However, in Python the calculation start with 0, that will make the result not satisfy as the expect calculation result in MATLAB
    Therefore, in the code, 
    I setup the for loop calculation start with 1 and length of A will be add 1, in order to satisfy the expected result
    More details please check the following code
"""
# Part A :
# u is a random number 
# use np.random.rand() to generate the random number (0,1)
u = np.ndarray(shape = (6)) # create a 1 x 6 matrix for u

b = np.transpose(np.ndarray(shape = (1, 6))) # b is a 1 x 6 matrix

# create a full zero matrix with 6 x 6 shape 
A = np.zeros(shape = (6,6)) # A is a 6 x 6 matrix 
# make A as the Orthogonal matrix

for i in range(1, len(A) + 1):
    """
    Here the for loop start the calculation with 1, 
    then in Python the range will be start from 1 to 5.
    in order to do the for loop 6 times,
    the len(A) has to be add 1, to get the expected result

    Also, in order to satisfy the Python environment,
    the first element that comes out of Python count as 0, 
    therefore, 
    in for loop has to use i - 1 to satisfy the result, 
    because when i = 1, i -1 = 0 which fit for the rule for python counting 
    
    """
    u[i-1] = np.random.rand() # generate random number for each element in u
    b[i-1] = i ** 2 - 15  # bi = i ^ 2 - 15
    for j in range(1, len(A) + 1):
        if i == j :
            # A(ii) = 100 * u
            A[i-1, j-1] = 100 * u[i-1]
        else:
            # A(ij) = (-1)^(i+j) * (i + j)
            A[i-1, j-1] = (i + j) * (-1) ** (i + j)


# Part B : calculation the condition number of matrix A
""""
Use np.linalg.cond() to find the condition number of matrix A
Reference website:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
"""
# Here is the question how can I define if the condition number is big or small?
# Is there any threshold that I can set up to do the comparsion
condition_number = np.linalg.cond(A) 

# Part C : calculate the inverse of matrix A and solve X
"""
Use np.linalg.inv() function to find the inverse matrix of A
Reference website:
https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html

Use np.transpose() function to do transpose operations on vector b 
Turn the shape from 1 x 6 into 6 x 1
Reference website:
https://numpy.org/doc/stable/reference/generated/numpy.transpose.html

"""
X = np.linalg.inv(A) * np.transpose(b)

# Part D : Find the LU-decomposition of A
"""
Use scipy.linalg.lu() function to find the P, L, U
Here the input for the scipy.linalg.lu() function is the matrix A,
The output for the scipy.linalg.lu() function is :
    P : permutation matrix
    L : lower triangular with unit diagonal elements
    U : upper triangualr 
Reference website :
https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.lu.html

"""
P, L, U = scipy.linalg.lu(A)

# Part E : Use Jacobi iterative method to find approximate solution to the linear system.

def Jacobi(Input_matrix_A, Input_vector_b, Initial_X, epsilon, max_iter):
    """
    Jacobi() function is the way to solve linear equations AX = b

    The method that will be used is Iterative methos for solving system linear equations.

    The reference book is : "Basic of Numerical Analysis" Page 160

    Input variables :

        Input_matrix : Input matrix A (shape : N x N) for linear equations

        Input_vector_b : Input of the b vector (shape : N x 1) for linear equation

        Initial_X : Initial X_0 value of the linear equation (shape : 1 x N)

        epsilon : A threshold which is use for the tolerance 

        max_iter : Maximum iteration of the calculation to get the converge result

    Output variables :

        x : The converged solution of the linear equations (shape = N x 1)

        i : The number of iterations that the calculation did in the code to solve obtain the converage result
    
    """
    # get elements on diagonal of A
    d_a = np.diag(Input_matrix_A)  # get diagonal value from the matrix A
    # obtain the matrix C, based on equation below:
    # A = D + C
    C = Input_matrix_A - np.diag(d_a)
    # Ch
    D_inv = np.diag(np.divide(1, d_a))
    # Letting B = -D^(-1) * C
    B = - np.matmul(D_inv, C)
    # Check the b1 shape 
    """
    Here should use dot product to do the calculation in order to make b1 is 6 x 1 vector
    """
    b1 = np.matmul(D_inv, Input_vector_b) 
    """
    Here in order to do the for loop for the linear system calculation.

    In MATLAB doesnt have to put "i" in the for loop, however, in python has to use "i" include in for loop. 

    Therefore, in here the work flow will be :

    1. Generate an empty matrix with all Nan values as the elements in the Matrix

    2. Force the first row in the empty matrix as the Input initial X values

    3. During the for loop, every time when the new x generate that not satisfied the if condition, put it into "old_x[i - 1]"
    
    4. Generate X until it satisfied the if condition, print the X result, and the iterative number "i"
    """
    # set up the initial X_0
    n_matrix_shape = Input_matrix_A.shape[0]
    # create the empty numpy 
    old_x = np.empty(shape = (max_iter, n_matrix_shape))
    # make all the values as nan
    old_x[:,:] = np.nan
    # force the first value initial value become the initial_X
    old_x[0,:] = Initial_X

    # Use for loop to do the interative calculation to force the result converge
    for i in range(1, max_iter + 1):
        x = np.matmul(B, np.array(np.transpose(old_x[i - 1, :])).reshape(6 , 1)) + b1
        # set up a threshold to stop the converge
        # if the error smaller than the norm * epsilon then stop the calculation,
        # Otherwise keep doing the calculation
        """
        Generate two conditions for if condition

        Condition_a : x - old_x

        Condition_b : old_x

        Then calculate the norm for condition_a and condition_b
        """
        condition_a = x - np.array(np.transpose(old_x[i - 1, :])).reshape(6 , 1)
        condition_b = np.array(np.transpose(old_x[i - 1, :])).reshape(6 , 1)

        if np.linalg.norm(condition_a) > epsilon * np.linalg.norm(condition_b):
            old_x[i, :] = np.array(x).reshape(6)
        else:
            break
    # return the smallest x and i 
    # no need to append to generate the list, because the value that we are looking for is the smallest
    print("The epsilon for this case is :", epsilon)
    print("Approximate solution for X is :", x)
    print("The number of iterations are :", i)
    return x , i
# First question in Part E
# Set the Maxiter = 10000, and consider the epsilon = 10 ^ -2
x_1 , i_1 = Jacobi(Input_matrix_A= A,
                Input_vector_b= b,
                Initial_X = np.zeros(shape = (1,6)),
                epsilon= 10 ** -2,
                max_iter= 10000)
# Second question in Part E
# Set the Maxiter = 10000, and consider the epsilon = 10 ^ -3
x_2 , i_2 = Jacobi(Input_matrix_A= A,
                Input_vector_b= b,
                Initial_X = np.zeros(shape = (1,6)),
                epsilon= 10 ** -3,
                max_iter= 10000)
# Third question in Part E
# Set the Maxiter = 10000, and consider the epsilon = 10 ^ -4
x_3 , i_3 = Jacobi(Input_matrix_A= A,
                Input_vector_b= b,
                Initial_X = np.zeros(shape = (1,6)),
                epsilon= 10 ** -4,
                max_iter= 10000)
# Fourth question in Part E
# Set the Maxiter = 10000, and consider the epsilon = 10 ^ -5
x_4 , i_4 = Jacobi(Input_matrix_A= A,
                Input_vector_b= b,
                Initial_X = np.zeros(shape = (1,6)),
                epsilon= 10 ** -5,
                max_iter= 10000)


# %% 2. Use the Gauss-Hermite quadrature formular to approximate E[f(x)]  where  X ~ N(mu, sigma^2)

# define the Gauss-Hermite function
def GaussHermite(mu, sigma2, N):
    """
    Function description :

    This is the function to use Gauss-Hermite method to  


    Input variables :

    mu : Random Variables

    sigma2 :

    N : number of nodes
    
    """
    Hpoly_1 = np.array([0, 0 ,1/ np.pi ** 0.25]).reshape(3)
    print(Hpoly_1)
    Hpoly_2 = np.array([np.sqrt(2)/np.pi ** 0.25 , 0, 0]).reshape(3) # define the Hpoly as a 
    print(Hpoly_2)
    Hpoly_3_list = []
    for i in range(1, (N - 1)):
        a = np.sqrt(2 / (i + 1)) * Hpoly_2  # a will be a 1x3 vector
        b = np.sqrt(i / (i + 1)) * Hpoly_1  # b will be 
        Hpoly_3 = a - b
        Hpoly_3_list.append(Hpoly_3)
        Hpoly_1 = Hpoly_2
        Hpoly_2 = Hpoly_3




    x_1 = np.roots(Hpoly_3)
    w_1 = np.zeros(shape = (N,1))
    w_1_list = []
    for i in range(1, N):
        w_1[i] = 1/(N)/ np.polyval(Hpoly_1, x_1[i]) ** 2
        w_1_list.append(w_1)
    [x, index] = np.array(np.sort(x_1 * np.sqrt(2 * sigma2) + mu))
    w = np.array(w_1[index]/np.sqrt(np.pi))

    return x, w




# %%
x , w = GaussHermite(10,20,5)


# %%
N = 5
Hpoly_1 = np.array([1/ np.pi ** 0.25])
print(Hpoly_1)
Hpoly_2 = np.array([np.sqrt(2)/np.pi ** 0.25 , 0]) # define the Hpoly as a 
print(Hpoly_2)

Hpoly_3_list = []
Hpoly_1_list = []
Hpoly_2_list = []
for i in range(1, (N)):
    a = np.sqrt(2 / (i + 1)) * Hpoly_2  # a will be a 1x3 vector
    b = np.sqrt(i / (i + 1)) * Hpoly_1  # b will be 
    Hpoly_3 = a - b
    Hpoly_3_list.append(Hpoly_3)
    Hpoly_1 = Hpoly_2
    Hpoly_1_list.append(Hpoly_1)
    Hpoly_2 = Hpoly_3
    Hpoly_2_list.append(Hpoly_2)



print(Hpoly_3_list)
#%%
sigma2 = 20
mu = 10
w_1 = np.zeros(shape = (N,1))
w_1_list = []
for i in range(1, N):
    x_1 = np.roots(Hpoly_3_list[i - 1])

    c = np.array(1/(N)/ np.polyval(Hpoly_1, x_1) ** 2)
    print(c)
    w_1 = c[0]
    w_1_list.append(w_1)
#%%
[x, index] = np.array(np.sort(x_1 * np.sqrt(2 * sigma2) + mu))
w = np.array(w_1/np.sqrt(np.pi))


# %%
Hpoly_1 = np.array([1/ np.pi ** 0.25])
print(Hpoly_1)
Hpoly_2 = np.array([np.sqrt(2)/np.pi ** 0.25 , 0]) # define the Hpoly as a 
print(Hpoly_2)
#%%
a = np.sqrt(2 / (i + 1)) * Hpoly_2  # a will be a 1x3 vector
b = np.sqrt(i / (i + 1)) * Hpoly_1  # b will be 
Hpoly_3 = a - b
print(Hpoly_3)
#%%
Hpoly_1 = Hpoly_2
print(Hpoly_1)
Hpoly_2 = Hpoly_3
print(Hpoly_2)
# %%
# Question 3