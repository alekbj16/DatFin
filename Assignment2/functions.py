import numpy as np
from numpy.lib.twodim_base import diag 

def optimal_portofolio(r_f,r_1,r_2,s_1,s_2,rho):
    correlation_matrix = np.zeros((2,2))
    correlation_matrix[0,0] = 1
    correlation_matrix[1,1] = 1
    correlation_matrix[0,1] = rho
    correlation_matrix[1,0] = rho
    mu = np.array([r_1,r_2])
    sigma = np.array([s_1,s_2])
    omega = np.dot(np.dot(np.diag(sigma),correlation_matrix),diag(sigma)) #Could have used numpy multidot
    A = 2*omega
    A = np.c_[ A, -(mu-r_f)]
    A = np.vstack([A, np.append([(mu-r_f)],0)])
    r_p = 0.4 #Assuming a trivial value for expected return. Does not affect when finding optimal X because of normalization
    b = np.zeros((2,1))
    b = np.vstack([b,r_p-r_f])
    X = np.dot(np.linalg.inv(A),b)
    X_opt = X[0:2]*(1/np.sum(X[0:2]))
    return X_opt

def portofolio_return(r_1,r_2,X_opt):
    return X_opt[0]*r_1 + X_opt[1]*r_2

def portofolio_variance(r_1,r_2,s_1,s_2,X_opt,rho):
    return (X_opt[0]**2)*(s_1**2)+(X_opt[1]**2)*(s_2**2)+2*X_opt[0]*X_opt[1]*s_1*s_2*rho

def get_slope(r_f,portofolio_return,portofolio_variance):
    return (portofolio_return-r_f)/portofolio_variance


