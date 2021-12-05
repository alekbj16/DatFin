#Import
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from numpy.lib.twodim_base import diag 

#***1: Efficient frontier with and withoud riskless lending and borrowing***

r_f = 0.02 #Risk free rate: 2%
r_1 = 0.10 #Asset 1 yearly expected return: 10%
s_1 = 0.10 #Asset 1 standard dev
r_2 = 0.20 #Asset 2 yearly expected return: 20%
s_2 = 0.20 #Asset 2 standard dev

corr_coeff = np.array([1,0.5,0,-1]) #The different correlation coefficients between asset 1 and 2

#***1.1: How to plot efficient frontier *with* borrowing

#****1.1.1: 
#For the case rho = 0.5
rho = corr_coeff[1]
correlation_matrix = np.zeros((2,2))
correlation_matrix[0,0] = 1
correlation_matrix[1,1] = 1
correlation_matrix[0,1] = rho
correlation_matrix[1,0] = rho

mu = np.array([r_1,r_2])
sigma = np.array([s_1,s_2])
omega = np.dot(np.dot(np.diag(sigma),correlation_matrix),diag(sigma)) #Could have used numpy multidot
print(omega)


A = 2*omega
A = np.c_[ A, -(mu-r_f)]
A = np.vstack([A, np.append([(mu-r_f)],0)])

r_p = 0.4 #Assuming a trivial value for expected return. Does not affect when finding optimal X because of normalization
b = np.zeros((2,1))
b = np.vstack([b,r_p-r_f])

X = np.dot(np.linalg.inv(A),b)
print(f"X{X}")
X_opt = X[0:2]*(1/np.sum(X[0:2]))
print(X_opt)

return_optimal = X_opt[0]*r_1 + X_opt[1]*r_2
print(return_optimal)

var = (X_opt[0]**2)*(s_1**2)+(X_opt[1]**2)*(s_2**2)+2*X_opt[0]*X_opt[1]*s_1*s_2*rho
slope = (return_optimal-r_f)/var
print(var)
print(slope)
x = np.linspace(0,5.0,100)
f_x = slope*x+r_f
plt.plot(x,f_x)
plt.show()
# Xopt is the point which has highest slope and 
# describes the optimal amount of stock

#print(correlation_matrix)
#print(omega)

