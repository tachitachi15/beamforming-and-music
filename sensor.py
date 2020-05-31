import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

M = 5
J = 2
I = 4


lamda = 4/np.sqrt(2)
d = 1
theta_1 = np.pi/2 #90
theta_2 = np.pi/4 #45
var = 0.01

s_1 = np.array([1,1,-1,-1])
s_2 = np.array([1,1j,-1,0])


S = np.array([s_1,s_2])
h_1 = np.zeros(M,dtype=np.complex)
h_2 = np.zeros(M,dtype=np.complex)
for i in range(M):
    h_1[i] = np.exp(complex(0,i)*2*np.pi*d*np.cos(theta_1)/lamda)
    h_2[i] = np.exp(complex(0,i)*2*np.pi*d*np.cos(theta_2)/lamda)

H = np.array([h_1,h_2]).T
N = np.zeros((M,I),dtype=np.complex)
N.real = np.random.normal(0,np.sqrt(var/2),(M,I))
N.imag = np.random.normal(0,np.sqrt(var/2),(M,I))

Y = np.dot(H,S) + N

R = np.dot(Y,np.conjugate(Y.T))/I

#BF
theta_est = np.linspace(0,np.pi,180)

P_bf = np.zeros(180,dtype=np.float32)
h_conj = np.zeros(M,dtype=np.complex)

for i in range(180):
    for j in range(M):
        h_conj[j] = np.exp(complex(0,-j)*2*np.pi*d*np.cos(theta_est[i])/lamda) #hの共役
    P_bf[i] = h_conj@R@np.conjugate(h_conj.T)

#MUSIC
P_music = np.zeros(180,dtype=np.float32)
eig_val, eig_vec =linalg.eig(R)
eig_vec = eig_vec.T
sig_num = 0
for i in range(M):
    if eig_val[i]>var:
        sig_num+=1

E_n = eig_vec[sig_num:M]
E_n = E_n.T
for i in range(180):
    for j in range(M):
        h_conj[j] = np.exp(complex(0,-j)*2*np.pi*d*np.cos(theta_est[i])/lamda) 
    temp= h_conj@E_n@np.conjugate(E_n.T)@np.conjugate(h_conj.T)
    P_music[i]=1/temp

fig = plt.figure(figsize=(8, 8))
plt.xlabel("degree")
plt.plot(np.rad2deg(theta_est),P_bf,label="BF")
plt.plot(np.rad2deg(theta_est),P_music,label="MUSIC")

plt.legend()
plt.savefig("d1l4I4.png")
plt.show()
