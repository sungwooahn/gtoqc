import cirq
import numpy as np
import math
import scipy
from scipy.stats import boltzmann
from parse import compile
from numpy import log as ln

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

#### usage
# python3 samsung_noise.py

parse_result = compile("result={}")
    
ql = cirq.LineQubit.range(30)        

# Build your circuit
shot_number = 300
x_axis_number = 100 

theta_list = []

for index in range(x_axis_number):
    theta=(10*np.pi/x_axis_number)*index
    theta_list.append(theta)

eta = 0.1
n_bar_list = [0.3, 1.5, 5]
n_bar = n_bar_list[2]
theta=theta_list[0]

lambda_ = ln(1+1/n_bar)
N=100
bolt=boltzmann(lambda_, N)
shot=bolt.rvs(shot_number)

s = cirq.Simulator()
counter_list=[]

for x_axis in range(x_axis_number):
    theta=theta_list[x_axis]
    counter = 0
    for index in range(shot_number):
        n = shot[index]
        LP_func = scipy.special.genlaguerre(n, 0)
        theta_samsung = theta * math.exp(n_bar * (eta**2)) * LP_func(eta**2)
        
        samsung_circuit=cirq.Circuit(cirq.Rx(rads=theta_samsung)(ql[0]), cirq.measure(ql[0], key='result'))
        
        sample=s.run(samsung_circuit, repetitions=1)
        result=(parse_result.parse(str(sample)))[0]
        if result=="1":
            counter+=1
    counter_list.append(counter)

print(counter_list)

    

fig = plt.figure(figsize=(10,4))
ax = plt.axes()
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.grid(True,which='major',axis='both',alpha=0.3)
         
font1 = {'weight': 'bold',
         'size': 12
         }
plt.xlabel('theta', labelpad=5, fontdict=font1)
plt.ylabel('Number of 1',labelpad=5, fontdict=font1)

ax.plot(theta_list,counter_list,marker='o',markersize=8,markerfacecolor='orange',color='orange',markeredgecolor='black', markeredgewidth=2, linestyle='--',linewidth=3,label='n_bar: 5')
ax.legend()

save_file="./samsung_result/5_result.png"
ax.figure.savefig(save_file, bbox_inches='tight')