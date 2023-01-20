# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import solver as solver
import dom
import tensorflow as tf

num_nodes = 100
num_layers = 3
domain = dom.Box3D()
rho = 1.0 
gamma = 5/3.
mu0 = 1.0
init_mu = 1. 

# domain.plot_boundary()

value = 6.16
beta = 1000.
epochs = 20000 
n_sample = 1000 
save_dir = "../data/solution_BN"
factor_mu = (500.)**(1./epochs)

system = solver.Solver(domain, value)
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
optimizer = tf.keras.optimizers.Adam(learning_rate)
system.learn(optimizer, epochs, n_sample, save_dir)
system.plot(resolution=6, save_dir=save_dir)