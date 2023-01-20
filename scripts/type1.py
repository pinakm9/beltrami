# add modules to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
sys.path.insert(0, module_dir + '/modules')
print(module_dir)

import type1 as solver
import dom
import tensorflow as tf

domain = dom.Box3D()

# domain.plot_boundary()

value = 6.16
epochs = 1000 
n_sample = 1000 
save_dir = "../data/type1"


system = solver.Solver(domain, value)
learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [1e-2, 1e-3, 5e-4, 1e-4])
optimizers = [tf.keras.optimizers.Adam(learning_rate) for i in range(2)]
#system.A.load_weights('{}/{}'.format(save_dir, system.phi.name))
system.learn(optimizers, epochs, n_sample, save_dir)
