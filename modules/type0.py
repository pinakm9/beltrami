import imp
from tracemalloc import DomainFilter
from xml import dom
import tensorflow as tf 
import arch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from curl import curl

"""
B = grad phi
"""




class Solver:
    def __init__(self, domain, value):
        self.domain = domain 
        self.value = value 
        self.phi = arch.LSTMForgetNet(num_nodes=50, num_layers=3, out_dim=1, name="potential", last_bias=False, dtype=tf.float64)
        self.lam = arch.LSTMForgetNet(num_nodes=50, num_layers=3, out_dim=1, name="multiplier", last_bias=False, dtype=tf.float64)
        self.rho = 1.0
    

    @tf.function
    def B(self, x, y, z):
        with tf.GradientTape() as tape:
            tape.watch([x, y, z])
            phi = self.phi(x, y, z)
        return tape.gradient(phi, [x, y, z])

    
    @tf.function
    def divB(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz = self.B(x, y, z)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z) 
        return Bx_x + By_y + Bz_z
        
    
    def loss_F(self, front_boundary_data):
        x, y, z, nx, ny, nz = front_boundary_data
        Bx, By, Bz = self.B(x, y, z)
        return 10.0 * tf.reduce_mean((By - self.value)**2) 

    def loss_B(self, back_boundary_data):
        x, y, z, nx, ny, nz = back_boundary_data
        Bx, By, Bz = self.B(x, y, z)
        return tf.reduce_mean((By - self.value)**2) 

    def loss_nB(self, boundary_data):
        x, y, z, nx, ny, nz = boundary_data
        Bx, By, Bz = self.B(x, y, z)
        return tf.reduce_mean((Bx*nx + By*nx + Bz*nz)**2)

    def loss_energy(self, domain_data):
        Bx, By, Bz = self.B(*domain_data)
        return tf.reduce_mean(Bx**2 + By**2 + Bz**2)

    def loss_divB(self, domain_data):
        return  1e8 * tf.reduce_mean(self.divB(*domain_data)**2) #tf.reduce_mean(self.divB(*domain_data) * self.lam(*domain_data))


    def total_loss_b(self, boundary_data):
        right, left, front, back, up, down  = boundary_data
        loss = self.loss_F(front) + self.loss_B(back)
        for data in [right, left, up, down]:
            loss += 0. #self.loss_nB(data)
        return loss
    

    def loss_mul(self, mul0, domain_data):
        return tf.reduce_mean((self.lam(*domain_data) - mul0)**2)

    @tf.function
    def train_step_main(self, optimizer, domain_data, boundary_data):
        with tf.GradientTape() as tape:
            loss =  100.*self.total_loss_b(boundary_data) + self.loss_divB(domain_data)
        grads = tape.gradient(loss, self.phi.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.phi.trainable_weights))
        return loss
        

    @tf.function
    def train_step_mul(self, optimizer, domain_data):
        mul0 = self.lam(*domain_data) + self.rho * self.divB(*domain_data)
        with tf.GradientTape() as tape:
            loss = self.loss_mul(mul0, domain_data)
        grads = tape.gradient(loss, self.lam.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.lam.trainable_weights))
        return loss

    def learn(self, optimizers, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>12}{:>18}".format('epoch', 'loss_main', 'loss_mul', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                self.rho = (1000.)**(epoch/epochs)
                l1 = self.train_step_main(optimizers[0], domain_data, boundary_data)
                l2 = 0.# self.train_step_mul(optimizers[1], domain_data)
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:12.6f}{:12.4f}'.format(epoch, l1, l2, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)
                    self.phi.save_weights('{}/{}'.format(save_dir, self.phi.name))
                    self.lam.save_weights('{}/{}'.format(save_dir, self.lam.name))
        

    def plot(self, resolution, save_dir):
        self.phi.load_weights('{}/{}'.format(save_dir, self.phi.name)).expect_partial()
        fig = plt.figure(figsize=(16, 16))
        ax_B = fig.add_subplot(221, projection='3d')
        ax_divB = fig.add_subplot(222, projection='3d')
        ax_modB = fig.add_subplot(223, projection='3d')
        ax_curlB = fig.add_subplot(224, projection='3d')
        x, y, z = self.domain.grid_sample(resolution)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)
        Bx, By, Bz = self.B(x, y, z)


        divB = self.divB(x, y, z).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(divB)
        ax_divB.scatter(x.flatten(), y.flatten(), z.flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_divB)
        ax_divB.set_title('$\\nabla\\cdot B$', fontsize=20)
        ax_divB.grid(False)

        x_ = tf.reshape(tf.convert_to_tensor(x, dtype='float64'), shape=(-1, 1))
        y_ = tf.reshape(tf.convert_to_tensor(y, dtype='float64'), shape=(-1, 1))
        z_ = tf.reshape(tf.convert_to_tensor(z, dtype='float64'), shape=(-1, 1))
        curlB = tf.sqrt(tf.reduce_sum(curl(lambda *args: tf.concat(self.B(*args), axis=-1), x_, y_, z_)**2, axis=-1)).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(curlB)
        ax_curlB.scatter(x.flatten(), y.flatten(), z.flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_curlB)
        ax_curlB.set_title('$|\\nabla\\times B|$', fontsize=20)
        ax_curlB.grid(False)

        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(-1), q.reshape(-1), r.reshape(-1)
        ax_B.quiver(x, y, z, p, q, r, length=0.2, colors=['blue']*len(x))
        ax_B.set_title('$B$', fontsize=20)
        ax_B.grid(False)

        modB = tf.sqrt(Bx**2 + By**2 + Bz**2).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(modB)
        ax_modB.scatter(x.flatten(), y.flatten(), z.flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_modB)
        ax_modB.set_title('$|B|$', fontsize=20)
        ax_modB.grid(False)

        plt.savefig('{}/solution.png'.format(save_dir))

    