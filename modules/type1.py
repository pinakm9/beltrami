import imp
import pandas as pd
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


class Network(tf.keras.models.Model):
    def __init__(self, out_dim, name='vanilla_network', dtype=tf.float32, last_bias=None):
        super().__init__(dtype=dtype, name=name) 
        self.middle_layers = [tf.keras.layers.Dense(units=50, activation='tanh', dtype=dtype) for _ in range(2)]
        self.final_layer = tf.keras.layers.Dense(units=out_dim, activation=None, use_bias=last_bias, dtype=dtype)
  
    def call(self, *args):
        x = tf.concat(args, axis=-1)
        for ml in self.middle_layers:
            x = ml(x)
        return self.final_layer(x)

class Solver:
    def __init__(self, domain):
        self.domain = domain 
        self.A = Network(out_dim=3, name="potential", dtype=domain.dtype, last_bias=False)
        self.lam = Network(out_dim=1, name="multiplier", dtype=domain.dtype, last_bias=True)
        self.rho = 1.0

    
    def value_x(self, x, y, z):
        return tf.sin(z) + tf.cos(y)

    def value_y(self, x, y, z):
        return tf.sin(x) + tf.cos(z)

    def value_z(self, x, y, z):
        return tf.sin(y) + tf.cos(x)

    def value(self, x, y, z):
        return tf.concat([self.value_x(x, y, z), self.value_y(x, y, z), self.value_z(x, y, z)], axis=-1)
    
    

    @tf.function
    def B(self, x, y, z):
        return curl(self.A, x, y, z)


    @tf.function
    def divB(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z) 
        return Bx_x + By_y + Bz_z
    
    @tf.function
    def curlB(self, x, y, z): 
        return curl(self.B, x, y, z)


    
    def loss_F(self, front_boundary_data):
        x, y, z, nx, ny, nz = front_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#tf.reduce_mean((By - self.value_y(x, y, z))**2) 

    def loss_B(self, back_boundary_data):
        x, y, z, nx, ny, nz = back_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#return tf.reduce_mean((By - self.value_y(x, y, z))**2) 

    def loss_U(self, up_boundary_data):
        x, y, z, nx, ny, nz = up_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#return tf.reduce_mean((Bz)**2) 

    def loss_D(self, down_boundary_data):
        x, y, z, nx, ny, nz = down_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#return tf.reduce_mean((Bz)**2)
    
    def loss_L(self, left_boundary_data):
        x, y, z, nx, ny, nz = left_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#return tf.reduce_mean((Bx - self.value_x(x, y, z))**2)

    def loss_R(self, right_boundary_data):
        x, y, z, nx, ny, nz = right_boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        _Bx, _By, _Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        
        return tf.reduce_mean((Bx-_Bx)**2 + (By-_By)**2 + (Bz-_Bz)**2)#return tf.reduce_mean((Bx - self.value_x(x, y, z))**2)


    def loss_energy(self, domain_data):
        Bx, By, Bz = tf.split(self.B(*domain_data), 3, axis=-1)
        return 0.5*tf.reduce_mean(Bx**2 + By**2 + Bz**2)

    def loss_curlB(self, domain_data):
        p, q, r = tf.split(self.curlB(*domain_data), 3, axis=-1)
        p1, q1, r1 = tf.split(self.B(*domain_data), 3, axis=-1)
        mod_curl2 =  (p-p1)**2 + (q-q1)**2 + (r-r1)**2
        mod_curl = tf.sqrt(mod_curl2)
        return  (self.rho**2 / 2.0) * tf.reduce_mean(mod_curl2) +  tf.reduce_mean(mod_curl * self.lam(*domain_data))


    def total_loss_b(self, boundary_data):
        right, left, front, back, up, down  = boundary_data
        loss = self.loss_F(front) + self.loss_B(back) + self.loss_U(up) + self.loss_D(down) +\
               self.loss_L(left) + self.loss_R(right)
        return loss
    

    
    @tf.function
    def train_step_main(self, optimizer, domain_data, boundary_data):
        with tf.GradientTape() as tape:
            curl_loss = self.loss_curlB(domain_data)
            bdry_loss = self.total_loss_b(boundary_data)
            engy_loss = self.loss_energy(domain_data)
            loss = engy_loss + 100.*bdry_loss + curl_loss
        grads = tape.gradient(loss, self.A.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.A.trainable_weights))
        return loss, curl_loss, bdry_loss, engy_loss
        

    @tf.function
    def train_step_mul(self, optimizer, domain_data):
        p, q, r = tf.split(self.curlB(*domain_data), 3, axis=-1)
        p1, q1, r1 = tf.split(self.B(*domain_data), 3, axis=-1)
        mod_curl2 = (p-p1)**2 + (q-q1)**2 + (r-r1)**2
        mod_curl = tf.sqrt(mod_curl2)
        mul0 = self.lam(*domain_data) + self.rho * mod_curl
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean((self.lam(*domain_data) - mul0)**2)
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
        losses = np.zeros((epochs, 6))
        self.plot_solution(6, save_dir, 0)
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                #self.rho = (1000.)**(epoch/epochs)
                for i in range(10):
                    l1, curl_loss, bdry_loss, engy_loss = self.train_step_main(optimizers[0], domain_data, boundary_data)
                for i in range(10):
                    l2 = self.train_step_mul(optimizers[1], domain_data)
                div_loss = tf.reduce_mean(self.divB(*domain_data)**2)
                curl_loss = self.loss_curlB(domain_data)
                losses[epoch, :] = np.array([l1.numpy(), curl_loss.numpy(), bdry_loss.numpy(),\
                               engy_loss.numpy(), div_loss.numpy(), l2.numpy()])
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:12.6f}{:12.4f}'.format(epoch, l1, l2, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)
                    self.A.save_weights('{}/{}'.format(save_dir, self.A.name))
                    self.lam.save_weights('{}/{}'.format(save_dir, self.lam.name))
        pd.DataFrame(losses).to_csv('{}/loss.csv'.format(save_dir), index=None,\
        header=['main', 'curl', 'boundary', 'energy', 'div', 'multiplier'])
        self.plot_solution(4, save_dir, epoch+1)
        self.plot_loss(save_dir)
        

    
    def plot_solution(self, resolution, save_dir, index):
        #self.A.load_weights('{}/{}'.format(save_dir, self.A.name)).expect_partial()
        fig = plt.figure(figsize=(16, 16))
        ax_B = fig.add_subplot(121, projection='3d')
        ax_t = fig.add_subplot(122, projection='3d')

      
        x, y, z = self.domain.grid_sample(resolution)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)

        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(-1), q.reshape(-1), r.reshape(-1)
        ax_B.quiver(x, y, z, p, q, r, length=0.1, colors=['blue']*len(x))
        ax_B.set_title(r'Computed $B$', fontsize=40)
        ax_B.grid(False)
        ax_B.set_xlabel('x', fontsize=40)
        ax_B.set_ylabel('y', fontsize=40)
        ax_B.set_zlabel('z', fontsize=40)
        ax_B.tick_params(axis='x', labelsize=20)
        ax_B.tick_params(axis='y', labelsize=20)
        ax_B.tick_params(axis='z', labelsize=20)

        x, y, z = self.domain.grid_sample(resolution)
        Bx, By, Bz = tf.split(self.value(x, y, z), 3, axis=-1)
        x, y, z = x.reshape(-1), y.reshape(-1), z.reshape(-1)
        p, q, r = Bx.numpy(), By.numpy(), Bz.numpy()
        R = max(np.sqrt(p*p + q*q + r*r))
        p, q, r = p/R, q/R, r/R
        p, q, r = p.reshape(-1), q.reshape(-1), r.reshape(-1)
        ax_t.quiver(x, y, z, p, q, r, length=0.1, colors=['blue']*len(x))
        ax_t.set_title(r'True $B$', fontsize=40)
        ax_t.grid(False)
        ax_t.set_xlabel('x', fontsize=40)
        ax_t.set_ylabel('y', fontsize=40)
        ax_t.set_zlabel('z', fontsize=40)
        ax_t.tick_params(axis='x', labelsize=20)
        ax_t.tick_params(axis='y', labelsize=20)
        ax_t.tick_params(axis='z', labelsize=20)
        fig.savefig('{}/sol_{}.png'.format(save_dir, index))
        plt.close(fig)


    def plot_loss(self, save_dir):
        loss = np.genfromtxt('{}/loss.csv'.format(save_dir), dtype=np.float64, delimiter=',', skip_header=True)
        fig = plt.figure(figsize=(16, 16))
        losses = [r'$\log_{10}$(main loss)', r'$\log_{10}\left(\int_\Omega |\nabla\times B - B|^2\right)$', r'$\log_{10}$(boundary loss)',\
                 r'$\frac{1}{2}\int_\Omega B^2$', r'$\log_{10}\left(\int_\Omega (\nabla\cdot B)^2\right)$', r'$\log_{10}$(multiplier loss)']
        ax_m = fig.add_subplot(321)
        ax_c = fig.add_subplot(322)
        ax_b = fig.add_subplot(323)
        ax_e = fig.add_subplot(324)
        ax_d = fig.add_subplot(325)
        ax_mul = fig.add_subplot(326)

        axes = [ax_m ,ax_c, ax_b, ax_e, ax_d, ax_mul]
        x = list(range(1, len(loss)+1, 1))
        for i, ax in enumerate(axes):
            if i == 3:
                ax.plot(x, loss[:, i])
                ax.legend(fontsize=20)
            else:
                ax.plot(x, np.log10(loss[:, i]))
            ax.set_title(losses[i], fontsize=20)
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
        fig.tight_layout()
        fig.savefig('{}/loss.png'.format(save_dir))
        plt.close(fig)
        




    

    