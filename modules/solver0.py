import tensorflow as tf 
import arch
import os
import time
import matplotlib.pyplot as plt
import numpy as np

class Solver:
    def __init__(self, domain, value):
        self.domain = domain 
        self.value = value 
        self.B = arch.LSTMForgetNet(num_nodes=50, num_layers=2, out_dim=3, name="magnetic_potential", last_bias=False)
    

    @tf.function
    def divB(self, x, y, z):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y, z])
            Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        Bx_x = tape.gradient(Bx, x)
        By_y = tape.gradient(By, y)
        Bz_z = tape.gradient(Bz, z)
        return Bx_x + By_y + Bz_z

    
    def loss_F(self, front_boundary_data):
        x, y, z, nx, ny, nz = front_boundary_data
        n = tf.concat([nx, ny, nz], axis=-1)
        Bn = tf.reduce_sum(self.B(x, y, z) * n, axis=-1)
        return 10.0 * (tf.reduce_mean(Bn) - self.value)**2 

    def loss_B(self, back_boundary_data):
        x, y, z, nx, ny, nz = back_boundary_data
        n = tf.concat([nx, ny, nz], axis=-1)
        Bn = tf.reduce_sum(self.B(x, y, z) * n, axis=-1)
        return 10.0 * (tf.reduce_mean(Bn) + self.value)**2 

    def loss_nB(self, boundary_data):
        x, y, z, nx, ny, nz = boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        return tf.reduce_mean((Bx*nx + By*nx + Bz*nz)**2)

    def loss_FB(self, front_boundary_data):
        x, y, z, nx, ny, nz = front_boundary_data
        BFx, BFy, BFz = tf.split(self.B(x, y, z), 3, axis=-1)
        BBx, BBy, BBz = self.B(x + self.domain.a, y, z)
        return tf.reduce_mean((BFx - BBx)**2 + (BFy - BBy)**2 + (BFz - BBz)**2)

    def loss_LR(self, left_boundary_data, right_boundary_data):
        x, y, z, nx, ny, nz = left_boundary_data
        BLx, BLy, BLz = tf.split(self.B(x, y, z), 3, axis=-1)
        BLn = BLx*nx + BLy*ny + BLz*nz
        BRx, BRy, BRz = self.B(x, y + self.domain.b, z)
        x, y, z, nx, ny, nz = right_boundary_data
        BRn = BRx*nx + BRy*ny + BRz*nz
        LR = tf.reduce_mean((BLx - BRx)**2 + (BLy - BRy)**2 + (BLz - BRz)**2) 
        Ln = tf.reduce_mean(BLn**2)
        Rn = tf.reduce_mean(BRn**2)
        return LR + Ln + Rn 

    def loss_zeroB(self, boundary_data):
        x, y, z, nx, ny, nz = boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        return tf.reduce_mean((Bx*nx + By*nx + Bz*nz)**2)

    def loss_constB(self, boundary_data):
        x, y, z, nx, ny, nz = boundary_data
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)
        return tf.reduce_mean(Bx**2 + (By - self.value)**2 + Bz**2)

    def loss_energy(self, domain_data):
        Bx, By, Bz = tf.split(self.B(*domain_data), 3, axis=-1)
        return tf.reduce_mean(Bx**2 + By**2 + Bz**2)

    def loss_divB(self, domain_data):
        return tf.reduce_mean(self.divB(*domain_data)**2)


    def total_loss_b(self, boundary_data):
        right, left, front, back, up, down  = boundary_data
        loss = self.loss_F(front) + self.loss_B(back)
        for data in [right, left, up, down]:
            loss += self.loss_nB(data)
        return loss
    

    @tf.function
    def train_step(self, optimizer, domain_data, boundary_data):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_energy(domain_data) + 1e5*self.total_loss_b(boundary_data) + 1e10 * self.loss_divB(domain_data)
        grads = tape.gradient(loss, self.B.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.B.trainable_weights))
        return loss
        

    def learn(self, optimizer, epochs, n_sample, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        domain_data = self.domain.sample(n_sample)
        boundary_data = self.domain.boundary_sample(n_sample)
        heading = "{:>6}{:>12}{:>18}".format('epoch', 'loss_v', 'runtime(s)')
        print(heading)
        start = time.time()
        with open('{}/training_log.txt'.format(save_dir), 'w') as log:
            log.write(heading + '\n')
            for epoch in range(epochs):
                l1 = self.train_step(optimizer, domain_data, boundary_data)
                if epoch % 10 == 0:
                    stdout = '{:6d}{:12.6f}{:12.4f}'.format(epoch, l1, time.time()-start)
                    print(stdout)
                    log.write(stdout + '\n')
                    domain_data = self.domain.sample(n_sample)
                    boundary_data = self.domain.boundary_sample(n_sample)
        self.B.save_weights('{}/{}'.format(save_dir, self.B.name))
        

    def plot(self, resolution, save_dir):
        self.B.load_weights('{}/{}'.format(save_dir, self.B.name)).expect_partial()
        fig = plt.figure(figsize=(16, 16))
        ax_B = fig.add_subplot(221, projection='3d')
        ax_divB = fig.add_subplot(222, projection='3d')
        ax_modB = fig.add_subplot(223, projection='3d')
        x, y, z = self.domain.grid_sample(resolution)
        grid = (resolution, resolution, resolution)
        grid2 = (resolution, resolution)
        Bx, By, Bz = tf.split(self.B(x, y, z), 3, axis=-1)


        divB = self.divB(x, y, z).numpy().flatten()
        scamap = plt.cm.ScalarMappable(cmap='inferno')
        fcolors = scamap.to_rgba(divB)
        ax_divB.scatter(x.flatten(), y.flatten(), z.flatten() , c=fcolors)
        fig.colorbar(scamap, ax=ax_divB)
        ax_divB.set_title('$\\nabla\\cdot B$', fontsize=20)
        ax_divB.grid(False)

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

    