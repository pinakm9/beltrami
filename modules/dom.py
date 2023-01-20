from locale import normalize
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

class Disk:
    def __init__(self, center=np.zeros(2), radius=1.0):
        self.center = center
        self.radius = radius
        self.n_bdry_comps = 1 

    def sample(self, n_sample):
        r = tf.sqrt(tf.random.uniform(minval=0., maxval=self.radius**2, shape=(n_sample, 1)))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta) 

    def grid_sample(self, resolution):
        r = np.sqrt(np.linspace(start=0., stop=self.radius**2, num=resolution, endpoint=True, dtype='float32'))
        theta = np.linspace(start=0., stop=2.0*np.pi, num=resolution, endpoint=True, dtype='float32')
        r, theta = np.meshgrid(r, theta)
        r, theta = r.reshape(-1, 1), theta.reshape(-1, 1)
        return self.center[0] + r*np.cos(theta), self.center[1] + r*np.sin(theta) 


    def boundary_sample(self, n_sample):
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        c, s = tf.cos(theta), tf.sin(theta)
        x, y = self.center[0] + self.radius*c, self.center[1] + self.radius*s
        return [[x, y, c, s]]

class Annulus:
    def __init__(self, center=np.zeros(2), in_radius=0.5, out_radius=1.0):
        self.center = center
        self.in_radius = in_radius
        self.out_radius = out_radius
        self.n_bdry_comps = 2

    def sample(self, n_sample):
        r = tf.sqrt(tf.random.uniform(minval=self.in_radius, maxval=self.out_radius**2, shape=(n_sample, 1)))
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        return self.center[0] + r*tf.cos(theta), self.center[1] + r*tf.sin(theta)

    def grid_sample(self, resolution):
        r = np.sqrt(np.linspace(start=self.in_radius, stop=self.out_radius**2, num=resolution, endpoint=True, dtype='float32'))
        theta = np.linspace(start=0., stop=2.0*np.pi, num=resolution, endpoint=True, dtype='float32')
        r, theta = np.meshgrid(r, theta)
        r, theta = r.reshape(-1, 1), theta.reshape(-1, 1)
        return self.center[0] + r*np.cos(theta), self.center[1] + r*np.sin(theta)

    def boundary_sample(self, n_sample):
        theta = tf.random.uniform(minval=0., maxval=2.0*np.pi, shape=(n_sample, 1))
        c, s = tf.cos(theta), tf.sin(theta)
        x, y = self.center[0] + self.in_radius*c, self.center[1] + self.in_radius*s
        comp1 = [x, y, -c, -s]
        x, y = self.center[0] + self.out_radius*c, self.center[1] + self.out_radius*s
        comp2 = [x, y, c, s]
        return [comp1, comp2]



class Box2D:
    def __init__(self, center=np.zeros(2), width=1., height=1., dtype='float64'):
        self.center = center 
        self.width = width 
        self.height = height
        self.n_bdry_comps = 4
        self.dtype = dtype

    def sample(self, n_sample):
        x = tf.random.uniform(minval=-self.width/2., maxval=self.width/2., shape=(n_sample, 1), dtype=self.dtype)
        y = tf.random.uniform(minval=-self.height/2., maxval=self.height/2., shape=(n_sample, 1), dtype=self.dtype)
        return self.center[0] + x, self.center[1] + y

    def grid_sample(self, resolution):
        x = np.linspace(start=-self.width/2., stop=self.width/2., num=resolution, endpoint=True, dtype=self.dtype)
        y = np.linspace(start=-self.height/2., stop=self.height/2., num=resolution, endpoint=True, dtype=self.dtype)
        x, y = np.meshgrid(x, y)
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        return self.center[0] + x, self.center[1] + y

    def boundary_sample(self, n_sample):
        a, b = self.center
        w, h = self.width, self.height
        # bottom edge
        x =  tf.random.uniform(minval=a-w/2., maxval=a+w/2., shape=(n_sample, 1), dtype=self.dtype)
        comp1 = [x, (b-h/2.)*tf.ones_like(x), 0.*tf.ones_like(x), -1.*tf.ones_like(x)]
        # top edge
        comp3 = [x, (b+h/2.)*tf.ones_like(x), 0.*tf.ones_like(x), 1.*tf.ones_like(x)]
        # left edge
        y =  tf.random.uniform(minval=b-h/2., maxval=b+h/2., shape=(n_sample, 1), dtype=self.dtype)
        comp4 = [(a-w/2.)*tf.ones_like(y), y, -1.*tf.ones_like(y), 0.*tf.ones_like(x)]
        # right edge
        comp2 = [(a+w/2.)*tf.ones_like(y), y, 1.*tf.ones_like(y), 0.*tf.ones_like(x)]
        return [comp1, comp2, comp3, comp4]





class Box3D:
    def __init__(self, center=np.zeros(3), a=1., b=1., c=1., dtype='float64'):
        self.a = a
        self.b = b 
        self.c = c
        self.center = center
        self.n_bdry_comps = 6
        self.dtype = dtype

    def sample(self, n_sample):
        x = tf.random.uniform(minval=-self.a/2., maxval=self.a/2., shape=(n_sample, 1), dtype=self.dtype)
        y = tf.random.uniform(minval=-self.b/2., maxval=self.b/2., shape=(n_sample, 1), dtype=self.dtype)
        z = tf.random.uniform(minval=-self.c/2., maxval=self.c/2., shape=(n_sample, 1), dtype=self.dtype)
        return self.center[0] + x, self.center[1] + y, self.center[2] + z 

    def grid_sample(self, resolution):
        x = np.linspace(start=-self.a/2., stop=self.a/2., num=resolution, endpoint=True, dtype=self.dtype)
        y = np.linspace(start=-self.b/2., stop=self.b/2., num=resolution, endpoint=True, dtype=self.dtype)
        z = np.linspace(start=-self.c/2., stop=self.c/2., num=resolution, endpoint=True, dtype=self.dtype)
        x, y, z = np.meshgrid(x, y, z)
        x, y, z = x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1) 
        return self.center[0] + x, self.center[1] + y, self.center[2] + z

    def boundary_sample(self, n_sample):
        # x-wall front
        wall = Box2D(center=[self.center[1], self.center[2]], width=self.b, height=self.c, dtype=self.dtype)
        y, z = wall.sample(n_sample)
        x = self.center[0] + (self.a/2.)*tf.ones_like(y)
        comp1 = [x, y, z, tf.ones_like(x), tf.zeros_like(x), tf.zeros_like(x)]

        # x-wall back
        wall = Box2D(center=[self.center[1], self.center[2]], width=self.b, height=self.c, dtype=self.dtype)
        y, z = wall.sample(n_sample)
        x = self.center[0] - (self.a/2.)*tf.ones_like(y)
        comp2 = [x, y, z, -tf.ones_like(x), tf.zeros_like(x), tf.zeros_like(x)]
        
        # y-wall front
        wall = Box2D(center=[self.center[0], self.center[2]], width=self.a, height=self.c, dtype=self.dtype)
        x, z = wall.sample(n_sample)
        y = self.center[1] + (self.b/2.)*tf.ones_like(x)
        comp3 = [x, y, z, tf.zeros_like(x), tf.ones_like(x), tf.zeros_like(x)]

        # y-wall back
        wall = Box2D(center=[self.center[0], self.center[2]], width=self.a, height=self.c, dtype=self.dtype)
        x, z = wall.sample(n_sample)
        y = self.center[1] - (self.b/2.)*tf.ones_like(x)
        comp4 = [x, y, z, tf.zeros_like(x), -tf.ones_like(x), tf.zeros_like(x)]
        
        # z-wall front
        wall = Box2D(center=[self.center[0], self.center[1]], width=self.a, height=self.b, dtype=self.dtype)
        x, y = wall.sample(n_sample)
        z = self.center[0] + (self.c/2.)*tf.ones_like(y)
        comp5 = [x, y, z, tf.zeros_like(x), tf.zeros_like(x), tf.ones_like(x)]

        # z-wall back
        wall = Box2D(center=[self.center[0], self.center[1]], width=self.a, height=self.b, dtype=self.dtype)
        x, y = wall.sample(n_sample)
        z = self.center[0] - (self.c/2.)*tf.ones_like(y)
        comp6 = [x, y, z, tf.zeros_like(x), tf.zeros_like(x), -tf.ones_like(x)]

        return [comp1, comp2, comp3, comp4, comp5, comp6]

    def plot_boundary(self):
        sample = self.boundary_sample(100)
        fig = plt.figure(figsize=(8, 32))
        ax_F = fig.add_subplot(321, projection='3d')
        ax_B = fig.add_subplot(322, projection='3d')
        ax_L = fig.add_subplot(323, projection='3d')
        ax_R = fig.add_subplot(324, projection='3d')
        ax_D = fig.add_subplot(325, projection='3d')
        ax_U = fig.add_subplot(326, projection='3d')

        sides = ['right', 'left', 'front', 'back', 'up', 'down']
        axes = [ax_R, ax_L, ax_F, ax_B, ax_U, ax_D]
        for i in range(6):
            x, y, z, nx, ny, nz = sample[i]
            x, y, z = x.numpy().flatten(), y.numpy().flatten(), z.numpy().flatten()
            nx, ny, nz = nx.numpy().flatten(), ny.numpy().flatten(), nz.numpy().flatten()
            axes[i].scatter(x, y, z, c='orange')
            axes[i].quiver(x, y, z, nx, ny, nz, pivot = 'tail', length = 0.02)
            axes[i].set_title(sides[i])
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_zlabel('z')
        # fig.tight_layout()
        plt.show()