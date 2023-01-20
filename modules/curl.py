import tensorflow as tf 
import arch

def curl(f, x, y, z):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y, z])
        Ax, Ay, Az = tf.split(f(x, y, z), 3, axis=-1)
    Ax_y = tape.gradient(Ax, y)
    Ay_x = tape.gradient(Ay, x)
    Ax_z = tape.gradient(Ax, z)
    Az_x = tape.gradient(Az, x)
    Ay_z = tape.gradient(Ay, z)
    Az_y = tape.gradient(Az, y)
    return tf.concat([(Az_y - Ay_z), (Ax_z - Az_x), (Ay_x - Ax_y)], axis=-1) 


# test code
# g = lambda x, y, z: tf.concat([tf.sin(y), tf.cos(z), tf.tan(x)], axis=-1)
# g = arch.LSTMForgetNet(num_nodes=50, num_layers=3, out_dim=3)
# x, y, z = tf.ones((10, 1)), tf.ones((10, 1)), tf.ones((10, 1))
# print(g(x, y, z))
# print(curl(g, x, y, z))
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 2000, 10000], [5e-3, 1e-3, 5e-4, 1e-4])
# optimizer = tf.keras.optimizers.Adam(learning_rate)
# for i in range(100):
#     with tf.GradientTape() as tape:
#         tape.watch([x, y, z])
#         loss = tf.reduce_mean(curl(g, x, y, z)**2)
#     gradsx = tape.gradient(loss, g.trainable_weights)
#     optimizer.apply_gradients(zip(gradsx, g.trainable_weights))
#     print(g(x, y, z))

# print(g.summary())