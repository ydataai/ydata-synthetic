import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model

class Mode(Enum):
    WGANGP = 'wgangp'
    DRAGAN = 'dragan'
    CRAMER = 'cramer'
    CTGAN = 'ctgan'

def gradient_penalty(f, real, fake, mode: Mode, pac=None):
    """
    Compute the gradient penalty for a given discriminator.

    Args:
        f: A function that takes a tensor as input and outputs a tensor.
        real: A tensor representing real data.
        fake: A tensor representing fake data.
        mode: The mode of gradient penalty to compute.
        pac: An integer specifying the number of partitions for CTGAN.

    Returns:
        A tensor representing the gradient penalty.
    """
    def _interpolate(a, b=None):
        if b is None:
            beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
            b = a + 0.5 * tf.math.reduce_std(a) * beta
        shape_ = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape_, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    def _gradient_penalty(f, real, fake=None):
        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp

    def _gradient_penalty_cramer(f_crit, real, fake):
        epsilon = tf.random.uniform([real.shape[0], 1], 0.0, 1.0)
        x_hat = epsilon * real + (1 - epsilon) * fake[0]
        with tf.GradientTape() as t:
            t.watch(x_hat)
            f_x_hat = f_crit(x_hat, fake[1])
        gradients = t.gradient(f_x_hat, x_hat)
        c_dx = tf.norm(tf.reshape(gradients, [tf.shape(gradients)[0], -1]), axis=1)
        c_regularizer = (c_dx - 1.0) ** 2
        return c_regularizer

    def _gradient_penalty_ctgan(f, real, fake, pac=10):
        if pac is None:
            raise ValueError("For CTGAN mode, pac argument must be provided.")
        alpha = tf.random.uniform([real.shape[0] // pac, 1, 1], 0., 1.)
        alpha = tf.tile(alpha, constant([1, pac, real.shape[1]], tf.int32))
        alpha = tf.reshape(alpha, [-1, real.shape[1]])
        interpolate = alpha * real + ((1 - alpha) * fake)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            prediction = f(interpolate)
        gradient = tape.gradient(prediction, [interpolate])[0]
        gradient = tf.reshape(gradient, constant([-1, pac * real.shape[1]], tf.int32))
        slope = tf.reduce_euclidean_norm(gradient, axis=1)
        return tf.reduce_mean((slope - 1.) ** 2)

    if mode == Mode.DRAGAN:
        gp = _gradient_penalty(f, real)
    elif mode == Mode.CRAMER:
        gp = _gradient_penalty_cramer(f, real, fake)
    elif mode == Mode.WGANGP:
        gp = _gradient_penalty(f, real, fake)
    elif mode == Mode.CTGAN:
        gp = _gradient_penalty_ctgan(f, real, fake, pac=pac)

    return gp

# Example usage
# Define a discriminator model
input_layer = Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Flatten()(x)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
discriminator = Model(input_layer, output_layer)

# Compute the gradient penalty
gp = gradient_penalty(discriminator, real_images, fake_images, mode=Mode.WGANGP)
