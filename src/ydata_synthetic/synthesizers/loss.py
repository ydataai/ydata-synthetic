import tensorflow as tf

from enum import Enum

class Mode(Enum):
    WGANGP = 'wgangp'
    DRAGAN = 'dragan'
    CRAMER = 'cramer'
    CTGAN = 'ctgan'

## Original code loss from
## https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/blob/master/tf2gan/loss.py
def gradient_penalty(f, real, fake, mode, pac=None):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math.reduce_std(a) * beta
            shape_ = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape_, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

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
        alpha = tf.random.uniform([real.shape[0] // pac, 1, 1], 0., 1.)
        alpha = tf.tile(alpha, tf.constant([1, pac, real.shape[1]], tf.int32))
        alpha = tf.reshape(alpha, [-1, real.shape[1]])
        interpolate = alpha * real + ((1 - alpha) * fake)
        with tf.GradientTape() as tape:
            tape.watch(interpolate)
            prediction = f(interpolate)
        gradient = tape.gradient(prediction, [interpolate])[0]
        gradient = tf.reshape(gradient, tf.constant([-1, pac * real.shape[1]], tf.int32))
        slope = tf.math.reduce_euclidean_norm(gradient, axis=1)
        return tf.reduce_mean((slope - 1.) ** 2)

    if mode == Mode.DRAGAN:
        gp = _gradient_penalty(f, real)
    elif mode == Mode.CRAMER:
        gp = _gradient_penalty_cramer(f, real, fake)
    elif mode == Mode.WGANGP:
        gp = _gradient_penalty(f, real, fake)
    elif mode == Mode.CTGAN:
        if pac is not None:
            gp = _gradient_penalty_ctgan(f, real, fake, pac=pac)
        else:
            gp = _gradient_penalty_ctgan(f, real, fake)

    return gp
