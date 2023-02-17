from tensorflow import \
    (random, reshape, shape, GradientTape, reduce_mean, 
     norm as tfnorm, tile, constant, int32)
from tensorflow.math import reduce_std, reduce_euclidean_norm
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
                beta = random.uniform(shape=shape(a), minval=0., maxval=1.)
                b = a + 0.5 * reduce_std(a) * beta
            shape_ = [shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = random.uniform(shape=shape_, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter

        x = _interpolate(real, fake)
        with GradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tfnorm(reshape(grad, [shape(grad)[0], -1]), axis=1)
        gp = reduce_mean((norm - 1.)**2)
        return gp

    def _gradient_penalty_cramer(f_crit, real, fake):
        epsilon = random.uniform([real.shape[0], 1], 0.0, 1.0)
        x_hat = epsilon * real + (1 - epsilon) * fake[0]
        with GradientTape() as t:
            t.watch(x_hat)
            f_x_hat = f_crit(x_hat, fake[1])
        gradients = t.gradient(f_x_hat, x_hat)
        c_dx = tfnorm(reshape(gradients, [shape(gradients)[0], -1]), axis=1)
        c_regularizer = (c_dx - 1.0) ** 2
        return c_regularizer

    def _gradient_penalty_ctgan(f, real, fake, pac=10):
        alpha = random.uniform([real.shape[0] // pac, 1, 1], 0., 1.)
        alpha = tile(alpha, constant([1, pac, real.shape[1]], int32))
        alpha = reshape(alpha, [-1, real.shape[1]])
        interpolate = alpha * real + ((1 - alpha) * fake)
        with GradientTape() as tape:
            tape.watch(interpolate)
            prediction = f(interpolate)
        gradient = tape.gradient(prediction, [interpolate])[0]
        gradient = reshape(gradient, constant([-1, pac * real.shape[1]], int32))
        slope = reduce_euclidean_norm(gradient, axis=1)
        return reduce_mean((slope - 1.) ** 2)

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
