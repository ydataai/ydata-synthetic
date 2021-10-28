from tensorflow import random
from tensorflow import reshape, shape, math, GradientTape, reduce_mean
from tensorflow import norm as tfnorm

## Original code loss from
## https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/blob/master/tf2gan/loss.py
def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:   # interpolation in DRAGAN
                beta = random.uniform(shape=shape(a), minval=0., maxval=1.)
                b = a + 0.5 * math.reduce_std(a) * beta
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

    if mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode == 'wgangp':
        gp = _gradient_penalty(f, real, fake)

    return gp
