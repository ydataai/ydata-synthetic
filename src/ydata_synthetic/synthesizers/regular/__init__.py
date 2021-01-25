from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP

__all__ = [
    "VanilllaGAN",
    "CGAN",
    "WGAN",
    "WGAN_GP"
]
