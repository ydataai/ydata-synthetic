from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN

__all__ = [
    "VanilllaGAN",
    "CGAN",
    "WGAN",
    "WGAN_GP",
    "DRAGAN",
    "CRAMERGAN"
]
