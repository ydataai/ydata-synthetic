from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN
from ydata_synthetic.synthesizers.regular.cwgangp.model import CWGANGP
from ydata_synthetic.synthesizers.regular.pategan.model import PATEGAN

__all__ = [
    "VanilllaGAN",
    "CGAN",
    "WGAN",
    "WGAN_GP",
    "DRAGAN",
    "CRAMERGAN",
    "CWGANGP",
    "PATEGAN"
]
