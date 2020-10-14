from src.ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from src.ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from src.ydata_synthetic.synthesizers.regular.vanillagan.model import VanilllaGAN

__all__ = [
    "VanilllaGAN",
    "CGAN",
    "WGAN"
]