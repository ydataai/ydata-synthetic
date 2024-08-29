"""
    ydata_synthetic.synthesizers init file
"""
from warnings import warn

from ydata_synthetic.synthesizers.regular.ctgan.model import CTGAN
from ydata_synthetic.synthesizers.regular.cramergan.model import CRAMERGAN
from ydata_synthetic.synthesizers.regular.vanillagan.model import VanillaGAN
from ydata_synthetic.synthesizers.regular.gmm.model import GMM
from ydata_synthetic.synthesizers.regular.wgan.model import WGAN
from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP
from ydata_synthetic.synthesizers.regular.cwgangp.model import CWGANGP
from ydata_synthetic.synthesizers.regular.cgan.model import CGAN
from ydata_synthetic.synthesizers.regular.dragan.model import DRAGAN
from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN
from ydata_synthetic.synthesizers.timeseries.doppelganger.model import DoppelGANgerNetwork

warn(
    "`import ydata_synthetic.synthesizers` is deprecated. Please use `import ydata.sdk.synthesizers` instead."
    "For more information check https://docs.synthetic.ydata.ai/latest and https://docs.fabric.ydata.ai/latest/sdk",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ['CTGAN',
           'CRAMERGAN',
           'VanillaGAN',
           'WGAN',
           'WGAN_GP',
           'CWGANGP',
           'DRAGAN',
           'CGAN',
           'GMM',
           'TimeGAN',
           'DoppelGANgerNetwork'
           ]