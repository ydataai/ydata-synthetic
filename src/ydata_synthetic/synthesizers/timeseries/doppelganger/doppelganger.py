import tensorflow as tf
import numpy as np
from typing import Optional, Any, List, Tuple
from tqdm import Tqdm
import numbers

class DoppelGANgerNetwork:
    """
    Adapted from https://github.com/fjxmlzn/DoppelGANger/blob/master/gan/doppelganger.py.
    """

    def __init__(
        self,
        sess: tf.Session,
        epoch: int,
        batch_size: int,
        data_feature: Optional[np.ndarray] = None,
        data_attribute: Optional[np.ndarray] = None,
        attribute_cols_metadata: List[Any],
        sample_len: int,
        generator: Any,
        discriminator: Any,
        rounds: int,
        d_gp_coe: float,
        num_packing: int,
        attr_discriminator: Optional[Any] = None,
        attr_d_gp_coe: Optional[float] = None,
        g_attr_d_coe: float = 1.0,
        attribute_latent_dim: int = 5,
        feature_latent_dim: int = 5,
        fix_feature_network: bool = False,
        g_lr: float = 0.001,
        g_beta1: float = 0.5,
        d_lr: float = 0.001,
        d_beta1: float = 0.5,
        attr_d_lr: float = 0.001,
        attr_d_beta1: float = 0.5,
    ):
        """Constructor of DoppelGANger

        Args:
            sess: A tensorflow session
            epoch: Number of training epochs
            batch_size: Training batch size
            data_feature: Training features, in numpy float32 array format.
                The size is [(number of training samples) x (maximum length) x
                (total dimension of features)].
            data_attribute: Training attributes, in numpy float32 array format.
                The size is [(number of training samples) x (total dimension 
                of attributes)]
            sample_len: The time series batch size
            generator: An instance of network.DoppelGANgerGenerator
            discriminator: An instance of network.Discriminator
            rounds: Number of steps per batch
            d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
            num_packing: Packing degree in PacGAN (a method for solving mode
                collapse in NeurIPS 2018, see https://arxiv.org/abs/1712.04086)
            attr_discriminator: An instance of network.AttrDiscriminator. None
                if you do not want to use this auxiliary discriminator
            attr_d_gp_coe: Weight of gradient penalty loss in Wasserstein GAN
                for the auxiliary discriminator
            g_attr_d_coe: Weight of the auxiliary discriminator in the
                generator's loss
            attribute_latent_dim: The dimension of noise for generating 
                attributes
            feature_latent_dim: The dimension of noise for generating 
                features
            fix_feature_network: Whether to fix the feature network during 
                training
            g_lr: The learning rate in Adam for training the generator
            g_beta1: The beta1 in Adam for training the generator 
            d_lr: The learning rate in Adam for training the discriminator
            d_beta1: The beta1 in Adam for training the discriminator 
            attr_d_lr: The learning rate in Adam for training the auxiliary
                discriminator
            attr_d_beta1: The beta1 in Adam for training the auxiliary
                discriminator
        """
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_feature = data_feature
        self.data_attribute = data_attribute
        self.attribute_cols_metadata = attribute_cols_metadata
        self.sample_len = sample_len
