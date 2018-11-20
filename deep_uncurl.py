import keras
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K

# source: keras examples
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
    args (tensor): mean and log of variance of Q(z|X)

    # Returns:
    z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_encoder_model(input_dim, k, depth=3):
    """
    """
    input_layer = Input(shape=(input_dim,))


def create_decoder_model(k, output_dim, depth=3,
        variational=True):
    """
    Creates a decoder model...
    """

def create_encoder_decoder_model(input_dims, k,
        encoder_depth=3,
        decoder_depth=3):
    """
    """


class DeepUncurl(object):

    def __init__(self):
        pass


if __name__ == '__main__':
    pass
