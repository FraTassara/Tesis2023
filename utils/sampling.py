from tensorflow.keras import backend as K

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], K.shape(mu)[1]), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps