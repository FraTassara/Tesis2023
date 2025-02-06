
from tensorflow import keras
from keras import layers
from layers import Input, Dense, Lambda, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.legacy import Adam
from utils.sampling import sample_z

class CVAEBuilder:
    def __init__(self, n_x, n_y, n_z=2):
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        
    def build_model(self, params):
        # Encoder
        input_x = Input(shape=(self.n_x,), name='encoder_input_x')
        input_label = Input(shape=(self.n_y,), name='encoder_input_y')
        inputs = concatenate([input_x, input_label])
        
        encoder_h = Dense(params['encoder_dim1'], activation=params['activation_1'])(inputs)
        encoder_h2 = Dense(params['encoder_dim2'], activation=params['activation_2'])(encoder_h)
        mu = Dense(self.n_z, activation='linear', name='latent_mu')(encoder_h2)
        l_sigma = Dense(self.n_z, activation='linear', name='latent_sigma')(encoder_h2)
        
        z = Lambda(sample_z)([mu, l_sigma])
        zc = concatenate([z, input_label])
        
        # Decoder
        decoder_hidden = Dense(params['decoder_dim'], 
                             activation=params['activation_3'],
                             kernel_regularizer=regularizers.l2(1e-7))
        decoder_out = Dense(self.n_x, activation='sigmoid', name='decoder_output')
        
        h_p = decoder_hidden(zc)
        outputs = decoder_out(h_p)
        
        # Pérdidas
        def vae_loss(y_true, y_pred):
            recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
            kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
            return recon + kl
        
        # Compilar modelo
        cvae = Model([input_x, input_label], outputs)
        cvae.compile(optimizer=Adam(learning_rate=0.001),
                   loss=vae_loss,
                   metrics=[self.kl_loss, self.recon_loss])
        
        return cvae, mu, l_sigma
    
    def kl_loss(self, y_true, y_pred):
        return 0.5 * K.sum(K.exp(self.l_sigma) + K.square(self.mu) - 1. - self.l_sigma, axis=1)
    
    def recon_loss(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)