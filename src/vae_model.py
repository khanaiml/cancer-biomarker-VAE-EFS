import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.utils.class_weight import compute_class_weight

class VAELossLayer(layers.Layer):
    def __init__(self, recon_weight, beta_kl, **kwargs):
        super().__init__(**kwargs)
        self.recon_weight = recon_weight
        self.beta_kl = beta_kl
        self.mse = keras.losses.MeanSquaredError()
        
    def call(self, inputs):
        x_true, x_pred, z_mean, z_log_var = inputs
        recon = self.mse(x_true, x_pred)
        kl = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 
            axis=1
        )
        self.add_loss(self.recon_weight * recon + self.beta_kl * tf.reduce_mean(kl))
        return x_pred

def dense_block(x, units, dropout=0.0, l2_reg=0.0, batch_norm=True):
    x = layers.Dense(
        units, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_decoder(latent_dim, decoder_units, input_dim, dropout, l2_reg):
    z_input = layers.Input(shape=(latent_dim,), name='z_input')
    x = z_input
    for units in decoder_units:
        x = dense_block(x, units, dropout=dropout, l2_reg=l2_reg)
    output = layers.Dense(input_dim, activation='linear', name='reconstruction')(x)
    return models.Model(z_input, output, name='decoder')

def build_VAE(config, input_dim, num_classes=5):
    vae_config = config['vae']
    
    encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
    x = encoder_input
    
    for units in vae_config['encoder_units']:
        x = dense_block(x, units, dropout=0.0, l2_reg=0.0, batch_norm=True)
    
    z_mean = layers.Dense(vae_config['latent_dim'], name='z_mean')(x)
    z_log_var = layers.Dense(vae_config['latent_dim'], name='z_log_var')(x)
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    encoder = models.Model(encoder_input, z, name='encoder')
    
    decoder = build_decoder(
        vae_config['latent_dim'],
        vae_config['decoder_units'],
        input_dim,
        vae_config['dropout_rate'],
        vae_config['l2_reg']
    )
    
    reconstructed = decoder(z)
    _ = VAELossLayer(vae_config['recon_weight'], vae_config['beta_kl'])(
        [encoder_input, reconstructed, z_mean, z_log_var]
    )
    
    clf_input = z
    for units in vae_config['clf_units']:
        clf_input = dense_block(clf_input, units, dropout=vae_config['clf_dropout'], batch_norm=False)
    
    clf_output = layers.Dense(num_classes, activation='softmax', name='class_output')(clf_input)
    
    model = models.Model(encoder_input, clf_output, name='vae_clf')
    model.compile(
        optimizer=keras.optimizers.Adam(vae_config['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, encoder, decoder

def train_vae(model, X_train, y_train, config):
    vae_config = config['vae']
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=vae_config['early_stopping_patience'],
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_split=config['validation_split'],
        epochs=vae_config['epochs'],
        batch_size=vae_config['batch_size'],
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0
    )
    
    return history