from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
import pandas as pd
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class Clustering(Layer):
    """
    Converts input sample into soft clusters assignments
    Distribution is given by t-distribution, same as t-sne
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)

        self.n_clusters = n_clusters
        self.init_weights = weights
        self.alpha = alpha
        self.beta = beta

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        if self.init_weights is not None:
            self.set_weights(self.init_weights)
            del self.init_weights

            self.built = True
        else:
            self.clusters = self.add_weight(name='clusters',
                                            shape=(self.n_clusters, input_dim),
                                            initializer='random_normal')

            super().build(input_shape)
            
        

    def _pairwise_cluster_distance_regularizer(self):

        W = self.clusters

        pairwise_distance = K.square(K.expand_dims(W, axis=1) - W)

        inverse_distance = K.sum(1/(1+K.maximum(1 - pairwise_distance, 0))) / 2
        
        self.add_loss(self.beta * inverse_distance)

    def call(self, x):
        """
        This is where the t-distribution will be calculated

        :param x:
        :return:
        """

        p = 1.0 / (1.0 + K.sum(K.pow(K.expand_dims(x, axis=1) - self.clusters, 2), axis=-1) / self.alpha)
        p = p ** ((self.alpha + 1) / 2)

        p = p / K.sum(p, axis=-1, keepdims=True)
        
        self._pairwise_cluster_distance_regularizer()

        return p

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_clusters)


class DeepEmbeddingClustering(object):
    def __init__(self,
                 autoencoder_dimensions,
                 n_clusters,
                 alpha=1.0,
                 beta=1.0,
                 activations='relu'):

        self.a_dim = autoencoder_dimensions
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta
        self.activations = activations
        self.autoencoder, self.encoder, self.decoder = self._build_autoencoder()

        clustering_layer = Clustering(n_clusters=self.n_clusters,
                                      alpha=self.alpha,
                                      beta=self.beta,
                                      name='clustering')(self.encoder.output)
        
        self.clustering_layer = clustering_layer


        self.model = Model(inputs=self.encoder.input,
                           outputs=clustering_layer)

        self._pretrained = False

    def _build_autoencoder(self):

        h = Input(shape=(self.a_dim[0],))

        x = h

        n_encoder_layers = len(self.a_dim) - 1

        # encoder
        for i in range(n_encoder_layers):
            h = Dense(self.a_dim[i], kernel_initializer='glorot_normal')(h)
            h = Activation(self.activations)(h)
            
        # latent space
        h = Dense(self.a_dim[-1], kernel_initializer='glorot_normal')(h)        
        en = h

        # decoder
        for i in range(n_encoder_layers)[::-1]:
            h = Dense(self.a_dim[i], kernel_initializer='glorot_normal')(h)
            h = Activation(self.activations)(h)

        y = h

        ae_model = Model(inputs=x, outputs=y, name='Autoencoder')
        e_model = Model(inputs=x, outputs=en, name='Encoder')
#         d_model = Model(inputs=en, outputs=y, name='Decoder')
        d_model = None

        return ae_model, e_model, d_model

    def pretrain_autoencoder(self,
                             X,
                             y=None,
                             batch_size=32,
                             epochs=10):
        """
        This executes pretraining of the autoencoder,
        once the pretraining is done, we also initialize the clusters by K-means
        """

        self.autoencoder.compile(loss='mse', optimizer='adam')
        if y is None:
            self.autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs)
        else:
            self.autoencoder.fit(X, y, batch_size=batch_size, epochs=epochs)

        return self

    def _init_clusters(self, X):

        encoded = self.encoder.predict(X)

        kmeans = KMeans(self.n_clusters)
        kmeans.fit(encoded)

        self.clusters = kmeans.cluster_centers_

        self.model.layers[-1].set_weights([self.clusters])

        return self

    def encoder_features(self, X):
        return self.encoder.predict(X)

    def auxillary_distribution(self, q):
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)

        return p
      
    def compile(self, optimizer, loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, X, epochs=10, batch_size=32, **kwargs):

        pretrain_data = kwargs.get('pretrain_data')
        pretrain_epochs = kwargs.get('pretrain_epochs')
        pretrain_batch = kwargs.get('pretrain_batch')
        

        if self._pretrained == False:
            self.pretrain_autoencoder(pretrain_data, 
                                      epochs=pretrain_epochs, 
                                      batch_size=pretrain_batch)
            self._init_clusters(X)

            self._pretrained =True

        q = self.model.predict(X)
        p = self.auxillary_distribution(q)

        self.model.fit(X, p, epochs=epochs, batch_size=batch_size)
        
        
class ImprovedDeepEmbeddingClustering(DeepEmbeddingClustering):
    
    def __init__(self,
                 autoencoder_dimensions,
                 n_clusters,
                 alpha=1.0,
                 beta=1.0,
                 activations='relu'):

        """
        The difference is that clustering and autoencoder training 
        is done simultaneously instead of seperately
        """
        
        super().__init__(autoencoder_dimensions,
                         n_clusters,
                         alpha=1.0,
                         beta=1.0,
                         activations='relu')
        
        self.model = Model(inputs=self.encoder.input,
                           outputs=[self.autoencoder.output, self.clustering_layer])
        self.compile('adam')
        
    def compile(self, optimizer, loss=['mse', 'kld']):
        
        assert len(loss)==2, "Need to provide two losses for both outputs"
        
        self.model.compile(optimizer=optimizer, loss=loss)
        
        
    def fit(self, X, epochs=10, batch_size=32, **kwargs):
        """
        pretrain data should be corrupted version of train data X
        a denoising autoencoder is more suitable for clustering task
        """

        pretrain_data = kwargs.get('pretrain_data')
        pretrain_epochs = kwargs.get('pretrain_epochs')
        pretrain_batch = kwargs.get('pretrain_batch')
        
        if self._pretrained == False:
            self.pretrain_autoencoder(pretrain_data,
                                      X,
                                      epochs=pretrain_epochs, 
                                      batch_size=pretrain_batch)
            self._init_clusters(X)

            self._pretrained =True

        q = self.model.predict(X)[1]
        p = self.auxillary_distribution(q)

        self.model.fit(X, [X, p], epochs=epochs, batch_size=batch_size)


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

X_train = X_train/255
X_test = X_test/255

pretrain_X = X_train * np.random.randint(0, 2, size=X_train.shape)

dims = [784, 512, 512, 256, 128]
n_clusters = 10

pretraining_params = {'pretrain_data':pretrain_X,
                     'pretrain_epochs':40,
                     'pretrain_batch':2048}

dec = ImprovedDeepEmbeddingClustering(autoencoder_dimensions=dims, n_clusters=n_clusters, alpha=1, beta=0.1)
dec.fit(X_train, 20, 1028, verbose=1, **pretraining_params, validation_split)

images = dec.autoencoder.predict(X_test)
pred = dec.encoder.predict(X_test)
