# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:24:55 2020

@author: Octavian
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy 
import encoder
import decoder

class ANVAE(tf.keras.Model):
    def __init__(self, latent_spaces, batch_size):
        super(ANVAE, self).__init__()
        
        self.batch_size = batch_size
        self.latent_spaces = 3
        self.level_sizes = [1, 1, 1]
        self.input_s = [32, 32, 1]
        self.latent_channels = 20
        
        self.encoder = encoder.Encoder(self.latent_spaces, self.input_s)
        self.decoder = decoder.Decoder(self.encoder(tf.zeros([self.batch_size, 32, 32, 1]), False), latent_channels=self.latent_channels, level_sizes=self.level_sizes)
        inputs, disc_l, outputs = self.disc_function()
        self.discriminator = tf.keras.Model(inputs=[inputs], outputs=[outputs, disc_l])
        
        self.lr_ae = .0001
        self.lr_disc = .0001
        self.recon_loss_div = 1
        self.latent_loss_div = 1
        self.sig_mult = 10
        
        self.enc_optimizer = tf.keras.optimizers.Adam(self.lr_ae, 0.5, epsilon=.1, clipvalue=0.5, clipnorm=1)
        self.dec_optimizer = tf.keras.optimizers.Adam(self.lr_ae, 0.5, clipvalue=0.5, clipnorm=1)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.lr_disc, 0.5, clipvalue=0.5, clipnorm=1)
        
        self.lastEncVars = []
        self.lastDecVars = []
        self.lastDiscVars = []
        
        self.debugCount = 0
        
        
    def call(self, image_batch):
        features = self.encoder(image_batch, False)
        image, kl_losses = self.decoder(features)
        return features, image, kl_losses
    
    def encode(self, x, debug):
        mus = []
        sigmas = []
        features = self.encoder(x, debug)
        
        for z in features:
            flatten = tf.keras.layers.Flatten()(z)
            dim = flatten.shape[-1]
            dense = tf.keras.layers.Dense(dim*2)(flatten)
            mu, sigma = tf.split(dense, 2, 1)
            mus.append(tf.reshape(mu, z.shape))
            sigmas.append(tf.reshape(sigma, z.shape))
        
        return mus, sigmas, features
    
    def decode(self, zs):
        return self.decoder(zs)
    
    def dist_encode(self, x, debug):
        mus, sigmas, features = self.encode(x, debug)
        dists = []
        for mu, sigma in zip(mus, sigmas):
            dists.append(tfp.distributions.MultivariateNormalDiag(loc = mu, scale_diag = sigma))
            
        return dists, mus, sigmas, features
    
    def discriminate(self, x):
        return self.discriminator(x)

    def reconstruct(self, x):
        mean, _, _ = self.encode(x, False)
        return self.decode(mean)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    # @tf.function
    def compute_loss(self, x, debug):
        
        # pass through network
        q_zs, tempMus, tempSigmas, features = self.dist_encode(x, debug)
        
        zs = []
        for q_z in q_zs:
            zs.append(q_z.sample())
        
        p_zs = []    
        for z in zs:
            p_zs.append(tfp.distributions.MultivariateNormalDiag(
                loc=[0.0] * z.shape[-1], scale_diag=[1.0] * z.shape[-1]
            ))   
        
        xg = self.decode(zs)
        
        z_samps = []
        for z in zs:
            z_samps.append(tf.random.normal(z.shape))
        
        xg_samp = self.decode(z_samps)  
        d_xg, ld_xg = self.discriminate(xg)
        d_x, ld_x = self.discriminate(x)  
        d_xg_samp, ld_xg_samp = self.discriminate(xg_samp)
        

        # GAN losses
        disc_real_loss = self.gan_loss(logits=d_x, is_real=True) 
        disc_fake_loss = self.gan_loss(logits=d_xg_samp, is_real=False)  
        gen_fake_loss = self.gan_loss(logits=d_xg_samp, is_real=True)
        
        discrim_layer_recon_loss = (
            tf.reduce_mean(tf.reduce_mean(tf.math.square(ld_x - ld_xg), axis=0))
            / self.recon_loss_div
        )
        
        self.D_prop = self.sigmoid(
            disc_fake_loss - gen_fake_loss, shift=0.0, mult=self.sig_mult
        )
        
        kl_divs = []
        for q_z, p_z in zip(q_zs, p_zs):
            kl_divs.append(tfp.distributions.kl_divergence(q_z, p_z))
  
        latent_losses = []
        for kl_div in kl_divs:
            latent_losses.append(tf.reduce_mean(tf.maximum(kl_div, 10**-6)) / self.latent_loss_div)
        
        
        
        if latent_losses[0] == 0.0 and self.debugCount == 0:
            self.debugCount = 1
            debug = True
        
        if debug:
            print("Debugging")
            print("x")
            print(x)
            print("x_max")
            print(x.max())
            print("TrainableVariables")
            print("Encoder")
            print(self.encoder.trainable_variables)
            print("Decoder")
            print(self.decoder.trainable_variables)
            print("Discriminator")
            print(self.discriminator.trainable_variables)
            print("PreviousTrainableVariables")
            print("Encoder")
            print(self.lastEncVars)
            print("Decoder")
            print(self.lastDecVars)
            print("Discriminator")
            print(self.lastDiscVars)
            print("encoderFeatures")
            print(features)
            print("tempMus")
            print(tempMus)
            print("tempSigmas")
            print(tempSigmas)
            print("q_zs")
            print(q_zs)
            print("zs")
            print(zs) 
            print("p_zs")
            print(p_zs)
            print("xg")
            print(xg)
            print("z_samps")
            print(z_samps)
            print("xg_samp")
            print(xg_samp)
            print("d_xg")
            print(d_xg)
            print("ld_xg")
            print(ld_xg)
            print("d_x")
            print(d_x)
            print("ld_x")
            print(ld_x)
            print("d_xg_samp")
            print(d_xg_samp)
            print("ld_xg_samp")
            print(ld_xg_samp)
            print("disc_real_loss")
            print(disc_real_loss)
            print("disc_fake_loss")
            print(disc_fake_loss)
            print("gen_fake_loss")
            print(gen_fake_loss)
            print("discrim_layer_recon_loss")
            print(discrim_layer_recon_loss)
            print("D_prop")
            print(self.D_prop)
            print("kl_divs")
            print(kl_divs)
            print("latent_losses")
            print(latent_losses)
        
        self.lastEncVars = self.encoder.trainable_variables
        self.lastDecVars = self.decoder.trainable_variables
        self.lastDiscVars = self.discriminator.trainable_variables
        
        return (
            self.D_prop,
            latent_losses,
            discrim_layer_recon_loss,
            gen_fake_loss,
            disc_fake_loss,
            disc_real_loss,
        )

    # @tf.function
    def compute_gradients(self, x):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            (
                _,
                latent_losses,
                discrim_layer_recon_loss,
                gen_fake_loss,
                disc_fake_loss,
                disc_real_loss,
            ) = self.compute_loss(x, False)

            enc_loss = sum(latent_losses) + discrim_layer_recon_loss
            dec_loss = gen_fake_loss + discrim_layer_recon_loss
            disc_loss = disc_fake_loss + disc_real_loss

        enc_gradients = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
        dec_gradients = dec_tape.gradient(dec_loss, self.decoder.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        return enc_gradients, dec_gradients, disc_gradients

    @tf.function
    def apply_gradients(self, enc_gradients, dec_gradients, disc_gradients):
        self.enc_optimizer.apply_gradients(
            zip(enc_gradients, self.encoder.trainable_variables)
        )
        self.dec_optimizer.apply_gradients(
            zip(dec_gradients, self.decoder.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )

    def train(self, x):
        enc_gradients, dec_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(enc_gradients, dec_gradients, disc_gradients)
        
    def gan_loss(self, logits, is_real=True):
        
        if is_real:
            labels = tf.ones_like(logits)
        else:
            labels = tf.zeros_like(logits)
    
        return tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=labels, logits=logits
        )


    def sigmoid(self, x, shift=0.0, mult=20):
        """ squashes a value with a sigmoid
        """
        return tf.constant(1.0) / (
            tf.constant(1.0) + tf.exp(-tf.constant(1.0) * (x * mult))
        )
    
    def disc_function(self):
        inputs = tf.keras.layers.Input(shape=(32, 32, 1))
        flatten = tf.keras.layers.Flatten()(inputs)
        linear1 = tf.keras.layers.Dense(1000, 'relu')(flatten)
        linear2 = tf.keras.layers.Dense(1000, 'relu')(linear1)
        lastlayer = tf.keras.layers.Dense(units=512, activation="relu")(linear2)
        outputs = tf.keras.layers.Dense(units=1, activation = None)(lastlayer)
        return inputs, lastlayer, outputs