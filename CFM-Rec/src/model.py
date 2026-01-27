# src/model.py
import tensorflow as tf
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = tf.exp(-math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half) * 2 * math.pi
    
    t = tf.squeeze(tf.cast(timesteps, tf.float32), axis=-1) 
    args = t[:, None] * freqs[None]
    
    embedding = tf.concat([tf.cos(args), tf.sin(args)], axis=-1)
    if dim % 2:
        embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

class FlowModel(tf.keras.Model):
    def __init__(self, dims, time_emb_dim, dropout_rate=0.0):
        super(FlowModel, self).__init__()
        self.time_emb_dim = time_emb_dim

        self.time_dense = tf.keras.layers.Dense(time_emb_dim)
        self.time_act = tf.keras.layers.LeakyReLU(alpha=0.2) 

        self.mlp_layers = []
        for i, dim in enumerate(dims):
            self.mlp_layers.append(tf.keras.layers.Dense(dim))
  
            if i < len(dims) - 1:
                self.mlp_layers.append(tf.keras.layers.LeakyReLU(alpha=0.2))
                self.mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(self, x_t, cond, t, training=False):
        t_emb = timestep_embedding(t, self.time_emb_dim)
        
        t_emb = self.time_dense(t_emb)
        t_emb = self.time_act(t_emb)

        h = tf.concat([x_t, cond, t_emb], axis=-1)

        for layer in self.mlp_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                h = layer(h, training=training)
            else:
                h = layer(h)
                
        return h