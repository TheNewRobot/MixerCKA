from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf


class MLP(keras.layers.Layer):
    def __init__(self, num_input, num_hidden, dropout_rate):
        super().__init__()

        self._num_input = num_input
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        self.dense1 = keras.layers.Dense(num_hidden, input_shape=(num_input,))
        self.gelu = tfa.layers.GELU()
        self.dense2 = keras.layers.Dense(num_input, input_shape=(num_hidden,))
        self.dropout = keras.layers.Dropout(dropout_rate)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.gelu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_input": self._num_input,
                "num_hidden": self._num_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config


class TokenMixingMLP(keras.layers.Layer):
    def __init__(self, num_tokens, num_hidden, dropout_rate):
        super().__init__()

        self._hidden_dim = num_tokens
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        self.mlp = MLP(
            num_input=num_tokens,
            num_hidden=num_hidden,
            dropout_rate=dropout_rate
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.norm(inputs)
        x = tf.linalg.matrix_transpose(x)
        x = self.mlp(x)
        x = tf.linalg.matrix_transpose(x)
        x = x + inputs
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "hidden_dim": self._hidden_dim,
                "num_hidden": self._num_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config


class ChannelMixingMLP(keras.layers.Layer):
    def __init__(self, hidden_dim, num_hidden, dropout_rate):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._num_hidden = num_hidden
        self._dropout_rate = dropout_rate

        self.mlp = MLP(
            num_input=hidden_dim, 
            num_hidden=num_hidden, 
            dropout_rate=dropout_rate
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        x = self.norm(inputs)
        x = self.mlp(x)
        x = x + inputs
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "hidden_dim": self._hidden_dim,
                "num_hidden": self._num_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config


class MLPMixerLayer(keras.layers.Layer):
    def __init__(self, num_tokens, hidden_dim, num_token_hidden, num_channel_hidden, dropout_rate):
        super().__init__()

        self._num_tokens = num_tokens
        self._hidden_dim = hidden_dim
        self._num_token_hidden = num_token_hidden
        self._num_channel_hidden = num_channel_hidden
        self._dropout_rate = dropout_rate

        self.token_mixing_mlp = TokenMixingMLP(
            num_tokens=num_tokens,
            num_hidden=num_token_hidden,
            dropout_rate=dropout_rate,
        )
        self.channel_mixing_mlp = ChannelMixingMLP(
            hidden_dim=hidden_dim,
            num_hidden=num_channel_hidden,
            dropout_rate=dropout_rate,
        )
    
    def call(self, inputs):
        x = self.token_mixing_mlp(inputs)
        x = self.channel_mixing_mlp(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "num_tokens": self._num_tokens,
                "hidden_dim": self._hidden_dim,
                "num_token_hidden": self._num_token_hidden,
                "num_channel_hidden": self._num_channel_hidden,
                "dropout_rate": self._dropout_rate,
            }
        )
        return config

    @property
    def mlp1(self):
        return self.token_mixing_mlp
    
    @property
    def mlp2(self):
        return self.channel_mixing_mlp


class Patches(keras.layers.Layer):
    def __init__(self, patch_size, num_patches):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def call(self, images):
        #Extract the shape dimension in the position 0 = columns
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            #Without overlapping, stride horizontally and vertically
            strides=[1, self.patch_size, self.patch_size, 1],
            #Rate: Dilation factor [1 1* 1* 1] controls the spacing between the kernel points.
            rates=[1, 1, 1, 1],
            #Patches contained in the images are considered, no zero padding
            padding="VALID",
        )
        #shape[-1], number of colummns, as well as shape[0]
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, self.num_patches, patch_dims])
        return patches

    def get_config(self):
        config = super(Patches, self).get_config().copy()
        config.update ({
            'patch_size' : self.patch_size ,
            'num_patches' : self.num_patches
        })
        return config


class MLPMixer(keras.Model):
    def __init__(
        self, 
        data_augmentation,
        num_classes, 
        num_blocks, 
        image_shape, 
        patch_size, 
        num_tokens, 
        hidden_dim, 
        num_token_hidden, 
        num_channel_hidden,
        postional_encoding,
        dropout_rate,
    ):
        super().__init__()

        self._postional_encoding = postional_encoding

        self.data_augmentation = data_augmentation
        self.patches = Patches(patch_size=patch_size, num_patches=num_tokens)
        self.embedding = keras.layers.Dense(hidden_dim)
        if postional_encoding:
            positions = tf.range(start=0, limit=num_tokens, delta=1)
            self.position_embedding = keras.layer.Embedding(
                input_dim=num_tokens, output_dim=hidden_dim,
            )(positions)

        self.blocks = []
        for _ in range(num_blocks):
            self.blocks.append(MLPMixerLayer(num_tokens=num_tokens, hidden_dim=hidden_dim, num_token_hidden=num_token_hidden, num_channel_hidden=num_channel_hidden, dropout_rate=dropout_rate))

        self.pooling = keras.layers.GlobalAveragePooling1D()
        self.dropout = keras.layers.Dropout(rate=dropout_rate)
        self.logits = keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.data_augmentation(inputs)
        x = self.patches(x)
        x = self.embedding(x)
        if self._postional_encoding:
            x = x + self.position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        x = self.dropout(x)
        x = self.logits(x)
        return x



# m = MLP(32, 64, 0.2)
# m.summary()
        # Transpose inputs from [num_batches, num_patches, hidden_units] to [num_batches, hidden_units, num_patches].
# model = keras.Sequential(
#     [
#         keras.layers.Dense(units=49),
#         tfa.layers.GELU(),
#         keras.layers.Dense(units=384),
#         keras.layers.Dropout(rate=0.2),
#     ]
# )
# model.build(input_shape=(49, 384))
# model.summary()


def build_data_augmention(X_train, image_size):
    data_augmentation = keras.Sequential(
        [
            keras.layers.Normalization(),
            keras.layers.Resizing(image_size, image_size),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )
    data_augmentation.layers[0].adapt(X_train)
    return data_augmentation


(X_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

data_augmentation = build_data_augmention(X_train, image_size=224)
model = MLPMixer(
    data_augmentation=data_augmentation,
    num_classes=2,
    num_blocks=12,
    image_shape=(224,224),
    patch_size=32,
    num_tokens=49,
    hidden_dim=786,
    num_token_hidden=384,
    num_channel_hidden=3072,
    postional_encoding=False,
    dropout_rate=0.2,
)
model.build(input_shape=(32, 32, 3))
model.summary()
exit()


NUM_TOKENS = 49
HIDDEN_DIM = 768
NUM_TOKEN_HIDDEN = 384
NUM_CHANNEL_HIDDEN = 3072
DROPOUT_RATE = 0.2

model = keras.Sequential(
    [MLPMixerLayer(
        num_tokens=NUM_TOKENS,
        hidden_dim=HIDDEN_DIM,
        num_token_hidden=NUM_TOKEN_HIDDEN,
        num_channel_hidden=NUM_CHANNEL_HIDDEN,
        dropout_rate=DROPOUT_RATE,
    ) for _ in range(8)]
)
model.build(input_shape=(NUM_TOKENS, HIDDEN_DIM))
model.summary()


