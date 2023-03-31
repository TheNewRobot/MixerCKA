from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import pickle


class MLP(keras.Model):
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


class TokenMixingMLP(keras.Model):
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


class ChannelMixingMLP(keras.Model):
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


class MLPMixerLayer(keras.Model):
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
        self.input_layer = keras.layers.InputLayer(input_shape=(32, 32, 3))
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
        x = self.input_layer(inputs)
        x = self.data_augmentation(x)
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


def build_model(seed):
    tf.keras.utils.set_random_seed(seed)
    data_augmentation = build_data_augmention(X_train, image_size=32)
    model = MLPMixer(
        data_augmentation=data_augmentation,
        num_classes=10,
        num_blocks=6,
        image_shape=(32,32),
        patch_size=4,
        num_tokens=64,
        hidden_dim=128,
        num_token_hidden=128,
        num_channel_hidden=512,
        postional_encoding=False,
        dropout_rate=0.2,
    )
    model.build(input_shape=(None, 32, 32, 3))
    return model
(X_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

model = build_model(0)
model.summary(expand_nested=True)




# exit(0)
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


# def build_data_augmention(X_train, image_size):
#     data_augmentation = keras.Sequential(
#         [
#             keras.layers.Normalization(),
#             keras.layers.Resizing(image_size, image_size),
#             keras.layers.RandomFlip("horizontal"),
#             keras.layers.RandomZoom(
#                 height_factor=0.2, width_factor=0.2
#             ),
#         ],
#         name="data_augmentation",
#     )
#     data_augmentation.layers[0].adapt(X_train)
#     return data_augmentation


# (X_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# def build_model(seed):
#     tf.keras.utils.set_random_seed(seed)
#     data_augmentation = build_data_augmention(X_train, image_size=224)
#     model = MLPMixer(
#         data_augmentation=data_augmentation,
#         num_classes=10,
#         num_blocks=12,
#         image_shape=(224,224),
#         patch_size=32,
#         num_tokens=49,
#         hidden_dim=786,
#         num_token_hidden=384,
#         num_channel_hidden=3072,
#         postional_encoding=False,
#         dropout_rate=0.2,
#     )
#     model.build(input_shape=(None, 32, 32, 3))
#     return model


# weight_decay = 0.0001
# batch_size = 512 
# num_epochs = 50
# dropout_rate = 0.2
# learning_rate = 0.005


# def run_experiment(model):
#     # Create Adam optimizer with weight decay. Regularization that penalizes the increase of weight - with a facto alpha - to correct the overfitting
#     optimizer = tfa.optimizers.AdamW(
#         learning_rate=learning_rate, weight_decay=weight_decay,
#     )
#     # Compile the model.
#     model.compile(
#         optimizer=optimizer,
#         #Negative Log Likelihood = Categorical Cross Entropy
#         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         metrics=[
#             keras.metrics.SparseCategoricalAccuracy(name="acc"),
#             keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
#         ],
#     )
#     # Create a learning rate scheduler callback.
#     reduce_lr = keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss", factor=0.5, patience=5
#     )
#     # Create an early stopping regularization callback. 
#     # It ends at a point that corresponds to a minimum of the L2-regularized objective
#     #early_stopping = tf.keras.callbacks.EarlyStopping(
#     #    monitor="val_loss", patience=10, restore_best_weights=True
#     #)
#     # Fit the model.
#     history = model.fit(
#         x=X_train,
#         y=y_train,
#         batch_size=batch_size,
#         epochs=1,
#         validation_split=0.1,
#         callbacks=[reduce_lr],
#     )

#     _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

#     # Return history to plot learning curves.
#     return history, accuracy, top_5_accuracy


# def mlpmixer_generator(num_models):
#     #now = datetime.datetime.now()
#     #date = now.strftime("%Y-%m-%d_%H-%M")
#     for seed in range(num_models):
#         mlpmixer_classifier = build_model(seed) # Returns the model
#         history,accuracy, top_5_accuracy = run_experiment(mlpmixer_classifier)
#         mlpmixer_classifier.summary(expand_nested=True)

#         #Saving Results
#         mlpmixer_classifier.save(f"mlpmixer_B-32_{seed}_final")
#         np.save(f'mlpmixer_B-32_{seed}_final/history.npy',history.history)
#         with open(f'mlpmixer_B-32_{seed}_final/accuracy.pkl','wb') as file:
#             pickle.dump(accuracy,file)
#         with open(f'mlpmixer_B-32_{seed}_final/top5-accuracy.pkl','wb') as file:
#             pickle.dump(top_5_accuracy,file)


# mlpmixer_generator(10)