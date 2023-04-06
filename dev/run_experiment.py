import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from mixer import Mixer

def main():
    tf.random.set_seed(0)

    lr = 0.002
    batch_size = 16
    weight_decay = 0.0001
    dropout_rate = 0
    num_blocks = 8
    patch_size, num_patches = (4, 64)
    hidden_dim = 1024
    num_token_hidden = 1024
    num_channel_hidden = 2048

    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    datagen = keras.preprocessing.image.ImageDataGenerator(
        # featurewise_center=True,
        # width_shift_range=0.05,
        # height_shift_range=0.05,
        # rotation_range=5,
        # shear_range=5,
        # zoom_range=0.05,
        # horizontal_flip=True,
        # vertical_flip=True,
        rescale=1/255,
        validation_split=0.2,
    )
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
    val_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

    model = Mixer(
        num_classes=10,
        num_blocks=num_blocks,
        patch_size=patch_size,
        num_patches=num_patches,
        hidden_dim=hidden_dim,
        num_token_hidden=num_token_hidden,
        num_channel_hidden=num_channel_hidden,
        dropout_rate=dropout_rate,
    )
    model.build(input_shape=(None, 32, 32, 3))    
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=4,
    )
    # early_stopping = keras.callbacks.EarlyStopping(
    #     monitor='val_loss', 
    #     patience=10,
    # )
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=1000,
        steps_per_epoch=int((len(X_train) * 0.8) / batch_size),
        validation_steps=int((len(X_train) * 0.2) / batch_size),
        callbacks=[reduce_lr],
        # callbacks=[reduce_lr, early_stopping],
    )


if __name__ == "__main__":
    # NOTE: sanity check for mixer blocks
    main()
