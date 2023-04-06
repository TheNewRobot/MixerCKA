import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from mixer import Mixer


def create_cifar10_mixer(dropout_rate: float):
    mixer = Mixer(
        num_classes=10,
        num_blocks=6,
        patch_size=4,
        num_patches=64,
        hidden_dim=128,
        num_token_hidden=128,
        num_channel_hidden=512,
        dropout_rate=dropout_rate,
    )
    mixer.build(input_shape=(None, 32, 32, 3))
    return mixer


def create_cifar10_mixer_v2(dropout_rate: float):
    mixer = Mixer(
        num_classes=10,
        num_blocks=8,
        patch_size=4,
        num_patches=64,
        hidden_dim=256,
        num_token_hidden=128,
        num_channel_hidden=1024,
        dropout_rate=dropout_rate,
    )
    mixer.build(input_shape=(None, 32, 32, 3))
    return mixer


def main():
    tf.random.set_seed(0)

    run = wandb.init()

    lr = run.config.lr
    batch_size = run.config.batch_size
    weight_decay = run.config.weight_decay
    dropout_rate = run.config.dropout_rate
    num_blocks = run.config.num_blocks
    patch_size, num_patches = run.config.patch
    hidden_dim = run.config.hidden_dim
    num_token_hidden = run.config.num_token_hidden
    num_channel_hidden = run.config.num_channel_hidden
    preprocessing_level = run.config.preprocessing_level

    (X_train, y_train), _ = keras.datasets.cifar10.load_data()

    if preprocessing_level == 0:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1/255,
            validation_split=0.2,
        )
    elif preprocessing_level == 1:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1/255,
            validation_split=0.2,
        )
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            rotation_range=5,
            shear_range=5,
            zoom_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
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

    wandb_callback = wandb.keras.WandbCallback(save_model=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=2,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=4,
    )
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=15,
        steps_per_epoch=int((len(X_train) * 0.8) / batch_size),
        validation_steps=int((len(X_train) * 0.2) / batch_size),
        callbacks=[reduce_lr, early_stopping, wandb_callback],
    )


def start_sweep():
    sweep_config = {
        'method': 'bayes',
        'name': 'cifar10-arch-v3',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss',
        },
        'parameters': {
            'batch_size': {'values': [256]},
            'lr': {'max': 0.01, 'min': 0.0001},
            'weight_decay': {'max': 0.01, 'min': 0.0001},
            'dropout_rate': {'max': 0.5, 'min': 0.0},
            'num_blocks': {'values': [4, 6, 8]},
            'patch': {'values': [(4, 64), (8, 16)]},
            'hidden_dim': {'values': [64, 128, 256]},
            'num_token_hidden': {'values': [64, 128, 256]},
            'num_channel_hidden': {'values': [256, 512, 1024]},
            'preprocessing_level': {'values': [0, 1, 2]},
        },
        'run_cap': 250,
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='mlp-mixer')
    wandb.agent(sweep_id=sweep_id, function=main)


if __name__ == "__main__":
    # NOTE: sanity check for mixer blocks
    start_sweep()
