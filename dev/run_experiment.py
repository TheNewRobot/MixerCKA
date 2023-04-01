import wandb
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from mixer import Mixer


def create_cifar10_mixer():
    mixer = Mixer(
        num_classes=10,
        num_blocks=6,
        patch_size=4,
        num_patches=64,
        hidden_dim=128,
        num_token_hidden=128,
        num_channel_hidden=512,
        dropout_rate=0.1,
    )
    mixer.build(input_shape=(None, 32, 32, 3))
    return mixer


def main():
    tf.random.set_seed(0)

    run = wandb.init()

    # lr = run.config.lr
    # batch_size = run.config.batch_size
    # weight_decay = run.config.weight_decay
    lr = 0.0001
    batch_size = 512
    weight_decay = 0.0001

    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        zca_whitening=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rotation_range=20,
        shear_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255,
        validation_split=0.2,
    )
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='training')
    val_generator = datagen.flow(X_train, y_train, batch_size=batch_size, subset='validation')

    model = create_cifar10_mixer()
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
        patience=3,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=6,
    )
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=300,
        steps_per_epoch=int((len(X_train) * 0.8) / batch_size),
        validation_steps=int((len(X_train) * 0.2) / batch_size),
        callbacks=[reduce_lr, early_stopping, wandb_callback],
    )


def start_sweep():
    sweep_config = {
        'method': 'bayes',
        'name': 'test_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss',
        },
        'parameters': {
            'batch_size': {'values': [256, 512]},
            'lr': {'max': 0.001, 'min': 0.00001},
            'weight_decay': {'max': 0.001, 'min': 0.00001},
        },
        'run_cap': 100,
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='test-sweep-2')
    wandb.agent(sweep_id=sweep_id, function=main)


if __name__ == "__main__":
    start_sweep()

    # print(tf.config.list_physical_devices('GPU'))
    # (X_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # print(X_train[0])
    # exit()

    # mixer = create_cifar10_mixer()
    # # optimizer = tfa.optimizers.AdamW(weight_decay=0.004)
    # optimizer = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
    # mixer.compile(
    #     optimizer=optimizer,
    #     #Negative Log Likelihood = Categorical Cross Entropy
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[
    #         keras.metrics.SparseCategoricalAccuracy(name="acc"),
    #         keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
    #     ],
    # )

    # mixer.fit(X_train, y_train, batch_size=512, epochs=100)
