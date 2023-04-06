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

    # run = wandb.init()

    # lr = run.config.lr
    # batch_size = run.config.batch_size
    # weight_decay = run.config.weight_decay
    # dropout_rate = run.config.dropout_rate
    batch_size = 8

    (X_train, y_train), _ = keras.datasets.cifar10.load_data()
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        # zca_whitening=True,
        # width_shift_range=0.05,
        # height_shift_range=0.05,
        # rotation_range=5,
        # shear_range=5,
        # zoom_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255,
    )
    # datagen.fit(X_train)
    train_generator = datagen.flow(X_train[:16], y_train[:16], batch_size=batch_size)
    import matplotlib.pyplot as plt

    X_batch, y_batch = next(train_generator)

    # take first image
    for i in range(8):
        image = X_batch[i].astype(float)
        # take first image label index
        label = y_batch[i]
        # Reshape the image
        # image = image.reshape(3,32,32)
        # Transpose the image
        # image = image.transpose(1,2,0)
        # Display the image
        plt.imshow(image)
        plt.title(str(label))
        plt.show()

    exit()
    model = create_cifar10_mixer_v2(dropout_rate=dropout_rate)
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
        'name': 'cifar10-arch-v2',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss',
        },
        'parameters': {
            'batch_size': {'values': [256]},
            'lr': {'max': 0.01, 'min': 0.00001},
            'weight_decay': {'max': 0.01, 'min': 0.00001},
            'dropout_rate': {'max': 0.5, 'min': 0.0},
        },
        'run_cap': 100,
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project='mlp-mixer')
    wandb.agent(sweep_id=sweep_id, function=main)


if __name__ == "__main__":
    main()

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
