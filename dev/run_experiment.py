import wandb
from tensorflow import keras
from mixer import Mixer
import tensorflow_addons as tfa


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
    sweep_config = {
        'method': 'grid',
        'name': 'test_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss',
        },
        'parameters': {
            'batch_size': {'values'}
        }
    }


def run_experiment(model):
    # Create Adam optimizer with weight decay. Regularization that penalizes the increase of weight - with a facto alpha - to correct the overfitting
    
    # Compile the model.
    model.compile(
        optimizer=optimizer,
        #Negative Log Likelihood = Categorical Cross Entropy
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )
    # Create a learning rate scheduler callback.
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # Create an early stopping regularization callback. 
    # It ends at a point that corresponds to a minimum of the L2-regularized objective
    #early_stopping = tf.keras.callbacks.EarlyStopping(
    #    monitor="val_loss", patience=10, restore_best_weights=True
    #)
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[reduce_lr],
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # Return history to plot learning curves.
    return history, accuracy, top_5_accuracy


if __name__ == "__main__":
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    exit()
    (X_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    mixer = create_cifar10_mixer()
    optimizer = tfa.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.005,
    )
    mixer.compile(
        optimizer=optimizer,
        #Negative Log Likelihood = Categorical Cross Entropy
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top5-acc"),
        ],
    )

    mixer.fit(X_train, y_train, batch_size=32, epochs=1)
