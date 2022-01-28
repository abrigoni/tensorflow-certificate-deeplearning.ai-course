import tensorflow as tf


class MyCallback(tf.keras.callbacks.Callback):

    def __init__(self, required_accuracy=0.9):
        super().__init__()
        self.required_accuracy = required_accuracy

    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')) >= 0.99:
            print(f"\nReached {round(self.required_accuracy * 100, 1)}% accuracy so cancelling training!")
            self.model.stop_training = True


if __name__ == "__main__":
    callback = MyCallback(required_accuracy=0.4)
    print(callback)
