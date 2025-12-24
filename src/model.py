import keras
from keras import layers


@keras.saving.register_keras_serializable()
class RealTimeCNN(keras.Model):
    def __init__(self, num_classes=7, **kwargs):
        super().__init__(**kwargs)
        self.num_classes: int = num_classes

        # Block 1
        self.conv1_1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.bn1 = layers.BatchNormalization()
        self.conv1_2 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.bn2 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop1 = layers.Dropout(0.25)

        # Block 2
        self.conv2_1 = layers.Conv2D(64, (5, 5), padding="same", activation="relu")
        self.bn3 = layers.BatchNormalization()
        self.conv2_2 = layers.Conv2D(64, (5, 5), padding="same", activation="relu")
        self.bn4 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop2 = layers.Dropout(0.25)

        # Block 3
        self.conv3_1 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")
        self.bn5 = layers.BatchNormalization()
        self.conv3_2 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")
        self.bn6 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop3 = layers.Dropout(0.25)

        # Block 4
        self.conv4_1 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")
        self.bn7 = layers.BatchNormalization()
        self.conv4_2 = layers.Conv2D(256, (3, 3), padding="same", activation="relu")
        self.bn8 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))
        self.drop4 = layers.Dropout(0.25)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation="relu")
        self.bn9 = layers.BatchNormalization()
        self.drop5 = layers.Dropout(0.5)

        self.dense2 = layers.Dense(256, activation="relu")
        self.bn10 = layers.BatchNormalization()
        self.drop6 = layers.Dropout(0.5)

        self.out = layers.Dense(num_classes, activation="softmax")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.conv1_1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv1_2(x)
        x = self.bn2(x, training=training)
        x = self.pool1(x)
        x = self.drop1(x, training=training)

        x = self.conv2_1(x)
        x = self.bn3(x, training=training)
        x = self.conv2_2(x)
        x = self.bn4(x, training=training)
        x = self.pool2(x)
        x = self.drop2(x, training=training)

        x = self.conv3_1(x)
        x = self.bn5(x, training=training)
        x = self.conv3_2(x)
        x = self.bn6(x, training=training)
        x = self.pool3(x)
        x = self.drop3(x, training=training)

        x = self.conv4_1(x)
        x = self.bn7(x, training=training)
        x = self.conv4_2(x)
        x = self.bn8(x, training=training)
        x = self.pool4(x)
        x = self.drop4(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.bn9(x, training=training)
        x = self.drop5(x, training=training)

        x = self.dense2(x)
        x = self.bn10(x, training=training)
        x = self.drop6(x, training=training)

        return self.out(x)

    def get_config(self):
        return super().get_config()

