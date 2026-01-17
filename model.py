import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
from keras.applications import EfficientNetB0
import os
from tensorflow.keras.applications import efficientnet
class BuildModel:
    def __init__(self, img_size, backbone_cls, num_class=1):
        self.num_class = num_class
        self.backbone_cls = backbone_cls
        self.img_size = img_size

    def build(self):
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))

        # aug = keras.Sequential(
        #     [
        #         layers.RandomRotation(0.15),
        #         layers.RandomTranslation(0.1, 0.1),
        #         layers.RandomFlip("horizontal"),
        #         layers.RandomContrast(0.1),
        #     ],
        #     name="img_augmentation",
        # )

        # x = aug(inputs)
        x = layers.Lambda(efficientnet.preprocess_input, name="preprocess")(inputs)  

        backbone = self.backbone_cls(
            include_top=False,
            weights="imagenet",
            input_tensor=x
        )
        backbone._name = "backbone"
        backbone.trainable = False

        x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2, name="top_dropout")(x)

        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)

        model = keras.Model(inputs, outputs, name="model_x")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
                tf.keras.metrics.AUC(name="auc")
            ]
        )
        return model

    def get_element(self,file):
        os.makedirs(file, exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(file, "epoch_{epoch:03d}_bestmodel.keras"),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        )


        reduce_lr = ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
        return checkpoint,reduce_lr,early_stop

