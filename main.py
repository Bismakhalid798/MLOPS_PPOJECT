import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from skimage.feature import canny
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision,Accuracy
from matplotlib import pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
H=256
W=256
dim=(H,W)
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#Metrices
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def eval_metrics(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the dvc file from the URL
    dvc_url = (
        "https://github.com/Bismakhalid798/MLOPS_PPOJECT/blob/60a2726c882c704d1d111e61dd60b3ae5e220b74/datasets.dvc"
    )
    try:
        data = pd.read_dvc(dvc_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test data, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


    import mlflow
    import mlflow.keras

    def Conv_Block(input,num_of_filter):
        x = Conv2D(num_of_filter, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(num_of_filter, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
  
        return x

    def Encoder(input,num_of_filter):
        x = Conv_Block(input, num_of_filter)
        p = MaxPool2D((2, 2))(x)
        return x, p

    def Decoder(input, skip_features, num_of_filter):
        x = Conv2DTranspose(num_of_filter, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = Conv_Block(x, num_of_filter)
        return x

    def UNET_Build(input_shape):
        mlflow.keras.autolog()  # Enable MLflow autologging for Keras

        inputs = Input(input_shape)
        s1, p1 = Encoder(inputs, 64)
        s2, p2 = Encoder(p1, 128)
        s3, p3 = Encoder(p2, 256)
        s4, p4 = Encoder(p3, 512)

        b1 = Conv_Block(p4, 1024)

        d1 = Decoder(b1, s4, 512)
        d2 = Decoder(d1, s3, 256)
        d3 = Decoder(d2, s2, 128)
        d4 = Decoder(d3, s1, 64)

        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="U-Net")
        return model

    # Specify the input shape
    input_shape = (256, 256, 3)

    # Create the U-Net model
    model = UNET_Build(input_shape)

    # Compile the model and set up optimizer, loss, etc.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Start an MLflow run   
    with mlflow.start_run():
        # Log the model architecture
        mlflow.keras.log_model(model, "model")
    
        # Train the model
        model.fit(train_x, train_y, epochs=10, validation_data=(x_val, y_val))

    with mlflow.start_run():
        lr = UNET_Build(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Feature model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="No_Feature.h5")
        else:
            mlflow.sklearn.log_model(lr, "model")
