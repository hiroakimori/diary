# henkousitayo
from keras.layers import Conv2D
from keras.utils import np_utils
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.image as mpimg
from scipy.misc import imresize
import numpy as np
import keras.backend as K
import math


K.clear_session()
img_size=299

classes = ["dog","cat","bird"]
num_classes = len(classes)

# メインの関数を定義する

X_train, X_test, y_train, y_test = np.load("./animal.npy")
X_train = X_train.astype("float") / 256 # 正規化：0~255が0〜1になる
X_test = X_test.astype("float") / 256
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=[.8, 1],
        channel_shift_range=30,
        fill_mode='reflect')

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=64, seed = 13)
test_generator = test_datagen.flow(X_test, y_test, batch_size=64, seed = 13)

print(train_generator)
print(test_generator)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpointとlogの出力？
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)

# train the model on the new data for a few epochs
model.fit_generator(train_generator,
                    steps_per_epoch=100,
                    epochs=5,
                    validation_data=test_generator,
                    validation_steps=80,
                    verbose=2,
                    callbacks=[reduce_lr, csv_logger, checkpointer])



