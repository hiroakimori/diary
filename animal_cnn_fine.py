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
#訓練データ拡張
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

#画像の読み込み
def load_images(root,nb_img):
    all_imgs = []
    all_classes = []

    for i in range(nb_img):
        img_name = "%s/dog.%d.jpg" % (root, i + 1)
        img_arr = mpimg.imread(img_name)
        resize_img_ar = imresize(img_arr, (img_size, img_size))
        all_imgs.append(resize_img_ar)
        all_classes.append(0)
    for i in range(nb_img):
        img_name = "%s/cat.%d.jpg" % (root, i + 1)
        img_arr = mpimg.imread(img_name)
        resize_img_ar = imresize(img_arr, (img_size, img_size))
        all_imgs.append(resize_img_ar)
        all_classes.append(1)
    return np.array(all_imgs), np.array(all_classes)

X_train, y_train = load_images('./train', 100)
X_test, y_test = load_images('./train', 40)
train_generator = train_datagen.flow(X_train, y_train, batch_size=64, seed = 13)
test_generator = test_datagen.flow(X_test, y_test, batch_size=64, seed = 13)

#Inception v3モデルの読み込み。最終層は読み込まない
base_model = InceptionV3(weights='imagenet', include_top=False)
#最終層の設定　add a global spatial average pooling layer　xにどんどん追加しているイメージ
x = base_model.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer 全結合層　1つのクラス、即ち犬or猫の出力
predictions = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid", kernel_regularizer=l2(.0005))(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#base_modelはweightsを更新しない
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# Checkpointとlogの出力？
checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)


# train the model on the new data for a few epochs
history = model.fit_generator(train_generator,
                    steps_per_epoch=1000,
                    epochs=5,
                    validation_data=test_generator,
                    validation_steps=800,
                    verbose=1,
                    callbacks=[reduce_lr, csv_logger, checkpointer])