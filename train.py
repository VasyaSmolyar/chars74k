from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

import os
import load
import settings

def get_model(input_shape, y_shape, conv=2, csize=3, filters=32, act='relu', drop=0.3):
    model = Sequential()
    model.add(Conv2D(filters, (csize, csize), activation=act, padding='same', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Dropout(drop))
    if conv > 0:
        for i in range(conv-1):
            filters *= 2
            model.add(Conv2D(filters, (csize, csize), activation=act, padding='same'))
            model.add(MaxPooling2D())
            model.add(Dropout(drop))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation(act))
    model.add(Dense(y_shape))
    model.add(Activation('softmax'))
    return model

def main():
    x_train, x_test, y_train, y_test = load.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    model = get_model((x_train.shape[1], x_train.shape[2], x_train.shape[3],), y_train.shape[1], conv=settings.num_of_conv, 
        csize=settings.conv_size, filters=settings.conv_filters, act=settings.activate, drop=settings.drop_rate)
    model.compile(loss="categorical_crossentropy", optimizer=settings.optimizer, metrics=["accuracy"])
    fname = 'model_conv{}size_{}_filt{}_{}_drop{}.h5'.format(settings.num_of_conv, settings.conv_size,
         settings.conv_filters, settings.activate, settings.drop_rate)
    name = os.path.join('data', fname)
    if os.path.isfile(name) and settings.load_weights:
        model.load_weights(name)
    checkpoint = ModelCheckpoint(name, monitor='val_acc', save_best_only=True, 
        save_weights_only=True, mode='auto')
    model.fit(x_train, y_train, batch_size=settings.batch, epochs=settings.nb_epoch, 
        validation_data=(x_test, y_test), shuffle=True, callbacks=[checkpoint])


if __name__ == '__main__':
    main()