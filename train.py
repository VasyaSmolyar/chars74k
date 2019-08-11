from keras.models import Sequential
from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Dropout, Flatten

import load

def main():
    x_train, x_test, y_train, y_test = load.load_data()
    
if __name__ == "__main__":
    main()