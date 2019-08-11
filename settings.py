# LOADER SETTINGS

#Resize pictures or not
resize = True

#width and height of resized pictures
size = (28,28)

#use only good images or not
only_good_imgs = False

#part of validation sample
validation_rate = 0.1

#save prepared array in h5 file and load from this file or not
save_img_narray = True

# CNN SETTINGS

#number of convolution layers
num_of_conv = 3

#convolution size
conv_size = 4

#convolution filters
conv_filters = 32

#activation function
activate = 'relu'

#dropout rate
drop_rate = 0.3

#optimizer
optimizer = 'adam'

#number of epochs
nb_epoch = 30

#batch size
batch = 128

#load weights before train or not
load_weights = True