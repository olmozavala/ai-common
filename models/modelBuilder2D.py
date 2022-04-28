from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        return config

def single_conv_lay_2d(input, num_filters, filter_size, batch_norm=False, dropout=False, activation='relu'):
    """ Adds a single convolutional layer with the specifiec parameters. It can add batch normalization and dropout
        :param input: The layer to use as input
        :param num_filters: The number of filters to use
        :param filter_size:  Size of the filters
        :param batch_norm: If we want to use batch normalization after the CNN
        :param dropout: If we want to use dropout after the CNN
        :return:
    """
    # conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation=activation)(input)
    pad = int(filter_size/2)
    padding = ReflectionPadding2D(padding=(pad,pad))(input)
    conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='valid', activation=activation)(padding)

    # Adding batch normalization
    if batch_norm :
        # Default values: (axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
        #                  beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
        #                  moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
        #                  beta_constraint=None, gamma_constraint=None)
        next = BatchNormalization()(conv1)  # Important, which axis?
    else:
        next = conv1
    # Adding dropout
    if dropout:
        final = Dropout(rate=0.2)(next)
    else:
        final = next
    return final


def multiple_conv_lay_2d(input, num_filters, filter_size, make_pool=True, batch_norm=False,
                         dropout=False, tot_layers=2, activation='relu'):
    """ Adds a N convolutional layers followed by a max pool
        :param tot_layers: how many layers we want to create
        :param input: The layer to use as input
        :param num_filters: The number of filters to use
        :param filter_size:  Size of the filters
        :param batch_norm: If we want to use batch normalization after the CNN
        :param dropout: If we want to use dropout after the CNN
    :return:
    """
    c_input = input
    for c_layer_idx in range(tot_layers):
        c_input = single_conv_lay_2d(c_input, num_filters, filter_size, batch_norm=batch_norm, dropout=dropout,
                                     activation=activation)

    if make_pool:
        maxpool = MaxPooling2D(pool_size=(2, 2))(c_input)
    else:
        maxpool = []

    return c_input, maxpool

def make_multistream_cnn_rnn(inputs, num_filters=8, filter_size=3, num_levels=3,
                             batch_norm_encoding=False,
                             batch_norm_decoding=True,
                             dropout_encoding=False,
                             dropout_decodign=True, activation='relu',
                             last_activation='sigmoid',
                             number_output_filters = 1,
                             output_cnn_layers = 4):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param batch_norm_encoding: Indicates if we are using batch normalization in the encoding phase
    :param batch_norm_decoding: Indicates if we are using batch normalization in the decoding phase
    :param dropout_encoding: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm_encoding,
                                                  activation=activation,
                                                  dropout=dropout_encoding)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
        print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')
    else: # This is the single stream case (Default 2D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]
        print(F'Size at the bottom: {merged.shape}')


    # Convoulutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm_decoding,
                                          activation=activation,
                                          dropout=dropout_decodign)

    print("\n ------------- DECODING PATH ----------------")
    print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")
    # conv_t = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_p)
    # conv_t_u = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
    # conv_t = BatchNormalization()(conv_b)
    conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")


    for level in range(1,num_levels+1):
        bottom_up_level = num_levels-level

        # print(F" Concatenating {conv_t.shape} with previous convs {streams[0]['convs'][bottom_up_level].shape} (each)")
        merged_temp = [conv_t]
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['convs'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')

        filters = num_filters * (2 ** (bottom_up_level))
        [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False,
                                              activation=activation,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")

        if level != (num_levels):
            # conv_t = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_p)
            # conv_t_u = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
            # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
            # conv_t = BatchNormalization()(conv_b)
            conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
            print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")

    c_output_cnn = 2
    while output_cnn_layers > c_output_cnn:
        [conv_p, dele] = multiple_conv_lay_2d(conv_p, filters, filter_size,
                                              activation=activation,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        c_output_cnn += 1
    # prev_last_conv = Conv2D(number_output_filters, (1, 1), activation=activation)(conv_p)
    # last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(prev_last_conv)
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model

def make_multistream_2d_unet(inputs, num_filters=8, filter_size=3, num_levels=3,
                             batch_norm_encoding=False,
                             batch_norm_decoding=True,
                             dropout_encoding=False,
                             dropout_decodign=True, activation='relu',
                             last_activation='sigmoid',
                             number_output_filters = 1,
                             output_cnn_layers = 4):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param batch_norm_encoding: Indicates if we are using batch normalization in the encoding phase
    :param batch_norm_decoding: Indicates if we are using batch normalization in the decoding phase
    :param dropout_encoding: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm_encoding,
                                                  activation=activation,
                                                  dropout=dropout_encoding)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
        print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')
    else: # This is the single stream case (Default 2D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]
        print(F'Size at the bottom: {merged.shape}')


    # Convoulutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm_decoding,
                                          activation=activation,
                                          dropout=dropout_decodign)

    print("\n ------------- DECODING PATH ----------------")
    print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")
    # conv_t = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_p)
    # conv_t_u = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
    # conv_t = BatchNormalization()(conv_b)
    conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")


    for level in range(1,num_levels+1):
        bottom_up_level = num_levels-level

        # print(F" Concatenating {conv_t.shape} with previous convs {streams[0]['convs'][bottom_up_level].shape} (each)")
        merged_temp = [conv_t]
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['convs'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')

        filters = num_filters * (2 ** (bottom_up_level))
        [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False,
                                              activation=activation,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")

        if level != (num_levels):
            # conv_t = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_p)
            # conv_t_u = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
            # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
            # conv_t = BatchNormalization()(conv_b)
            conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
            print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")

    c_output_cnn = 2
    while output_cnn_layers > c_output_cnn:
        [conv_p, dele] = multiple_conv_lay_2d(conv_p, filters, filter_size,
                                              activation=activation,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        c_output_cnn += 1
    # prev_last_conv = Conv2D(number_output_filters, (1, 1), activation=activation)(conv_p)
    # last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(prev_last_conv)
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model

def make_pre_upsampling_super_resolution(inputs, num_filters=8, filter_size=3, num_cnn_layers=8,
                                             inc_res_factor=2,
                                             batch_norm=True,
                                             dropout=False,
                                             activation='relu',
                                             last_activation='sigmoid',
                                             number_output_filters = 1):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_cnn_layers: The number of levels that the U-net will have
    :param inc_res_factor: The resolution factor we want to increase. It must be a power of two
    :param batch_norm: Indicates if we are using batch normalization 
    :param dropout: Indicates if we are using dropout 
    :return:
    """

    # Upsample
    # CNNs
    # Output
    # ==================== INCREASING THE RESOLUTION HERE=====================
    # First increase
    c_input = inputs[0]
    conv_t = UpSampling2D((2, 2), interpolation='nearest')(c_input) # Upsample one level (duplicate resolution)
    # Additional increases
    for c_inc_factor in range(int(np.log2(inc_res_factor))-1):
        conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_t)

    # ==================== Making CNNs ==================
    [conv_p, dele] = multiple_conv_lay_2d(conv_t, num_filters, filter_size,
                                          activation=activation,
                                          tot_layers=num_cnn_layers, # How many CNNs we want to do
                                          batch_norm=batch_norm, 
                                          dropout=dropout)
    # ==================== Last layer ==================
    # Last layer one more CNN but with filter size 1
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model

def make_SRResNet(inputs, batch_norm=True, dropout=False, activation='relu', last_activation=None, number_output_filters = 3):
    """Makes an original SRCNN. The input should be an image with already HR by bicubic interpolation
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param batch_norm: Indicates if we are using batch normalization
    :param dropout: Indicates if we are using dropout
    :return:
    """
    # First increase
    c_input = inputs[0]
    # Additional increases
    # ==================== Making CNNs ==================
    [conv_p, dele] = multiple_conv_lay_2d(c_input, 128, (9,9), activation=activation, tot_layers=1, batch_norm=batch_norm, dropout=dropout)
    [conv_p, dele] = multiple_conv_lay_2d(conv_p, 64, (3,3), activation=activation, tot_layers=1, batch_norm=batch_norm, dropout=dropout)
    [conv_p, dele] = multiple_conv_lay_2d(conv_p, 1, (5,5), activation=activation, tot_layers=1, batch_norm=batch_norm, dropout=dropout)
    # ==================== Last layer ==================
    # Last layer one more CNN but with filter size 1
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model

def make_SRCNN(inputs, batch_norm=True, dropout=False, activation='relu', last_activation=None, number_output_filters = 3):
    """Makes an original SRCNN. The input should be an image with already HR by bicubic interpolation
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param batch_norm: Indicates if we are using batch normalization
    :param dropout: Indicates if we are using dropout
    :return:
    """
    # First increase
    c_input = inputs[0]
    # Additional increases
    # ==================== Making CNNs ==================
    [conv_p, dele] = multiple_conv_lay_2d(c_input, 128, 9, activation=activation, tot_layers=4, batch_norm=batch_norm, dropout=dropout)
    [conv_p, dele] = multiple_conv_lay_2d(conv_p, 64, 3, activation=activation, tot_layers=4, batch_norm=batch_norm, dropout=dropout)
    [conv_p, dele] = multiple_conv_lay_2d(conv_p, 1, 5, activation=activation, tot_layers=4, batch_norm=batch_norm, dropout=dropout)
    # ==================== Last layer ==================
    # Last layer one more CNN but with filter size 1
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model

def make_multistream_2d_unet_superresolution(inputs, num_filters=8, filter_size=3, num_levels=3,
                                             inc_res_factor=2,
                             batch_norm_encoding=False,
                             batch_norm_decoding=True,
                             dropout_encoding=False,
                             dropout_decodign=True, activation='relu',
                             last_activation='sigmoid',
                             number_output_filters = 1,
                             output_cnn_layers = 4):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param inc_res_factor: The resolution factor we want to increase. It must be a power of two
    :param batch_norm_encoding: Indicates if we are using batch normalization in the encoding phase
    :param batch_norm_decoding: Indicates if we are using batch normalization in the decoding phase
    :param dropout_encoding: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm_encoding,
                                                  activation=activation,
                                                  dropout=dropout_encoding)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
        print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')
    else: # This is the single stream case (Default 2D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]
        print(F'Size at the bottom: {merged.shape}')


    # Convoulutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm_decoding,
                                          activation=activation,
                                          dropout=dropout_decodign)

    print("\n ------------- DECODING PATH ----------------")
    print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")
    # conv_t = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv_p)
    # conv_t_u = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
    # conv_t = BatchNormalization()(conv_b)
    conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
    print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")


    for level in range(1,num_levels+1):
        bottom_up_level = num_levels-level

        # print(F" Concatenating {conv_t.shape} with previous convs {streams[0]['convs'][bottom_up_level].shape} (each)")
        merged_temp = [conv_t]
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['convs'][bottom_up_level])
        merged = concatenate(merged_temp)
        print(F'Merged size: {merged.shape}')

        filters = num_filters * (2 ** (bottom_up_level))
        [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False,
                                              activation=activation,
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        print(F"Filters {filters} Conv (before deconv): {conv_p.shape}")

        if level != (num_levels): # At the end we don't do the upsampling
            conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p)
            print(F"Filters {filters} Conv (after deconv): {conv_t.shape}")

    # ==================== INCREASING THE RESOLUTION HERE=====================
    # Until here is a normal UNET, we need to increase the resolution by inc_res_factor
    # conv_b = Conv2D(filters, (1, 1), activation=activation)(conv_t_u)
    for c_inc_factor in range(int(np.log2(inc_res_factor))):
        conv_t = UpSampling2D((2, 2), interpolation='nearest')(conv_p) # Upsample one level and do 2 CNN
        [conv_p, dele] = multiple_conv_lay_2d(conv_t, filters, filter_size,
                                              tot_layers=2, # How many CNNs we want to do
                                              activation=activation,
                                              batch_norm=batch_norm_decoding,
                                              dropout=dropout_decodign)

    c_output_cnn = 2 # How many cnns we want at the output
    while output_cnn_layers > c_output_cnn:
        [conv_p, dele] = multiple_conv_lay_2d(conv_p, filters, filter_size,
                                              activation=activation,
                                              tot_layers=1, # How many CNNs we want to do
                                              batch_norm=batch_norm_decoding, dropout=dropout_decodign)
        c_output_cnn += 1

    # Last layer one more CNN but with filter size 1
    last_conv = Conv2D(number_output_filters, (1, 1), activation=last_activation)(conv_p)
    print(F"Final shape {last_conv.shape}")

    model = Model(inputs=inputs, outputs=[last_conv])
    return model


def make_multistream_2d_half_unet_for_classification(inputs, num_filters=8, filter_size=3, num_levels=3,
                                                     size_last_layer=10,
                                                     number_of_dense_layers=2,
                                                     batch_norm=True,
                                                     dropout=True, activation='relu', last_activation='softmax'):
    """Makes a 3D-Unet with N number of inputs streams
    :param inputs: An array of tensorflow inputs for example inputs = [Input((10,10,10, 1)), Input((10,10,10,1)]
    :param num_filters: The number of filters to start with, it will double for every new level
    :param filter_size: The size of the kernel filter. It is repeated in all dimensions
    :param num_levels: The number of levels that the U-net will have
    :param batch_norm: Indicates if we are using batch normalization in the encoding phase
    :param dropout: Indicates if we are using dropout in the encoding phase
    :param dropout_decodign: Indicates if we are using dropout in the encoding phase
    :return:
    """

    tot_streams = len(inputs)
    streams = []
    print(F"\n----------- ENCONDING PATH  ----------- ")
    for c_stream in range(tot_streams):
        print(F"----------- Stream {c_stream} ----------- ")
        c_input = inputs[c_stream]
        convs = []
        maxpools = []
        for level in range(num_levels):
            print()
            filters = num_filters * (2 ** level)
            conv_t, pool_t = multiple_conv_lay_2d(c_input, filters, filter_size, make_pool=True,
                                                  batch_norm=batch_norm,
                                                  dropout=dropout, activation=activation)
            print(F"Filters {filters} Conv (before pool): {conv_t.shape} Pool: {pool_t.shape} ")
            convs.append(conv_t)
            maxpools.append(pool_t)
            c_input = maxpools[-1]  # Set the next input as the last output

        streams.append({'convs':convs,'maxpools':maxpools})

    # First merging is special because it is after pooling
    print(F"\n----------- MERGING AT THE BOTTOM  ----------- ")
    print(F"Concatenating previous convs {streams[0]['maxpools'][-1].shape} (each)")
    merged_temp = []
    bottom_up_level = num_levels-1
    # Merging at the bottom
    if tot_streams > 1:
        for cur_stream in range(tot_streams):
            merged_temp.append(streams[cur_stream]['maxpools'][bottom_up_level])
        merged = concatenate(merged_temp)
    else: # This is the single stream case (Default 3D UNet)
        merged = streams[0]['maxpools'][bottom_up_level]

    print(F'Merged size: {merged.shape}')
    # Convolutions at the bottom
    filters = num_filters * (2 ** (num_levels))
    [conv_p, dele] = multiple_conv_lay_2d(merged, filters, filter_size, make_pool=False, batch_norm=batch_norm,
                                          dropout=dropout, activation=activation)
    # Dense layers
    dense_lay = Flatten()(conv_p)  # Flattens all the neurons
    print(F"Size of flatten: {dense_lay.shape} ")
    units = num_filters * (2 ** (num_levels))
    for cur_dense_layer in range(number_of_dense_layers-1):
        print(F"Neurons for dense layer {units} ")
        dense_lay = Dense(filters, activation=activation)(dense_lay)

    final_lay = Dense(size_last_layer, activation=last_activation)(dense_lay)
    print(F"Final number of units: {size_last_layer}")

    model = Model(inputs=inputs, outputs=[final_lay])
    return model


def make_dense_cnn(inputs, num_filters=8, filter_size=3, num_layers=3, batch_norm=True, dropout=True,
                   activation='relu', last_activation='sigmoid'):
    """
    Makes a model with just CNNs one after the other. It can only change the activation function on the last layer
    """
    
    if num_layers > 1:
        inputs, _ = multiple_conv_lay_2d(inputs, num_filters, filter_size, make_pool=False, batch_norm=batch_norm,
                                 dropout=dropout, tot_layers=num_layers-1, activation=activation)

    output, _ = multiple_conv_lay_2d(inputs, num_filters, filter_size, make_pool=False, batch_norm=False,
                                         dropout=False, tot_layers=1, activation=last_activation)

    model = Model(inputs=inputs, outputs=[output])
    return model
