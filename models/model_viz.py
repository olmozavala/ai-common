import numpy as np
import pylab
from img_viz.common import create_folder
from matplotlib import cm
import matplotlib.pyplot as plt
from constants.AI_params import CNNTypes
from os.path import join

def print_layer_names(model):
    for cur_lay in model.layers:
        print(F'Name: {cur_lay.name}  # of params: {cur_lay.count_params()}')


def plot_cnn_filters_by_layer(layer, title='', cnntype=CNNTypes.TwoD, filter_per_row = 4):
    '''
    Plots the filters of an specific layer
    :param layer:
    :param title:
    :return:
    '''

    layer_weights = np.array(layer.get_weights()[0])
    tot_filters = layer_weights.shape[-1]
    tot_bands = layer_weights.shape[-2]

    print(F' Shape of filters in this layer is {layer_weights.shape}')
    # print(layer_weights)
    print(F' Number of filters {tot_filters}')
    print(F' Number of bands {tot_bands}')

    tot_rows = int(np.ceil(tot_filters/filter_per_row))
    if tot_rows > 1:
        tot_cols = filter_per_row
    else:
        tot_cols = tot_filters

    norm = cm.colors.Normalize(vmax=np.amax(layer_weights), vmin=np.amin(layer_weights))
    for idx_band in range(tot_bands):
        fig = plt.figure(figsize=(4*tot_cols,4*tot_rows))
        for idx_filter in range(tot_filters):
            if cnntype == CNNTypes.TwoD:
                cur_filt = layer_weights[:, :, idx_band, idx_filter]
            elif cnntype == CNNTypes.ThreeD:
                cur_filt = layer_weights[:,:,:,idx_band, idx_filter]

            ftitle = F'{title} filter:{idx_filter}  band:{idx_band}'
            fig.suptitle(ftitle)
            ax = fig.add_subplot(tot_rows,tot_cols, idx_filter+1)
            # ax.matshow(cur_filt)
            n_cur_filt = norm(cur_filt)
            # ax.imshow(n_cur_filt, cmap='gray')
            ax.imshow(cur_filt, cmap='gray')
            ax.axis('off')
        # plt.tight_layout()
        plt.show()
        fig.clear()


def plot_intermediate_3dcnn_feature_map(layer_output, slices, title=''):
    '''Plots an intermediate output'''

    all_dims = layer_output.shape
    tot_filters = all_dims[-1]
    filter_per_row = 4

    tot_rows = int(np.ceil(tot_filters / filter_per_row))
    if tot_rows > 1:
        tot_cols = filter_per_row
    else:
        tot_cols = tot_filters

    if slices == 'middle':
        slices = [int(layer_output.shape[1] / 2)]

    for cur_slice in slices:
        fig = plt.figure(figsize=(4 * tot_cols, 4 * tot_rows))
        for cur_filt_idx in range(tot_filters):
            ax = fig.add_subplot(tot_rows, tot_cols, cur_filt_idx + 1)
            cur_filt = layer_output[0, cur_slice, :, :, cur_filt_idx]
            ax.imshow(cur_filt, cmap='gray')
            ax.axis('off')

        plt.title(title)
        plt.tight_layout()
        plt.show()
        fig.clear()


def plot_intermediate_2dcnn_feature_map(layer_output, input_data=None, desired_output_data=None, nn_output_data=None,
                                        input_fields=None, title='', file_name='', output_folder='', disp_images=True):
    """
    This function is used to plot the output of hidden CNN layers.
    :param layer_output: The output of the hidden layer
    :param input_data:  If we want to plot also the input data of the input layer
    :param desired_output_data:  If we want to plot also the desired output data of the last layer
    :param output_data_nn:  If we want to plot also the obtained output from nn
    :param title: Title of the plot
    :param file_name:  File name to save the figure
    :param output_folder:  Output folder
    :param disp_images:  Do we want to display images or not
    :return:
    """
    all_dims = layer_output.shape
    tot_filters = all_dims[-1]
    fig_size = 6
    filter_per_row = 4
    input_bands = 0
    output_classes = 0
    output_nn_classes = 0

    # If input_data and desired_output_data are not None we take them into account to define number of rows and cols
    if input_data is not None:
        input_bands = input_data.shape[-1]
        filter_per_row = input_bands
    if desired_output_data is not None:
        output_classes = desired_output_data.shape[-1]
    if nn_output_data is not None:
        output_nn_classes = nn_output_data.shape[-1]

    tot_rows = int(np.ceil( (tot_filters + input_bands + output_classes + output_nn_classes)/ filter_per_row))

    if tot_rows > 1:
        tot_cols = filter_per_row
    else:
        tot_cols = tot_filters + input_bands + output_classes + output_nn_classes

    fig, axs = plt.subplots(squeeze=True, figsize=(fig_size * tot_cols, fig_size * tot_rows), ncols=tot_cols)
    cur_ax_id = 0

    # ============== Plot Input bands ============
    for cur_ax_id in range(input_bands):
        ax = plt.subplot(tot_rows, tot_cols, cur_ax_id + 1)
        cur_band = input_data[0, :, :, cur_ax_id]
        ax.imshow(cur_band)
        if input_fields is not None:
            ax.set_title(input_fields[cur_ax_id])
        ax.axis('off')

    if input_data is not None:
        cur_ax_id += 1

    # ============== Plot hidden layer filters output============
    for cur_filt_idx in range(tot_filters):
        ax = plt.subplot(tot_rows, tot_cols, cur_ax_id + cur_filt_idx + 1)
        cur_filt = layer_output[0, :, :, cur_filt_idx]
        ax.imshow(cur_filt, cmap='gray')
        ax.set_title(F"{cur_filt.shape}")
        ax.axis('off')

    if desired_output_data is not None:
        cur_ax_id += cur_filt_idx + 1

    # ============== Plot desired output============
    for cur_output_id in range(output_classes):
        ax = plt.subplot(tot_rows, tot_cols, cur_ax_id + cur_output_id + 1)
        cur_band = desired_output_data[0, :, :, cur_output_id]
        ax.imshow(cur_band)
        ax.set_title(F"Desired output")
        ax.axis('off')

    if nn_output_data is not None:
        cur_ax_id += cur_output_id + 1

    # ============== Plot NN output============
    for cur_output_id in range(output_classes):
        ax = plt.subplot(tot_rows, tot_cols, cur_ax_id + cur_output_id + 1)
        cur_band = nn_output_data[0, :, :, cur_output_id]
        ax.imshow(cur_band)
        ax.set_title(F"NN output")
        ax.axis('off')

    fig.suptitle(title, fontsize=30)
    if output_folder != '':
        create_folder(output_folder)
        pylab.savefig(join(output_folder, F'{file_name}.png'), bbox_inches='tight')

    if disp_images:
        plt.show()
    plt.close()
