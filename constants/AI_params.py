from enum import Enum


class AiModels(Enum):
    # ====== UNET ============
    UNET_3D_SINGLE = 3
    UNET_3D_3_STREAMS = 4
    UNET_2D_SINGLE = 20
    UNET_2D_MULTISTREAMS = 21

    # ======= Classification UNET ==============
    HALF_UNET_2D_SINGLE_STREAM_CLASSIFICATION = 5
    HALF_UNET_3D_CLASSIFICATION_3_STREAMS = 6
    HALF_UNET_3D_CLASSIFICATION_SINGLE_STREAM = 7

    # ======= Superresolution UNET ==============
    UNET_2D_SINGLE_SUPERRES = 50
    PRE_UPSAMPLING_SUPERRES = 51
    SRCNN = 52
    SSResNets= 52

    # ======= For profiles CNN and RNN ==============
    MULTISTREAM_CNN_RNN = 100

    # =======  Dense CNN ========
    DENSE_CNN = 30
    
    # =======  Dense NN ========
    ML_PERCEPTRON = 10

class ModelParams(Enum):
    DROPOUT = 1  # If we should use Dropout on the NN
    BATCH_NORMALIZATION = 2  # If we are using Batch Normalization
    MODEL = 4
    INPUT_SIZE = 7
    OUTPUT_SIZE = 20
    START_NUM_FILTERS = 8  # How many filters are we using on the first layer and level
    NUMBER_LEVELS = 9  # Number of levels in the network (works for U-Net
    FILTER_SIZE = 10  # The size of the filters in each layer (currently is the same for each layer)

    # ========= For Classification ===============
    NUMBER_DENSE_LAYERS = 11  # Used in 2D
    NUMBER_OF_OUTPUT_CLASSES = 12  # Used in 2D

    # ========= For Superresolution ===============
    INC_RES_FACTOR = 30 # By how much we want to increase the resolution (must be a power of 2)
    INTERPOLATION_METHOD = 31 # It can be 'nearest','bicubic'
    PREDICTION_TYPE = 32 # If we want to predict 'hr' or 'diff' which means only the difference

    # ========= For 1D Multilayer perceptron===============
    HIDDEN_LAYERS = 13
    CELLS_PER_HIDDEN_LAYER = 14
    ACTIVATION_HIDDEN_LAYERS = 15
    ACTIVATION_OUTPUT_LAYERS = 16

class TrainingParams(Enum):
    # These are common training parameters
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    cases = 5  # A numpy array of the cases of interest or 'all'
    validation_percentage = 6
    test_percentage = 7
    evaluation_metrics = 10
    loss_function = 11
    batch_size = 12
    epochs = 13
    config_name = 14  # A name that allows you to identify the configuration of this training
    optimizer = 15
    data_augmentation = 16
    normalization_type = 17
    # ============ These parameters are for images (2D or 3D) ============
    output_imgs_folder = 50  # Where to store intermediate images
    show_imgs = 4  # If we want to display the images while are being generated (for PyCharm)
    # ============ These parameters are for segmentation trainings ============
    image_file_names = 8
    ctr_file_names = 9
    # ============ These parameters are for classification ============
    class_label_file_name = 20
    # ============ These parameters are for 1D approximation ============
    file_name = 30

class NormParams(Enum):
    min_max = 1
    mean_zero = 2

class ClassificationParams(Enum):
    input_folder = 1  # Where the images are stored
    output_folder = 2  # Where to store the segmented contours
    output_imgs_folder = 3  # Where to store intermediate images
    output_file_name = 4
    show_imgs = 5  # If we want to display the images while are being generated (for PyCharm)
    cases = 6  # A numpy array of the cases of interest or 'all'
    save_segmented_ctrs = 7  # Boolean that indicates if we need to save the segmentations
    model_weights_file = 8  # Which model weights file are we going to use
    # Indicates that we need to resample everything to the original resolution. If that is the case
    compute_original_resolution = 9
    resampled_resolution_image_name = 55  # Itk image name of a resampled resolution (for metrics in original size)
    original_resolution_image_name = 56  # Itk image name of a original resolution (for metrics in original size)
    original_resolution_ctr_name = 57  # Itk image name of a original resolution (for metrics in original size)
    metrics = 10
    segmentation_type = 11
    compute_metrics = 12  # This means we have the GT ctrs
    output_ctr_file_names = 13  # Only used if we are computing metrics
    input_img_file_names = 15  # Name of the images to read and use as input for the NN
    save_imgs = 16  # Indicates if we want to save images from the segmented contours
    save_img_slices = 17  # IF we are saving the images, it indicates which slices to save
    save_img_planes = 18  # IF we are saving the images, it indicates which plane to save
    split_file = 30  # This is the file that indicates how the split was made (training, validation, test)
#     ================== For Time series =============
    input_file = 19
    training_data_file= 20 # Points to the file used for training
    save_prediction = 21
    generate_images = 22  # Indicate if we are itnerested in creating the images (normally yes)

class VisualizationResultsParams(Enum):
    gt_data_file = 1
    nn_output =  2
    nn_metrics =  3

class CNNTypes(Enum):
    TwoD = 1
    ThreeD = 2

# ================= From original code related to Medical imaging, may be deprecatded ==========
class SubstractionParams(Enum):
    # This is what is being used to compute TZ. It uses two contours, compute the difference and obtain its DSC
    model_weights_file = 0
    ctr_file_name = 1

class ClassificationMetrics(Enum):
    DSC_3D = '3D_DSC'  # DSC in 3D
    DSC_2D = '2D_DSC'  # DSC in 3D
    MSE = 'MSE'

