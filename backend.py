from keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.activations import selu
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import UpSampling2D
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
#from tensorflow.python.keras.layers import Lambda
FULL_YOLO_BACKEND_PATH  = "full_yolo_backend.h5"   # should be hosted on a server
TINY_YOLO_BACKEND_PATH  = "tiny_yolo_backend.h5"   # should be hosted on a server
SQUEEZENET_BACKEND_PATH = "squeezenet_backend.h5"  # should be hosted on a server
MOBILENET_BACKEND_PATH  = "mobilenet_backend.h5"   # should be hosted on a server
INCEPTION3_BACKEND_PATH = "inception_backend.h5"   # should be hosted on a server
VGG16_BACKEND_PATH      = "vgg16_backend.h5"       # should be hosted on a server
RESNET50_BACKEND_PATH   = "resnet50_backend.h5"    # should be hosted on a server

class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")       

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)
def layer_norm(x):
       return tf.contrib.layers.layer_norm(x,activation_fn=tf.nn.relu, trainable=True, begin_norm_axis=1, begin_params_axis=-1, center=False, scale=False)
class FullYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = tf.keras.Input(shape=(input_size, input_size, 3))

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)
	#layer = LayerNormalization(axis=1)
        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        #x = BatchNormalization(name='norm_1')(x)
	x = layer_norm(x)
        #x = LeakyReLU(alpha=0.1)(x)
	x = selu(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        #x = BatchNormalization(name='norm_2')(x)
        #x = LeakyReLU(alpha=0.1)(x)
	x = layer_norm(x)
	x = selu(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        #x = BatchNormalization(name='norm_3')(x)
        #x = LeakyReLU(alpha=0.1)(x)
	x = layer_norm(x)
	x = selu(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        #x = BatchNormalization(name='norm_4')(x)
        #x = LeakyReLU(alpha=0.1)(x)
	x = layer_norm(x)
	x = selu(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        #x = BatchNormalization(name='norm_5')(x)
        #x = LeakyReLU(alpha=0.1)(x)
	x = layer_norm(x)
	x = selu(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        #x = BatchNormalization(name='norm_6')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        #x = BatchNormalization(name='norm_7')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        #x = BatchNormalization(name='norm_8')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        #x = BatchNormalization(name='norm_9')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        #x = BatchNormalization(name='norm_10')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        #x = BatchNormalization(name='norm_11')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        #x = BatchNormalization(name='norm_12')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        #x = BatchNormalization(name='norm_13')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        #x = BatchNormalization(name='norm_14')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        #x = BatchNormalization(name='norm_15')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        #x = BatchNormalization(name='norm_16')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        #x = BatchNormalization(name='norm_17')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        #x = BatchNormalization(name='norm_18')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        #x = BatchNormalization(name='norm_19')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        #x = BatchNormalization(name='norm_20')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        #skip_connection = BatchNormalization(name='norm_21')(skip_connection)
	skip_connection = layer_norm(skip_connection)
	skip_connection = selu(skip_connection)
        #skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        #x = BatchNormalization(name='norm_22')(x)
	x = layer_norm(x)
	x = selu(x)
        #x = LeakyReLU(alpha=0.1)(x)
	
        self.feature_extractor = tf.keras.Model(inputs=input_image, outputs=x) 
        #self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

    def normalize(self, image):
        return image / 255.

class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = tf.keras.Input(shape=(input_size, input_size, 3))
# define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"
	#selu	= "selu"

        def fire_module(x, fire_id, squeeze=16, expand=64):
           s_id = 'fire' + str(fire_id) + '/'

           x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
           x     = Activation('relu', name=s_id + relu + sq1x1)(x)

           left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
           left  = Activation('relu', name=s_id + relu + exp1x1)(left)

           right = Conv2D(squeeze,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
           right = Activation('relu', name=s_id + relu + exp3x3)(right)
	   right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
	   x = concatenate([left, right], axis=3, name=s_id + 'concat')
           return x

        
        def conv_batchnorm_relu(x, filters, kernel_size, strides):
            x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x
 
 
        def identity_block(tensor, filters):
            x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
            x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
            x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
            x = BatchNormalization()(x)
 
            x = Add()([x, tensor])
            x = ReLU()(x)
            return x
 
 
        def projection_block(tensor, filters, strides):
        # left stream
            x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
            x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=strides)
            x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)  # notice: filters=4*filters
            x = BatchNormalization()(x)
 
        # right stream
            shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)  # notice: filters=4*filters
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = ReLU()(x)
            return x
 
 
        def resnet_block(x, filters, reps, strides):
            x = projection_block(x, filters=filters, strides=strides)
            for _ in range(reps-1):  # the -1 is because the first block was a Conv one
            x = identity_block(x, filters=filters)
            return x

        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
	#x = Dropout(.2)(x)
	

        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='valid', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
	    x = Dropout(.2)(x)
	    
	# Layer 6
        '''x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
	#x = layer_norm(x)
        x = BatchNormalization(name='norm_6')(x)
	#x = selu(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)     
	#x = Dropout(.5)(x)
	#x = Dense(16)(x)'''
        

        # Layer 7 - 8

	'''for i in range(0,2):
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
	    #x = layer_norm(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
	    #x = selu(x)
            x = LeakyReLU(alpha=0.1)(x)
	    #x  = sigmoid(x)'''

	x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
        x = resnet_block(x, filters=64, reps=3, strides=1)
        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
        x = resnet_block(x, filters=128, reps=4, strides=2)
	

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)
        x = Dropout(.5)(x)

        self.feature_extractor = tf.keras.Model(inputs=input_image, outputs=x)  
        #self.feature_extractor.load_weights(TINY_YOLO_BACKEND_PATH)

    def normalize(self, image):
        return image / 255.

class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        mobilenet = MobileNet(input_shape=(224,224,3), include_top=False)
        #mobilenet.load_weights(MOBILENET_BACKEND_PATH)

        x = mobilenet(input_image)

        self.feature_extractor = tf.keras.Model(inputs=input_image, outputs=x) 

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image		

class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = tf.keras.Model(inputs=input_image, outputs=x)
        #self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image    

class Inception3Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        inception = InceptionV3(input_shape=(input_size,input_size,3), include_top=False)
        inception.load_weights(INCEPTION3_BACKEND_PATH)

        x = inception(input_image)
	print(x)
	print(input_image.shape)

        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image

class VGG16Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        vgg16 = VGG16(input_shape=(input_size, input_size, 3), include_top=False)
        #vgg16.load_weights(VGG16_BACKEND_PATH)

        self.feature_extractor = vgg16

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image 

class ResNet50Feature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):
        resnet50 = ResNet50(input_shape=(input_size, input_size, 3), include_top=False)
        resnet50.layers.pop() # remove the average pooling layer
        #resnet50.load_weights(RESNET50_BACKEND_PATH)

        #self.feature_extractor = Model(resnet50.layers[0].input, resnet50.layers[-1].output)
	self.feature_extractor = resnet50

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image 
