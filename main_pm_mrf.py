
from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import scipy.interpolate
import scipy.ndimage
import numpy as np
import time
import argparse
import warnings
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('style_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")






def _calc_patch_grid_dims(shape, patch_size, patch_stride):
    x_w, x_h, x_c = shape
    num_rows = 1 + (x_h - patch_size) // patch_stride
    num_cols = 1 + (x_w - patch_size) // patch_stride
    return num_rows, num_cols


def make_patch_grid(x, patch_size, patch_stride=1):
    '''x shape: (num_channels, rows, cols)'''
    x = x.transpose(2, 1, 0)
    patches = extract_patches_2d(x, (patch_size, patch_size))
    x_w, x_h, x_c  = x.shape
    num_rows, num_cols = _calc_patch_grid_dims(x.shape, patch_size, patch_stride)
    patches = patches.reshape((num_rows, num_cols, patch_size, patch_size, x_c))
    patches = patches.transpose((0, 1, 4, 2, 3))
    #patches = np.rollaxis(patches, -1, 2)
    return patches


def combine_patches_grid(in_patches, out_shape):
    '''Reconstruct an image from these `patches`
    input shape: (rows, cols, channels, patch_row, patch_col)
    '''
    num_rows, num_cols = in_patches.shape[:2]
    num_channels = in_patches.shape[-3]
    patch_size = in_patches.shape[-1]
    num_patches = num_rows * num_cols
    in_patches = np.reshape(in_patches, (num_patches, num_channels, patch_size, patch_size))  # (patches, channels, pr, pc)
    in_patches = np.transpose(in_patches, (0, 2, 3, 1)) # (patches, p, p, channels)
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon.transpose(2, 1, 0)


class PatchMatcher(object):
    '''A matcher of image patches inspired by the PatchMatch algorithm.
    image shape: (width, height, channels)
    '''
    def __init__(self, input_shape, target_img, patch_size=1, patch_stride=1, jump_size=0.5,
            num_propagation_steps=5, num_random_steps=5, random_max_radius=1.0, random_scale=0.5):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.jump_size = jump_size
        self.num_propagation_steps = num_propagation_steps
        self.num_random_steps = num_random_steps
        self.random_max_radius = random_max_radius
        self.random_scale = random_scale
        self.num_input_rows, self.num_input_cols = _calc_patch_grid_dims(input_shape, patch_size, patch_stride)
        self.target_patches = make_patch_grid(target_img, patch_size)
        self.target_patches_normed = self.normalize_patches(self.target_patches)
        self.coords = np.random.uniform(0.0, 1.0,  # TODO: switch to pixels
            (2, self.num_input_rows, self.num_input_cols))# * [[[self.num_input_rows]],[[self.num_input_cols]]]
        self.similarity = np.zeros(input_shape[:2:-1], dtype ='float32')
        self.min_propagration_row = 1.0 / self.num_input_rows
        self.min_propagration_col = 1.0 / self.num_input_cols
        self.delta_row = np.array([[[self.min_propagration_row]], [[0.0]]])
        self.delta_col = np.array([[[0.0]], [[self.min_propagration_col]]])

    def update(self, input_img, reverse_propagation=False):
        input_patches = self.get_patches_for(input_img)
        self.update_with_patches(self.normalize_patches(input_patches), reverse_propagation=reverse_propagation)

    def update_with_patches(self, input_patches, reverse_propagation=False):
        self._propagate(input_patches, reverse_propagation=reverse_propagation)
        self._random_update(input_patches)

    def get_patches_for(self, img):
        return make_patch_grid(img, self.patch_size)

    def normalize_patches(self, patches):
        norm = np.sqrt(np.sum(np.square(patches), axis=(2, 3, 4), keepdims=True))
        return patches / norm

    def _propagate(self, input_patches, reverse_propagation=False):
        if reverse_propagation:
            roll_direction = 1
        else:
            roll_direction = -1
        sign = float(roll_direction)
        for step_i in range(self.num_propagation_steps):
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 1) + self.delta_row * sign)
            coords_row, similarity_row = self.eval_state(new_coords, input_patches)
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 2) + self.delta_col * sign)
            coords_col, similarity_col = self.eval_state(new_coords, input_patches)
            self.coords, self.similarity = self.take_best(coords_row, similarity_row, coords_col, similarity_col)

    def _random_update(self, input_patches):
        for alpha in range(1, self.num_random_steps + 1):  # NOTE this should actually stop when the move is < 1
            new_coords = self.clip_coords(self.coords + np.random.uniform(-self.random_max_radius, self.random_max_radius, self.coords.shape) * self.random_scale ** alpha)
            self.coords, self.similarity = self.eval_state(new_coords, input_patches)

    def eval_state(self, new_coords, input_patches):
        new_similarity = self.patch_similarity(input_patches, new_coords)
        delta_similarity = new_similarity - self.similarity
        coords = np.where(delta_similarity > 0, new_coords, self.coords)
        best_similarity = np.where(delta_similarity > 0, new_similarity, self.similarity)
        return coords, best_similarity

    def take_best(self, coords_a, similarity_a, coords_b, similarity_b):
        delta_similarity = similarity_a - similarity_b
        best_coords = np.where(delta_similarity > 0, coords_a, coords_b)
        best_similarity = np.where(delta_similarity > 0, similarity_a, similarity_b)
        return best_coords, best_similarity

    def patch_similarity(self, source, coords):
        '''Check the similarity of the patches specified in coords.'''
        target_vals = self.lookup_coords(self.target_patches_normed, coords)
        err = source * target_vals
        return np.sum(err, axis=(2, 3, 4))

    def clip_coords(self, coords):
        # TODO: should this all be in pixel space?
        coords = np.clip(coords, 0.0, 1.0)
        return coords

    def lookup_coords(self, x, coords):
        x_shape = np.expand_dims(np.expand_dims(x.shape, -1), -1)
        i_coords = np.round(coords * (x_shape[:2] - 1)).astype('int32')
        return x[i_coords[0], i_coords[1]]

    def get_reconstruction(self, patches=None, combined=None):
        if combined is not None:
            patches = make_patch_grid(combined, self.patch_size)
        if patches is None:
            patches = self.target_patches
        patches = self.lookup_coords(patches, self.coords)
        recon = combine_patches_grid(patches, self.input_shape)
        return recon

    def scale(self, new_shape, new_target_img):
        '''Create a new matcher of the given shape and replace its
        state with a scaled up version of the current matcher's state.
        '''
        new_matcher = PatchMatcher(new_shape, new_target_img, patch_size=self.patch_size,
                patch_stride=self.patch_stride, jump_size=self.jump_size,
                num_propagation_steps=self.num_propagation_steps,
                num_random_steps=self.num_random_steps,
                random_max_radius=self.random_max_radius,
                random_scale=self.random_scale)
        new_matcher.coords = congrid(self.coords, new_matcher.coords.shape, method='neighbour')
        new_matcher.similarity = congrid(self.similarity, new_matcher.coords.shape, method='neighbour')
        return new_matcher


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print ("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = [i for i in range(np.rank(newcoords))]
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None








args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_paths = args.style_image_paths
style_image_paths = [path for path in args.style_image_paths]
result_prefix = args.result_prefix
content_weight = args.content_weight
total_variation_weight = args.tv_weight

scale_sizes = []
size = args.img_size
while size > 64:
    scale_sizes.append(size/2)
    size /= 2

img_width = img_height = 0

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

read_mode = "color"
style_weights = []
if len(style_image_paths) != len(args.style_weight):
    weight_sum = sum(args.style_weight) * args.style_scale
    count = len(style_image_paths)

    for i in range(len(style_image_paths)):
        style_weights.append(weight_sum / count)
else:
    style_weights = [weight*args.style_scale for weight in args.style_weight]

def pooling_func(x):
    # return AveragePooling2D((2, 2), strides=(2, 2))(x)
    return MaxPooling2D((2, 2), strides=(2, 2))(x)

#start proc_img

def preprocess_image(image_path, sc_size=args.img_size, load_dims=False):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB"
    # mode = "RGB" if read_mode == "color" else "L"
    img = imread(image_path, mode=mode)  # Prevents crashes due to PNG images (ARGB)

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = sc_size
        img_height = int(img_width * aspect_ratio)

    img = imresize(img, (img_width, img_height)).astype('float32')

    # RGB -> BGR
    img = img[:, :, ::-1]

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68


    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

combination_prev = ""

for scale_size in scale_sizes:
    base_image = K.variable(preprocess_image(base_image_path, scale_size, True))

    style_reference_images = [K.variable(preprocess_image(path)) for path in style_image_paths]

    # this will contain our generated image
    if combination_prev != "":
        combination_image = imresize(combination_prev, (img_width, img_height), interp="bilinear").astype('float32')
    else:
        combination_image = K.placeholder((1, img_width, img_height, 3)) # tensorflow

    image_tensors = [base_image]
    for style_image_tensor in style_reference_images:
        image_tensors.append(style_image_tensor)
    image_tensors.append(combination_image)

    nb_tensors = len(image_tensors)
    nb_style_images = nb_tensors - 2 # Content and Output image not considered

    # combine the various images into a single Keras tensor
    input_tensor = K.concatenate(image_tensors, axis=0)

    shape = (nb_tensors, img_width, img_height, 3) #tensorflow


    #build the model
    model_input = Input(tensor=input_tensor, shape=shape)

    # build the VGG16 network with our 3 images as input
    x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1', border_mode='same')(model_input)
    x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same')(x)
    x = pooling_func(x)

    x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same')(x)
    x = pooling_func(x)

    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same')(x)
    x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same')(x)
    x = pooling_func(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same')(x)
    x = pooling_func(x)

    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_1', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_2', border_mode='same')(x)
    x = Convolution2D(512, 3, 3, activation='relu', name='conv5_3', border_mode='same')(x)
    x = pooling_func(x)

    model = Model(model_input, x)

    weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

    print("Weights Path: ", weights)

    model.load_weights(weights)

    print('Model loaded.')

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])


    # compute the neural style loss
    # first we need to define 4 util functions

    # the gram matrix of an image tensor (feature-wise outer product)
    def gram_matrix(x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram


    # the 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent
    def total_variation_loss(x):
        assert K.ndim(x) == 4
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))


    def mrf_loss(style, combination, patch_size=3, patch_stride=1):
        # extract patches from feature maps with PatchMatch algorithm
        style_patches = style_pmatcher.get_patches_for(style)
        style_patches_norm = style_pmatcher.normalize_patches(style)
        combination_patches = style_pmatcher.get_patches_for(style)
        #    style_patches, style_patches_norm = make_patches(style, patch_size, patch_stride)
        style_pmatcher.update(style, True)
        patch_coords = style_pmatcher.coords()
        best_style_patches = K.reshape(patch_coords, K.shape(style_patches))
        loss = K.sum(K.square(best_style_patches - combination_patches)) / patch_size ** 2
        return loss

    # an auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image
    def content_loss(base, combination):
        channels = K.shape(base)[-1]
        size = img_width * img_height

        if args.content_loss_type == 1:
            multiplier = 1 / (2. * channels ** 0.5 * size ** 0.5)
        elif args.content_loss_type == 2:
            multiplier = 1 / (channels * size)
        else:
            multiplier = 1.

        return multiplier * K.sum(K.square(combination - base))

    # combine these loss functions into a single scalar
    loss = K.variable(0.)
    layer_features = outputs_dict[args.content_layer]  # 'conv5_2' or 'conv4_2'
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[nb_tensors - 1, :, :, :]
    loss += content_weight * content_loss(base_image_features,
                                          combination_features)

    channel_index = -1

    #Style Loss calculation
    mrf_layers = ['conv3_1', 'conv4_1']
    # feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for layer_name in mrf_layers:
        output_features = outputs_dict[layer_name]
        shape = shape_dict[layer_name]
        combination_features = output_features[nb_tensors - 1, :, :, :]

        style_features = output_features[1:nb_tensors - 1, :, :, :]
        sl = []
        for j in range(nb_style_images):
            sl.append(mrf_loss(style_features[j], combination_features))
        for j in range(nb_style_images):
            loss += (style_weights[j] / len(mrf_layers)) * sl[j]

    loss += total_variation_weight * total_variation_loss(combination_image)

    # get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, combination_image)

    outputs = [loss]
    if type(grads) in {list, tuple}:
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([combination_image], outputs)


    def eval_loss_and_grads(x):
        x = x.reshape((1, img_width, img_height, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


    # # this Evaluator class makes it possible
    # # to compute loss and gradients in one pass
    # # while retrieving them via two separate functions,
    # # "loss" and "grads". This is done because scipy.optimize
    # # requires separate functions for loss and gradients,
    # # but computing them separately would be inefficient.
    class Evaluator(object):
        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values


    evaluator = Evaluator()

    # (L-BFGS)

    if "content" in args.init_image or "gray" in args.init_image:
        x = preprocess_image(base_image_path, True)
    elif "noise" in args.init_image:
        x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

        if K.image_dim_ordering() == "th":
            x = x.transpose((0, 3, 1, 2))
    else:
        print("Using initial image : ", args.init_image)
        x = preprocess_image(args.init_image)

    num_iter = args.num_iter
    prev_min_val = -1

    for i in range(num_iter):
        print("Starting iteration %d of %d" % ((i + 1), num_iter))
        start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        combination_prev = x

        if prev_min_val == -1:
            prev_min_val = min_val

        improvement = (prev_min_val - min_val) / prev_min_val * 100

        print('Current loss value:', min_val, " Improvement : %0.3f" % improvement, "%")
        prev_min_val = min_val
        # save current generated image
        img = deprocess_image(x.copy())

        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp="bilinear")

        fname = result_prefix + '_at_iteration_%d.png' % (i + 1)
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))
