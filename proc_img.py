import numpy as np

def preprocess_image(image_path, load_dims=False, read_mode="color", maintain_aspect_ratio=False):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == "color" else "L"
    img = imread(image_path, mode=mode)  # Prevents crashes due to PNG images (ARGB)

    if mode == "L":
        # Expand the 1 channel grayscale to 3 channel grayscale image
        temp_img = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp_img[:, :, 0] = img
        temp_img[:, :, 1] = img.copy()
        temp_img[:, :, 2] = img.copy()

        img = temp_img

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = args.img_size
        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = args.img_size

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



base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))

style_reference_images = []
for style_path in style_image_paths:
    style_reference_images.append(K.variable(preprocess_image(style_path)))

# this will contain our generated image
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

ip = Input(tensor=input_tensor, shape=shape)


    
