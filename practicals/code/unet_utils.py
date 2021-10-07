import numpy as np
from PIL import Image
import gryds
#import time
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import extract_patches_2d

def load_data(impaths_all, test=False):
    """
    Load data with corresponding masks and segmentations

    :param impaths_all: Paths of images to be loaded
    :param test: Boolean, part of test set?
    :return: Numpy array of images, masks and segmentations
    """
    # Save all images, masks and segmentations
    images = []
    masks = []
    segmentations = []

    # Load as numpy array and normalize between 0 and 1
    for im_path in impaths_all:
        images.append(np.array(Image.open(im_path)) / 255.)
        mask_path = im_path.replace('images', 'mask').replace('.png', '_mask.gif')
        masks.append(np.array(Image.open(mask_path)) / 255.)
        if not test:
            seg_path = im_path.replace('images', '1st_manual').replace('training.png', 'manual1.gif')
        else:
            seg_path = im_path.replace('images', '1st_manual').replace('test.png', 'manual1.gif')
        segmentations.append(np.array(Image.open(seg_path)) / 255.)
    return np.array(images), np.expand_dims(np.array(masks), axis=-1), np.expand_dims(np.array(segmentations), axis=-1)


def pad_image(image, desired_shape):
    """
    Pad image to square

    :param image: Input image
    :param desired_shape: Desired shape of padded image
    :return: Padded image
    """
    padded_image = np.zeros((desired_shape[0], desired_shape[1], image.shape[-1]), dtype=image.dtype)
    pad_val_x = desired_shape[0] - image.shape[0]
    pad_val_y = desired_shape[1] - image.shape[1]
    padded_image[int(np.ceil(pad_val_x / 2)):padded_image.shape[0]-int(np.floor(pad_val_x / 2)),
                 int(np.ceil(pad_val_y / 2)):padded_image.shape[0]-int(np.floor(pad_val_y / 2)), :] = image
    return padded_image


# Pad to squares
def preprocessing(images, masks, segmentations, desired_shape):
    """
    Pad all images, masks and segmentations to desired shape

    :param images: Numpy array of images
    :param masks: Numpy array of masks
    :param segmentations: Numpy array of segmentations
    :param desired_shape: Desired shape of padded image
    :return: Padded images, masks and segmentations
    """
    padded_images = []
    padded_masks = []
    padded_segmentations = []
    for im, mask, seg in zip(images, masks, segmentations):
        padded_images.append(pad_image(im, desired_shape))
        padded_masks.append(pad_image(mask, desired_shape))
        padded_segmentations.append(pad_image(seg, desired_shape))

    return np.array(padded_images), np.array(padded_masks), np.array(padded_segmentations)


def extract_patches(images, segmentations, patch_size, patches_per_im, seed):
    """
    Extract patches from images

    :param images: Input images
    :param segmentations: Corresponding segmentations
    :param patch_size: Desired patch size
    :param patches_per_im: Amount of patches to extract per image
    :param seed: Random seed to ensure matching patches between image and segmentation
    :return: x: numpy array of patches and y: numpy array of patches segmentations
    """
    # The total amount of patches that will be obtained
    inp_size = len(images) * patches_per_im
    # Allocate memory for the patches and segmentations of the patches
    x = np.zeros((inp_size, patch_size[0], patch_size[1], images.shape[-1]))
    y = np.zeros((inp_size, patch_size[0], patch_size[1], segmentations.shape[-1]))

    # Loop over all the images (and corresponding segmentations) and extract random patches
    # using the extract_patches_2d function of scikit learn
    for idx, (im, seg) in enumerate(zip(images, segmentations)):
        # Note the random seed to ensure the corresponding segmentation is extracted for each patch
        x[idx * patches_per_im:(idx + 1) * patches_per_im] = extract_patches_2d(im, patch_size,
                                                                                max_patches=patches_per_im,
                                                                                random_state=seed)
        y[idx * patches_per_im:(idx + 1) * patches_per_im] = np.expand_dims(
            extract_patches_2d(seg, patch_size, max_patches=patches_per_im, random_state=seed),
            axis=-1)

    return x, y


# Create a very simple datagenerator
def datagenerator(images, segmentations, patch_size, patches_per_im, batch_size):
    """
    Simple data-generator to feed patches in batches to the network.
    To extract different patches each epoch, steps_per_epoch in fit_generator should be equal to nr_batches.

    :param images: Input images
    :param segmentations: Corresponding segmentations
    :param patch_size: Desired patch size
    :param patches_per_im: Amount of patches to extract per image
    :param batch_size: Number of patches per batch
    :return: Batch of patches to feed to the model
    """
    # Total number of patches generated per epoch
    total_patches = len(images) * patches_per_im
    # Amount of batches in one epoch
    nr_batches = int(np.ceil(total_patches / batch_size))

    while True:
        # Each epoch extract different patches from the training images
        x, y = extract_patches(images, segmentations, patch_size, patches_per_im, seed=np.random.randint(0, 500))

        # Feed data in batches to the network
        for idx in range(nr_batches):
            x_batch = x[idx * batch_size:(idx + 1) * batch_size]
            y_batch = y[idx * batch_size:(idx + 1) * batch_size]
            yield x_batch, y_batch


# Data augmentation
def brightness_offset(images, masks, segs, offset_range, nr_augmentations):
    aug_images = np.zeros((nr_augmentations, images.shape[1], images.shape[2], images.shape[3]))
    aug_masks  = np.zeros((nr_augmentations, masks.shape[1], masks.shape[2], masks.shape[3]))
    aug_segms  = np.zeros((nr_augmentations, segs.shape[1], segs.shape[2], segs.shape[3]))

    for i in range(nr_augmentations):
        offset = np.random.uniform(offset_range[0], offset_range[1])
        image_id = np.random.randint(len(images))
        new_image = images[image_id] + offset
        aug_images[i, :, :, :] = new_image
        aug_segms[i, :, :, :]  = segs[image_id]
        aug_masks[i, :, :, :]  = masks[image_id]

    return np.concatenate((images, aug_images), axis=0), \
        np.concatenate((masks, aug_masks), axis=0), \
        np.concatenate((segs, aug_segms), axis=0)


# def brightness_offset(images, masks, segs, offset_range, nr_augmentations):
#     aug_images = np.zeros((nr_augmentations, images.shape[1], images.shape[2], images.shape[3]))
#     aug_masks = np.zeros((nr_augmentations, masks.shape[1], masks.shape[2], masks.shape[3]))
#     aug_segms = np.zeros((nr_augmentations, segs.shape[1], segs.shape[2], segs.shape[3]))

#     for i in range(nr_augmentations):
#         offset = np.random.uniform(offset_range[0], offset_range[1])
#         image_id = np.random.randint(len(images))
#         new_image = images[image_id] + offset
#         aug_images[i, :, :, :] = new_image
#         aug_segms[i, :, :, :] = segs[image_id]
#         aug_masks[i, :, :, :] = masks[image_id]

#     return np.concatenate((images, aug_images), axis=0), \
#            np.concatenate((masks, aug_masks), axis=0), \
#            np.concatenate((segs, aug_segms), axis=0)

def bspline_brightness_offset(images, masks, segs, offset_range, nr_augmentations):
    aug_images = np.zeros((nr_augmentations, images.shape[1], images.shape[2], images.shape[3]))
    aug_masks = np.zeros((nr_augmentations, masks.shape[1], masks.shape[2], masks.shape[3]))
    aug_segms = np.zeros((nr_augmentations, segs.shape[1], segs.shape[2], segs.shape[3]))

    for i in range(nr_augmentations):
        # Load image number to be augmented
        image_id = np.random.randint(len(images))

        # Create empty original size variables
        transf_image = np.zeros_like(images[image_id])

        # Random 3x3 B-spline grid for a 2D image
        random_grid = np.random.rand(2, 3, 3) # Random 3x3 between [0,1]
        random_grid -= 0.5 # make random between -0.5 and 0.5
        random_grid /= 5 # random between -0.1 and 0.1 (i.e. max 10% deformation in all directions)

        # Define a B-spline transformation object
        bspline = gryds.BSplineTransformation(random_grid)

        # Define an interpolator object for the image (per color channel):
        interpolator_image1 = gryds.Interpolator(images[image_id,:,:,0])
        interpolator_image2 = gryds.Interpolator(images[image_id,:,:,1])
        interpolator_image3 = gryds.Interpolator(images[image_id,:,:,2])
        interpolator_seg = gryds.Interpolator(segs[image_id,:,:,0])
        interpolator_mask = gryds.Interpolator(masks[image_id,:,:,0])

        # Transform the image using the B-spline transformation (Per color channel)
        transf_image1 = interpolator_image1.transform(bspline)
        transf_image2 = interpolator_image2.transform(bspline)
        transf_image3 = interpolator_image3.transform(bspline)
        transf_seg = interpolator_seg.transform(bspline)
        transf_mask = interpolator_mask.transform(bspline)

        # Recombine color channels
        transf_image[:,:,0], transf_image[:,:,1], transf_image[:,:,2] = transf_image1, transf_image2, transf_image3

        # Add brightness offset only to image
        offset = np.random.uniform(offset_range[0], offset_range[1])
        new_image = transf_image + offset
        aug_images[i, :, :, :] = new_image
        aug_segms[i, :, :, :] = transf_seg[:,:, np.newaxis]
        aug_masks[i, :, :, :] = transf_mask[:,:, np.newaxis]

    return np.concatenate((images, aug_images), axis=0), \
           np.concatenate((masks, aug_masks), axis=0), \
           np.concatenate((segs, aug_segms), axis=0)
