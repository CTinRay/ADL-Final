import argparse
import numpy as np
import os
from keras import backend as K
from keras.models import load_model
import pdb
import sys
import traceback
import matplotlib.pyplot as plt

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


# dimensions of the generated pictures for each filter.
img_width = 28
img_height = 28

def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model



def get_test(csv):
    xs = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)



def main():
    n_iter = 100

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=3)
    model.summary()
    model.load_weights('result/trained_model.h5')

    layer_dict = dict([layer.name, layer] for layer in model.layers)
    layer = layer_dict["primarycap_squash"]
    """
    n_filters = layer.filters
    print()
    print('layer={}'.format(layer))
    print()
    filter_imgs = [[] for i in range(n_filters)]
    for ind_filter in range(n_filters):
        filter_imgs[ind_filter] = np.random.random((1, img_width, img_height, 1))
        activation = K.mean(layer.output[:, :, :, ind_filter])
        grads = normalize(K.gradients(activation, model.inputs[0])[0])
        iterate = K.function([model.inputs[0], K.learning_phase()],
                             [activation, grads])

        print('processing filter %d' % ind_filter)
        for i in range(n_iter):
            act, g = iterate([filter_imgs[ind_filter], 0])
            filter_imgs[ind_filter] += g
    """
    n_filters = 32
    filter_imgs = [[] for i in range(n_filters)]
    for ind_filter in range(n_filters):
        filter_imgs[ind_filter] = np.random.random((1, img_width, img_height, 1))
        #activation = K.mean(layer.output[:, 36*ind_filter:36*(ind_filter+1), :])
        activation = K.mean(layer.output[:, ind_filter::36, :])
        grads = normalize(K.gradients(activation, model.inputs[0])[0])
        iterate = K.function([model.inputs[0], K.learning_phase()],
                             [activation, grads])
        print('processing filter %d' % ind_filter)
        for i in range(n_iter):
            act, g = iterate([filter_imgs[ind_filter], 0])
            filter_imgs[ind_filter] += g

    fig = plt.figure(figsize=(14, 2 * ind_filter / 16))
    for ind_filter in range(n_filters):
        ax = fig.add_subplot(n_filters / 16 + 1, 16, ind_filter + 1)
        ax.imshow(filter_imgs[ind_filter].reshape(img_width, img_height),
                  cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('filter %d' % ind_filter)
        plt.tight_layout()

    fig.suptitle('Filters of PrimaryCap')
    fig.savefig('filters.png')


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
