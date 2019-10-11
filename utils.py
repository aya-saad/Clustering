#################################################################################################################
# A Modularized implementation for
# Image enhancement, extracting descriptors, clustering, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_DESCRIPTORS - CLUSTERING - EVALUATION_VISUALIZATION
# Implementation of the utils
# Author: Aya Saad
# Date created: 24 September 2019
#
#################################################################################################################

import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects as PathEffects
from PIL import Image, ImageDraw, ImageColor
from tqdm import tqdm

color_list = ['r', 'b', 'g', 'c', 'k', 'y', 'm']
box_color_list = ['red', 'blue', 'green', 'cyan', 'yellow', 'magenta']

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    #palette = 10*color_list
    # palette = np.array(sns.color_palette(color_palette, num_classes))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=20)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

def tile_scatter(x, colors, input_data):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    sns.set_palette(palette)
    #####
    tx, ty = x[:, 0], x[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
    plt.plot(tx, ty, '.')
    width = 1000
    height = 1000
    max_dim = 100
    full_image = Image.new('RGB', (width, height), (255, 255, 255))
    images = input_data.iloc[:, 0]

    box_color = 10*box_color_list

    for img, x, y, c in tqdm(zip(images, tx, ty, colors)):
        tile = Image.open(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((max(1, int(tile.width / rs)), max(1, int(tile.height / rs))), Image.ANTIALIAS)

        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile.size[0] - 1, tile.size[1] - 1], fill=None, outline=box_color[c], width=5)

        full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)))

    #####
    plt.imshow(full_image)

    f = plt.figure(figsize=(16, 12))


    return f