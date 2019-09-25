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

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)