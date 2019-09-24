import os

def str2bool(v):
    return v.lower() in ('true', '1')

def prepare_dirs(config):
    for path in [config.output_dir, config.plot_dir]:
        if not os.path.exists(path):
            os.makedirs(path)