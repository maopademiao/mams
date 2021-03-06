import yaml
import os
from train.test import test, test_ensemble

config = yaml.safe_load(open('config.yml'))
mode = config['mode']
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['aspect_' + mode + '_model'][config['aspect_' + mode + '_model']['type']]['gpu'])
test_ensemble(config)