import os
import pylearn2
from pylearn2.config import yaml_parse

with open('/u/huilgolr/trial/mlp2hid.yaml') as f:
	train = f.read()
hyper_params = {
	'train_stop':50000,
	'valid_stop':60000,
	'dim_h0':100,
	'max_epochs':10,
	'save_path':'.'}
train = train % (hyper_params)
train = yaml_parse.load(train)
train.main_loop()