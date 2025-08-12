import os
import numpy as np
import random
import torch

def count_parameters(model):
	""" 
	Count parameters in model.
	"""
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return params/1000000

def set_seed(args):
	"""
	Set seed for python random, numpy and pytorch.
	"""
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if hasattr(args, 'n_gpu'):
		if args.n_gpu > 0:
			torch.cuda.manual_seed_all(args.seed) 
