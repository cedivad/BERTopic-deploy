import json
import numpy as np
import threading
import time

import os 
import cuml
import hdbscan
from  pympler import asizeof
import pympler
import pickle

import triton_python_backend_utils as pb_utils

class TritonPythonModel:

	def initialize(self, args):
		
		filename = '/models/umap.pickle'
		filename = open(filename,'rb')
		self.umap_model = pickle.load(filename)
		filename.close()

	def execute(self, requests):
		ret = []
		req = []
		#print("req:", len(requests))
		for request in requests:
			embeddings = pb_utils.get_input_tensor_by_name(request, 'IN').as_numpy()
			if(len(embeddings) > 1):
				print("broke assumptions")
				exit(0)
				
			req.append(embeddings[0])
		req = np.vstack(req)
		#print(req)
		req = cuml.preprocessing.normalize(req)
		umap_embeddings = self.umap_model.transform(req)
		#print("umap_embeddings:", len(umap_embeddings))
		for emb in umap_embeddings:
			out_output = pb_utils.Tensor("REDUCED", np.array([emb], np.float32))
			response = pb_utils.InferenceResponse(output_tensors=[out_output])
			ret.append(response)
		
		return ret