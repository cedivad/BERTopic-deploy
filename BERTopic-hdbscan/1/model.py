import json
import numpy as np
import threading
import time

import os 
import hdbscan
from  pympler import asizeof
import pympler
import pickle


import triton_python_backend_utils as pb_utils

class TritonPythonModel:

	def initialize(self, args):
		filename = '/models/hdbscan.pickle'
		filename = open(filename,'rb')
		self.hdbscan_model = pickle.load(filename)
		filename.close()
		
		filename = '/models/mappings.pickle'
		filename = open(filename,'rb')
		self.mappings = pickle.load(filename)
		filename.close()

	def execute(self, requests):
		
		
		ret = []
		req = []
		for request in requests:
			
			reduced = pb_utils.get_input_tensor_by_name(request, 'REDUCED').as_numpy()
			if(len(reduced) > 1):
				print("broke assumptions")
				exit(0)
				
			req.append(reduced[0])
		
		predictions, probabilities = hdbscan.approximate_predict(self.hdbscan_model, req) # or in_input[0] ?? prob not!
				
		mapped_predictions = [self.mappings[prediction]
								  if prediction in self.mappings
								  else -1
								  for prediction in predictions]
		
		for pred in mapped_predictions:
			
			out_output = pb_utils.Tensor("OUT", np.array([pred], np.int32))
			response = pb_utils.InferenceResponse(output_tensors=[out_output])
			ret.append(response)
			
		return ret