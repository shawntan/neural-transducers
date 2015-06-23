import numpy as np
import random

np.random.seed(1234)
random.seed(1234)
def copy(dimension,sequence_length):
	sequence = np.random.binomial(1,0.5,(sequence_length,dimension)).astype(np.int8)
	input_sequence  = np.zeros((sequence_length*2+1,dimension+1),dtype=np.int8)
	output_sequence = np.zeros((sequence_length*2+1,dimension+1),dtype=np.int8)
	
	input_sequence[:sequence_length,:-1]  = sequence
	input_sequence[sequence_length,-1] = 1
        output_sequence[sequence_length+1:,:-1] = sequence[::-1]
	return input_sequence,output_sequence

