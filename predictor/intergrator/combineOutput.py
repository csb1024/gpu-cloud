def combineResults(output_vectors, output_types):
	output_vec_size=len(output_types)
	output_vec_num = len(output_vectors)
	pred_result = []
	for i in range(output_vec_size):
		output_type = output_types[i]
		if output_type == "exec-time":
			result = 0.
			for j in range(output_vec_num):
				# summation
				result = result + float(output_vectors[j][i])
			pred_result.append(result)
		elif output_type == "allocated_mem":
			result = 0.
			for j in range(output_vec_num):
				#summation
				result = result + float(output_vectors[j][i])
			pred_result.append(result)
	return pred_result

