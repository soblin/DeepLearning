CC=g++
NVCC=/usr/local/cuda-9.1/bin/nvcc
CUDA_TOP=/usr/local/cuda-9.1
INC=-I$(CUDA_TOP)/include -gencode=arch=compute_50,code=\"sm_50,compute_50\"


OBJ=mat_ones_kernel.o mat_mul_elementwise_kernel.o mat_mul_elementwise_plus_kernel.o slice_rows_kernel.o mat_div_kernel.o mat_mod_kernel.o mat_cos_kernel.o mat_sin_kernel.o mat_exp_kernel.o mat_log_kernel.o mat_sqrt_kernel.o mat_sqrt_d_kernel.o mat_inverse_kernel.o mat_inverse_d_kernel.o sigmoid_kernel.o sigmoid_d_kernel.o tanh_kernel.o tanh_d_kernel.o relu_kernel.o relu_d_kernel.o prelu_kernel.o prelu_d_kernel.o mat_sum_kernel.o mat_l2_kernel.o softmax_kernel.o softmax_cross_entropy_kernel.o

libcumat.so:$(OBJ)
	gcc -shared -o libcumat.so $(OBJ)

mat_ones_kernel.o: mat_ones_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_ones_kernel.cu $(INC)

mat_mul_elementwise_kernel.o: mat_mul_elementwise_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_mul_elementwise_kernel.cu $(INC)

mat_mul_elementwise_plus_kernel.o: mat_mul_elementwise_plus_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_mul_elementwise_plus_kernel.cu $(INC)

slice_rows_kernel.o: slice_rows_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c slice_rows_kernel.cu $(INC)

mat_div_kernel.o: mat_div_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_div_kernel.cu $(INC)

mat_mod_kernel.o: mat_mod_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_mod_kernel.cu $(INC)

mat_cos_kernel.o: mat_cos_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_cos_kernel.cu $(INC)

mat_sin_kernel.o: mat_sin_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sin_kernel.cu $(INC)

mat_exp_kernel.o: mat_exp_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_exp_kernel.cu $(INC)

mat_log_kernel.o: mat_log_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_log_kernel.cu $(INC)

mat_sqrt_kernel.o: mat_sqrt_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sqrt_kernel.cu $(INC)

mat_sqrt_d_kernel.o: mat_sqrt_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sqrt_d_kernel.cu $(INC)

mat_inverse_kernel.o: mat_inverse_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_inverse_kernel.cu $(INC)

mat_inverse_d_kernel.o: mat_inverse_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_inverse_d_kernel.cu $(INC)

sigmoid_kernel.o: sigmoid_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c sigmoid_kernel.cu $(INC)

sigmoid_d_kernel.o: sigmoid_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c sigmoid_d_kernel.cu $(INC)

tanh_kernel.o: tanh_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c tanh_kernel.cu $(INC)

tanh_d_kernel.o: tanh_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c tanh_d_kernel.cu $(INC)

relu_kernel.o: relu_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_kernel.cu $(INC)

relu_d_kernel.o: relu_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c relu_d_kernel.cu $(INC)

prelu_d_kernel.o: prelu_d_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c prelu_d_kernel.cu $(INC)

prelu_kernel.o: prelu_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c prelu_kernel.cu $(INC)

mat_sum_kernel.o: mat_sum_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_sum_kernel.cu $(INC)

mat_l2_kernel.o: mat_l2_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c mat_l2_kernel.cu $(INC)

softmax_kernel.o: softmax_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c softmax_kernel.cu $(INC)

softmax_cross_entropy_kernel.o: softmax_cross_entropy_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c softmax_cross_entropy_kernel.cu $(INC)

element_wise_clip_kernel.o: element_wise_clip_kernel.cu
	$(NVCC) -Xcompiler -fPIC -c element_wise_clip_kernel.cu $(INC)

clean:
	rm -f libcumat.so
	rm -f *.o
