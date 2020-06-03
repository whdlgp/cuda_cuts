/********************************************************************************************
* Implementing Graph Cuts on CUDA using algorithm given in CVGPU '08                       **
* paper "CUDA Cuts: Fast Graph Cuts on GPUs"                                               **
*                                                                                          **
* Copyright (c) 2008 International Institute of Information Technology.                    **
* All rights reserved.                                                                     **
*                                                                                          **
* Permission to use, copy, modify and distribute this software and its documentation for   **
* educational purpose is hereby granted without fee, provided that the above copyright     **
* notice and this permission notice appear in all copies of this software and that you do  **
* not sell the software.                                                                   **
*                                                                                          **
* THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR    **
* OTHERWISE.                                                                               **
*                                                                                          **
* Created By Vibhav Vineet.                                                                **
********************************************************************************************/

#include "CudaCuts.h"

CudaCuts::CudaCuts(int width, int height, int numOfLabels, int* dataTerm_error, int* smoothness_table, int* hcue, int* vcue)
{
	int initCheck = cudaCutsInit(width, height, numOfLabels);

	dataTerm = dataTerm_error;
	smoothTerm = smoothness_table;
	hCue = hcue;
	vCue = vcue;

	printf("Compute Capability %d\n", initCheck);

	if (initCheck > 0)
	{
		printf("The grid is initialized successfully\n");
	}
	else
	if (initCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int dataCheck = cudaCutsSetupDataTerm();

	if (dataCheck == 0)
	{
		printf("The dataterm is set properly\n");

	}
	else
	if (dataCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int smoothCheck = cudaCutsSetupSmoothTerm();

	if (smoothCheck == 0)
	{
		printf("The smoothnessterm is set properly\n");
	}
	else
	if (smoothCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}


	int hcueCheck = cudaCutsSetupHCue();

	if (hcueCheck == 0)
	{
		printf("The HCue is set properly\n");
	}
	else
	if (hcueCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}

	int vcueCheck = cudaCutsSetupVCue();


	if (vcueCheck == 0)
	{
		printf("The VCue is set properly\n");
	}
	else
	if (vcueCheck == -1)
	{
		printf("Error: Please check the device present on the system\n");
	}
}

CudaCuts::~CudaCuts()
{
	cudaCutsFreeMem();
}

void CudaCuts::run(std::vector<int> labels)
{
	for(int i = 0; i < num_Labels; i++)
	//for(int i = num_Labels-1; i >= 0 ; i--)
	{
		cudaCutsResetMem();
		cudaCutsSetupAlpha(labels[i]);
		cudaCutsSetupGraph();
		cudaCutsStochasticOptimize();
	}
	cudaCutsGetResult();
}

/********************************************************************
* cudaCutsInit(width, height, numOfLabels) function sets the      **
* width, height and numOfLabels of grid. It also initializes the  **
* block size  on the device and finds the total number of blocks  **
* running in parallel on the device. It calls checkDevice         **
* function which checks whether CUDA compatible device is present **
* on the system or not. It allocates the memory on the host and   **
* the device for the arrays which are required through the        **
* function call h_mem_init and segment_init respectively. This    **
* function returns 0 on success or -1 on failure if there is no   **
* * * CUDA compatible device is present on the system             **
* *****************************************************************/

int CudaCuts::cudaCutsInit(int widthGrid, int heightGrid, int labels)
{
	deviceCount = checkDevice();

	printf("No. of devices %d\n", deviceCount);
	if (deviceCount < 1)
		return -1;

	int cuda_device = 0;

	cudaSetDevice(cuda_device);

	cudaDeviceProp device_properties;

	CUDA_SAFE_CALL(cudaGetDeviceProperties(&device_properties, cuda_device));

	if ((3 <= device_properties.major) && (device_properties.minor < 1))
		deviceCheck = 2;
	else
	if ((3 <= device_properties.major) && (device_properties.minor >= 1))
		deviceCheck = 1;
	else
		deviceCheck = 0;



	width = widthGrid;
	height = heightGrid;
	num_Labels = labels;

	blocks_x = 1;
	blocks_y = 1;
	num_of_blocks = 1;

	num_of_threads_per_block = 256;
	threads_x = 32;
	threads_y = 8;

	width1 = threads_x * ((int)ceil((float)width / (float)threads_x));
	height1 = threads_y * ((int)ceil((float)height / (float)threads_y));

	graph_size = width * height;
	graph_size1 = width1 * height1;
	size_int = sizeof(int)* graph_size1;

	blocks_x = (int)((ceil)((float)width1 / (float)threads_x));
	blocks_y = (int)((ceil)((float)height1 / (float)threads_y));

	num_of_blocks = (int)((ceil)((float)graph_size1 / (float)num_of_threads_per_block));

	h_mem_init();
	d_mem_init();
	cueValues = 0;

	return deviceCheck;

}


int CudaCuts::checkDevice()
{
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		return -1;
	}


	return deviceCount;
}


void CudaCuts::h_mem_init()
{
	h_reset_mem = (int*)malloc(sizeof(int)* graph_size1);
	h_graph_height = (int*)malloc(size_int);
	pixelLabel = (int*)malloc(size_int);
	h_pixel_mask = (bool*)malloc(sizeof(bool)* graph_size1);

	for (int i = 0; i < graph_size1; i++)
	{
		pixelLabel[i] = 0;
		h_graph_height[i] = 0;
	}

	for (int i = 0; i < graph_size1; i++)
	{
		h_reset_mem[i] = 0;
	}
}


void CudaCuts::d_mem_init()
{
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_left_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_right_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_down_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_up_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_push_reser, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_sink_weight, sizeof(int)* graph_size1));

	CUDA_SAFE_CALL(cudaMalloc((void**)&s_left_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&s_right_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&s_down_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&s_up_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&s_push_reser, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&s_sink_weight, sizeof(int)* graph_size1));


	CUDA_SAFE_CALL(cudaMalloc((void**)&d_stochastic, sizeof(int)* num_of_blocks));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_stochastic_pixel, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_terminate, sizeof(int)* num_of_blocks));


	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_sink_weight, sizeof(int) * graph_size1 ) );
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_sink_weight, sizeof(int) * graph_size1 ) );
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_sink_weight, sizeof(int) * graph_size1 ) );
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_sink_weight, sizeof(int) * graph_size1 ) );


	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pull_left, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pull_right, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pull_down, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pull_up, sizeof(int)* graph_size1));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_graph_heightr, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_graph_heightw, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_relabel_mask, sizeof(int)* graph_size1));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_pixel_mask, sizeof(bool)*graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_over, sizeof(bool)* 1));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_counter, sizeof(int)));

	CUDA_SAFE_CALL(cudaMalloc((void **)&dPixelLabel, sizeof(int)* width1 * height1));
	CUDA_SAFE_CALL(cudaMemcpy(d_left_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_right_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_down_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_up_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_push_reser, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_sink_weight, h_reset_mem, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));

	h_relabel_mask = (int*)malloc(sizeof(int)*width1*height1);

	h_stochastic = (int *)malloc(sizeof(int)* num_of_blocks);
	h_stochastic_pixel = (int *)malloc(sizeof(int)* graph_size1);



	for (int i = 0; i < graph_size1; i++)
		h_relabel_mask[i] = 1;


	CUDA_SAFE_CALL(cudaMemcpy(d_relabel_mask, h_relabel_mask, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));

	int *dpixlab = (int*)malloc(sizeof(int)*width1*height1);

	for (int i = 0; i < width1 * height1; i++)
	{
		dpixlab[i] = 0;
		h_stochastic_pixel[i] = 1;
	}

	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 1;
	}

	CUDA_SAFE_CALL(cudaMemcpy(d_stochastic, h_stochastic, sizeof(int)* num_of_blocks, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_terminate, h_stochastic, sizeof(int)* num_of_blocks, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_stochastic_pixel, h_stochastic_pixel, sizeof(int)* graph_size1, cudaMemcpyHostToDevice));


	CUDA_SAFE_CALL(cudaMemcpy(dPixelLabel, dpixlab, sizeof(int)* width1 * height1, cudaMemcpyHostToDevice));

	free(dpixlab);
}


int CudaCuts::cudaCutsSetupDataTerm()
{
	if (deviceCheck < 1)
		return -1;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dDataTerm, sizeof(int)* width * height * num_Labels));

	CUDA_SAFE_CALL(cudaMemcpy(dDataTerm, dataTerm, sizeof(int)* width * height * num_Labels, cudaMemcpyHostToDevice));

	return 0;
}


int CudaCuts::cudaCutsSetupSmoothTerm()
{
	if (deviceCheck < 1)
		return -1;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dSmoothTerm, sizeof(int)* num_Labels * num_Labels));

	CUDA_SAFE_CALL(cudaMemcpy(dSmoothTerm, smoothTerm, sizeof(int)* num_Labels * num_Labels, cudaMemcpyHostToDevice));

	return 0;
}

int CudaCuts::cudaCutsSetupHCue()
{

	if (deviceCheck < 1)
		return -1;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dHcue, sizeof(int)* width * height));

	CUDA_SAFE_CALL(cudaMemcpy(dHcue, hCue, sizeof(int)* width * height, cudaMemcpyHostToDevice));

	cueValues = 1;

	return 0;
}

int CudaCuts::cudaCutsSetupVCue()
{
	if (deviceCheck < 1)
		return -1;

	CUDA_SAFE_CALL(cudaMalloc((void **)&dVcue, sizeof(int)* width * height));

	CUDA_SAFE_CALL(cudaMemcpy(dVcue, vCue, sizeof(int)* width * height, cudaMemcpyHostToDevice));

	return 0;
}

int CudaCuts::cudaCutsSetupAlpha(int alpha)
{
	alpha_label = alpha;
	return alpha_label;
}

int CudaCuts::cudaCutsSetupGraph()
{

	if (deviceCheck < 1)
		return -1;

	for (int i = 0; i < graph_size1; i++)
	{
		h_reset_mem[i] = 0;
		h_graph_height[i] = 0;
	}

	int blockEdge = (int)((ceil)((float)(width * height) / (float)256));
	dim3 block_weight(256, 1, 1);
	dim3 grid_weight(blockEdge, 1, 1);

	if (cueValues == 1)
	{
		CudaWeightCue << < grid_weight, block_weight >> >(alpha_label, d_left_weight, d_right_weight, d_down_weight,
			d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm,
			dSmoothTerm, dHcue, dVcue, width, height, num_Labels);
	}
	else
	{
		CudaWeight << < grid_weight, block_weight >> >(alpha_label, d_left_weight, d_right_weight, d_down_weight,
			d_up_weight, d_push_reser, d_sink_weight, dPixelLabel, dDataTerm,
			dSmoothTerm, width, height, num_Labels);

	}

	int *temp_left_weight, *temp_right_weight, *temp_down_weight, *temp_up_weight, *temp_source_weight, *temp_terminal_weight;

	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_left_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_right_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_down_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_up_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_source_weight, sizeof(int)* graph_size1));
	CUDA_SAFE_CALL(cudaMalloc((void **)&temp_terminal_weight, sizeof(int)* graph_size1));

	int blockEdge1 = (int)((ceil)((float)(width1 * height1) / (float)256));
	dim3 block_weight1(256, 1, 1);
	dim3 grid_weight1(blockEdge1, 1, 1);

	adjustedgeweight << <grid_weight1, block_weight1 >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser,
		d_sink_weight, temp_left_weight, temp_right_weight, temp_down_weight, temp_up_weight,
		temp_source_weight, temp_terminal_weight, width, height, graph_size, width1,
		height1, graph_size1);

	copyedgeweight << <grid_weight1, block_weight1 >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_push_reser, d_sink_weight,
		temp_left_weight, temp_right_weight, temp_down_weight, temp_up_weight, temp_source_weight,
		temp_terminal_weight, d_pull_left, d_pull_right, d_pull_down, d_pull_up, d_relabel_mask,
		d_graph_heightr, d_graph_heightw, width, height, graph_size, width1, height1, graph_size1);

	CUDA_SAFE_CALL(cudaFree(temp_left_weight));
	CUDA_SAFE_CALL(cudaFree(temp_right_weight));
	CUDA_SAFE_CALL(cudaFree(temp_up_weight));
	CUDA_SAFE_CALL(cudaFree(temp_down_weight));
	CUDA_SAFE_CALL(cudaFree(temp_source_weight));
	CUDA_SAFE_CALL(cudaFree(temp_terminal_weight));
	return 0;
}

int CudaCuts::cudaCutsAtomicOptimize()
{
	if (deviceCheck < 1)
	{
		return -1;
	}

	cudaCutsAtomic();

	bfsLabeling();

	return 0;

}

int CudaCuts::cudaCutsStochasticOptimize()
{
	if (deviceCheck < 1)
	{
		return -1;
	}

	cudaCutsStochastic();

	bfsLabeling();

	return 0;

}

void CudaCuts::cudaCutsAtomic()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	bool finish = true;

	counter = num_of_blocks;

	int numThreadsEnd = 256, numBlocksEnd = 1;
	if (numThreadsEnd > counter)
	{
		numBlocksEnd = 1;
		numThreadsEnd = counter;
	}
	else
	{
		numBlocksEnd = (int)ceil(counter / (double)numThreadsEnd);
	}

	dim3 End_block(numThreadsEnd, 1, 1);
	dim3 End_grid(numBlocksEnd, 1, 1);

	int *d_counter;

	bool *d_finish;
	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 0;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_counter, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_finish, sizeof(bool)));

	CUDA_SAFE_CALL(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	counter = 0;
	int *d_relabel;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_relabel, sizeof(int)));

	int h_relabel = 0;

	int block_num = width1 / 32;

	int *d_block_num;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_block_num, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_block_num, &block_num, sizeof(int), cudaMemcpyHostToDevice));

	int h_count_blocks = num_of_blocks;
	int *d_count_blocks;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_count_blocks, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));

	h_count_blocks = 0;



	CUDA_SAFE_CALL(cudaMemcpy(d_relabel, &h_relabel, sizeof(int), cudaMemcpyHostToDevice));

	counter = 1;
	kernel_push1_start_atomic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
		d_sink_weight, d_push_reser,
		d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
		graph_size1, width1, height1, d_relabel, d_stochastic, d_counter, d_finish);

	int h_terminate_condition = 0;
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	do
	{

		if (counter % 10 == 0)
		{
			finish = true;
			CUDA_SAFE_CALL(cudaMemcpy(d_finish, &finish, sizeof(bool), cudaMemcpyHostToDevice));
			kernel_push_stochastic1 << <grid_push, block_push >> >(d_push_reser, s_push_reser, d_count_blocks, d_finish, d_block_num, width1);
			CUDA_SAFE_CALL(cudaMemcpy(&finish, d_finish, sizeof(bool), cudaMemcpyDeviceToHost));
			if (finish == false)
				h_terminate_condition++;
		}
		if (counter % 11 == 0)
		{
			CUDA_SAFE_CALL(cudaMemset(d_terminate, 0, sizeof(int)*num_of_blocks));
			h_count_blocks = 0;
			CUDA_SAFE_CALL(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));
			kernel_push_atomic2 << <grid_push, block_push >> >(d_terminate, d_push_reser, s_push_reser, d_block_num, width1);

			kernel_End << <End_grid, End_block >> >(d_terminate, d_count_blocks, d_counter);

		}

		if (counter % 2 == 0)
		{

			kernel_push1_atomic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1);

			/*kernel_push2_atomic<<<grid_push,block_push>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
			d_sink_weight, d_push_reser,d_pull_left, d_pull_right, d_pull_down, d_pull_up,
			d_relabel_mask,d_graph_heightr,d_graph_heightw, graph_size,width,height,
			graph_size1, width1 , height1 );
			*/
			kernel_relabel_atomic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1);
		}
		else
		{
			kernel_push1_atomic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1);

			/*kernel_push2_atomic<<<grid_push,block_push>>>(d_left_weight,d_right_weight, d_down_weight, d_up_weight,
			d_sink_weight, d_push_reser,d_pull_left, d_pull_right, d_pull_down, d_pull_up,
			d_relabel_mask,d_graph_heightr,d_graph_heightw, graph_size,width,height,
			graph_size1, width1 , height1);
			*/
			kernel_relabel_atomic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1);

		}
		counter++;
	} while (h_terminate_condition != 2);

	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	float time;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
	printf("TT Cuts :: %f ms\n", time);

}


void CudaCuts::cudaCutsStochastic()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	bool finish = true;

	counter = num_of_blocks;

	int numThreadsEnd = 256, numBlocksEnd = 1;
	if (numThreadsEnd > counter)
	{
		numBlocksEnd = 1;
		numThreadsEnd = counter;
	}
	else
	{
		numBlocksEnd = (int)ceil(counter / (double)numThreadsEnd);
	}

	dim3 End_block(numThreadsEnd, 1, 1);
	dim3 End_grid(numBlocksEnd, 1, 1);




	bool *d_finish;
	for (int i = 0; i < num_of_blocks; i++)
	{
		h_stochastic[i] = 0;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_counter, sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_finish, sizeof(bool)));

	CUDA_SAFE_CALL(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	counter = 0;
	int *d_relabel;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_relabel, sizeof(int)));

	int h_relabel = 0;


	int block_num = width1 / 32;

	int *d_block_num;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_block_num, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_block_num, &block_num, sizeof(int), cudaMemcpyHostToDevice));


	int h_count_blocks = num_of_blocks;
	int *d_count_blocks;

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_count_blocks, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));

	h_count_blocks = 0;

	CUDA_SAFE_CALL(cudaMemcpy(d_relabel, &h_relabel, sizeof(int), cudaMemcpyHostToDevice));

	counter = 1;
	kernel_push1_start_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
		d_sink_weight, d_push_reser,
		d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
		graph_size1, width1, height1, d_relabel, d_stochastic, d_counter, d_finish);
	int h_terminate_condition = 0;
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//for (int i = 0 ; i < 400; i++ )
	do
	{
		if (counter % 10 == 0)
		{
			finish = true;
			CUDA_SAFE_CALL(cudaMemcpy(d_finish, &finish, sizeof(bool), cudaMemcpyHostToDevice));
			kernel_push_stochastic1 << <grid_push, block_push >> >(d_push_reser, s_push_reser, d_count_blocks, d_finish, d_block_num, width1);
			CUDA_SAFE_CALL(cudaMemcpy(&finish, d_finish, sizeof(bool), cudaMemcpyDeviceToHost));
		}
		if (counter % 11 == 0)
		{
			CUDA_SAFE_CALL(cudaMemset(d_stochastic, 0, sizeof(int)*num_of_blocks));
			CUDA_SAFE_CALL(cudaMemset(d_terminate, 0, sizeof(int)*num_of_blocks));
			h_count_blocks = 0;
			CUDA_SAFE_CALL(cudaMemcpy(d_count_blocks, &h_count_blocks, sizeof(int), cudaMemcpyHostToDevice));
			kernel_push_stochastic2 << <grid_push, block_push >> >(d_terminate, d_relabel_mask, d_push_reser, s_push_reser, d_stochastic, d_block_num, width1);

			kernel_End << <End_grid, End_block >> >(d_terminate, d_count_blocks, d_counter);

			//if ( finish == false ) printf("%d \n",counter);
			if (finish == false && counter % 121 != 0 && counter > 0)
				h_terminate_condition++;

		}
		if (counter % 2 == 0)
		{

			kernel_push1_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

			/*kernel_push2_stochastic<<<grid_push, block_push>>>( d_left_weight, d_right_weight, d_down_weight, d_up_weight,
			d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
			d_relabel_mask, d_graph_heightr, d_graph_heightw,
			graph_size, width, height, graph_size1, width1, height1, d_relabel, d_stochastic, d_counter, d_finish, d_block_num) ;
			*/
			kernel_relabel_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,/*d_pull_left, d_pull_right, d_pull_down, d_pull_up,*/
				d_relabel_mask, d_graph_heightr, d_graph_heightw, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

		}
		else
		{
			kernel_push1_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);


			/*kernel_push2_stochastic<<<grid_push, block_push>>>( d_left_weight, d_right_weight, d_down_weight, d_up_weight,
			d_sink_weight, d_push_reser, d_pull_left, d_pull_right, d_pull_down, d_pull_up,
			d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height, graph_size1,
			width1, height1, d_relabel, d_stochastic, d_counter, d_finish, d_block_num) ;
			*/

			kernel_relabel_stochastic << <grid_push, block_push >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight,
				d_sink_weight, d_push_reser,
				d_relabel_mask, d_graph_heightw, d_graph_heightr, graph_size, width, height,
				graph_size1, width1, height1, d_stochastic, d_block_num);

		}
		counter++;
	} while (h_terminate_condition == 0 && counter < 500);


	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	float time;
	CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
	printf("TT Cuts :: %f ms\n", time);

}

void CudaCuts::bfsLabeling()
{

	dim3 block_push(threads_x, threads_y, 1);
	dim3 grid_push(blocks_x, blocks_y, 1);

	dim3 d_block(num_of_threads_per_block, 1, 1);
	dim3 d_grid(num_of_blocks, 1, 1);

	CUDA_SAFE_CALL(cudaMemcpy(d_graph_heightr, h_graph_height, size_int, cudaMemcpyHostToDevice));

	for (int i = 0; i < graph_size; i++)
		h_pixel_mask[i] = true;

	CUDA_SAFE_CALL(cudaMemcpy(d_pixel_mask, h_pixel_mask, sizeof(bool)* graph_size1, cudaMemcpyHostToDevice));

	kernel_bfs_t << <d_grid, d_block, 0 >> >(d_push_reser, d_sink_weight, d_graph_heightr, d_pixel_mask, graph_size, width, height, graph_size1, width1, height1);
	counter = 1;

	CUDA_SAFE_CALL(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));

	do
	{
		h_over = false;

		CUDA_SAFE_CALL(cudaMemcpy(d_over, &h_over, sizeof(bool), cudaMemcpyHostToDevice));

		kernel_bfs << < d_grid, d_block, 0 >> >(d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_graph_heightr, d_pixel_mask,
			graph_size, width, height, graph_size1, width1, height1, d_over, d_counter);

		CUDA_SAFE_CALL(cudaMemcpy(&h_over, d_over, sizeof(bool), cudaMemcpyDeviceToHost));

		counter++;

		CUDA_SAFE_CALL(cudaMemcpy(d_counter, &counter, sizeof(int), cudaMemcpyHostToDevice));
	} while (h_over);

	updatePixelLabel<<<d_grid, d_block, 0>>>(alpha_label, dPixelLabel, d_graph_heightr, graph_size1, width, height, width1, height1);
}


int CudaCuts::cudaCutsGetResult()
{
	if (deviceCheck < 1)
		return -1;

	CUDA_SAFE_CALL(cudaMemcpy(pixelLabel, dPixelLabel, size_int, cudaMemcpyDeviceToHost));

	return 0;

}

void CudaCuts::cudaCutsFreeMem()
{
	free(h_reset_mem);
	free(h_graph_height);
	free(pixelLabel);
	free(h_pixel_mask);
	
	free(h_relabel_mask);
	free(h_stochastic);
	free(h_stochastic_pixel);
	
	free(hCue);
	free(vCue);
	free(dataTerm);
	free(smoothTerm);

	CUDA_SAFE_CALL(cudaFree(d_left_weight));
	CUDA_SAFE_CALL(cudaFree(d_right_weight));
	CUDA_SAFE_CALL(cudaFree(d_down_weight));
	CUDA_SAFE_CALL(cudaFree(d_up_weight));
	CUDA_SAFE_CALL(cudaFree(d_sink_weight));
	CUDA_SAFE_CALL(cudaFree(d_push_reser));

	CUDA_SAFE_CALL(cudaFree(d_pull_left));
	CUDA_SAFE_CALL(cudaFree(d_pull_right));
	CUDA_SAFE_CALL(cudaFree(d_pull_down));
	CUDA_SAFE_CALL(cudaFree(d_pull_up));

	CUDA_SAFE_CALL(cudaFree(d_graph_heightr));
	CUDA_SAFE_CALL(cudaFree(d_graph_heightw));

	CUDA_SAFE_CALL(cudaFree(s_left_weight));
	CUDA_SAFE_CALL(cudaFree(s_right_weight));
	CUDA_SAFE_CALL(cudaFree(s_down_weight));
	CUDA_SAFE_CALL(cudaFree(s_up_weight));
	CUDA_SAFE_CALL(cudaFree(s_push_reser));
	CUDA_SAFE_CALL(cudaFree(s_sink_weight));
	
	
	CUDA_SAFE_CALL(cudaFree(d_stochastic));
	CUDA_SAFE_CALL(cudaFree(d_stochastic_pixel));
	CUDA_SAFE_CALL(cudaFree(d_terminate));
	
	CUDA_SAFE_CALL(cudaFree(d_relabel_mask));
	
	CUDA_SAFE_CALL(cudaFree(d_pixel_mask));
	CUDA_SAFE_CALL(cudaFree(d_over));
	CUDA_SAFE_CALL(cudaFree(d_counter));
	
	CUDA_SAFE_CALL(cudaFree(dPixelLabel));
}

void CudaCuts::cudaCutsResetMem()
{
	cudaMemset(d_left_weight, 0, sizeof(int)* graph_size1);
	cudaMemset(d_right_weight, 0, sizeof(int)* graph_size1);
	cudaMemset(d_down_weight, 0, sizeof(int)* graph_size1);
	cudaMemset(d_up_weight, 0, sizeof(int)* graph_size1);
	cudaMemset(d_push_reser, 0, sizeof(int)* graph_size1);
	cudaMemset(d_sink_weight, 0, sizeof(int)* graph_size1);
}
