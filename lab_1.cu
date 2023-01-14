#include <stdio.h>
#include <cuda_runtime.h>

#define N 	256
#define 	THREADS_PER_BLOCK 256
#define BLOCKS 	ceil( float(N) / THREADS_PER_BLOCK )

__global__ void vector_addition (double *a, double *b, double *result)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) 
	result[idx] = a[idx] + b[idx];
}

int main()
{
   
    double *a = new double[N];
    double *b = new double[N];
    double *res = new double[N];

    double *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, N*sizeof(double));
    cudaMalloc(&d_b, N*sizeof(double));
    cudaMalloc(&d_res, N*sizeof(double));

    for(int i=0; i<N; i++)
    {
        a[i] = rand()%10;
        b[i] = rand()%10;
    }

    cudaMemcpy(d_a, a, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);

    vector_addition<<< BLOCKS, THREADS_PER_BLOCK >>>(d_a, d_b, d_res);

    cudaMemcpy(res, d_res, N*sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
	printf("%lf\n", res[i]);
    }

    delete a;
    delete b;
    delete res;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);


    return 0;
}
