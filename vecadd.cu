#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <sstream>

using namespace std;

void vecadd_host(const vector<float>& v1, const vector<float>& v2, vector<float>& v3)
{
    v3.resize(v1.size()); // For simplicity, we assume no user error for vector sizes
    for (size_t i = 0; i < v1.size(); i++)
    {
        v3[i] = v1[i] + v2[i];
    }
}





__global__
void vecadd_kernel(const float* v1, const float* v2, float* v3, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        v3[i] = v1[i] + v2[i];
    }
}

// NOTE: Checking for errors in cudaStatus ommitted here for simplicity
void vecadd_cuda(const vector<float>& v1_h, const vector<float>& v2_h, vector<float>& v3_h) // This function encapsulates all the memory allocation and running of the kernel
{
    cudaError_t cudaStatus;
    size_t n = v1_h.size();
    size_t num_bytes = n * sizeof(float);
    float *v1, *v2, *v3;

    cudaStatus = cudaMalloc((void**)& v1, num_bytes); 
    cudaStatus = cudaMalloc((void**)& v2, num_bytes);
    cudaStatus = cudaMalloc((void**)& v3, num_bytes);

    cudaStatus = cudaMemcpy(v1, v1_h.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(v2, v2_h.data(), num_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vecadd_kernel<<<blocksPerGrid, threadsPerBlock>>>(v1, v2, v3, n); // This is how we call the kernel function

    cudaStatus = cudaDeviceSynchronize(); // This just ensures the host waits for the computation to be complete before moving on



    v3_h.resize(n); //Ensuring the vector in the host memory is the correct size to receive the result
    cudaStatus = cudaMemcpy(v3_h.data(), v3, num_bytes, cudaMemcpyDeviceToHost); // Result vector is finally copied back over to host memory

    cudaStatus = cudaFree(v1);
    cudaStatus = cudaFree(v2);
    cudaStatus = cudaFree(v3);
}





int main()
{
    const size_t NUMBER_OF_ELEMENTS = 1000000;
    vector<float> vec1(NUMBER_OF_ELEMENTS);
    vector<float> vec2(NUMBER_OF_ELEMENTS);
    vector<float> vec3(NUMBER_OF_ELEMENTS);

    auto start = chrono::high_resolution_clock::now();
    vecadd_host(vec1, vec2, vec3);
    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();

    ostringstream oss;
    oss << "Here is the time taken for the host version with " << NUMBER_OF_ELEMENTS << " elements: " << duration << endl;
    cout << oss.str();


    start = chrono::high_resolution_clock::now();
    vecadd_cuda(vec1, vec2, vec3);
    end = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::microseconds>(end - start).count();


    oss.str("");
    oss.clear();
    oss << "Here is the time taken for the cuda version with " << NUMBER_OF_ELEMENTS << " elements: " << duration << endl;
    cout << oss.str();

    return 0;
}
