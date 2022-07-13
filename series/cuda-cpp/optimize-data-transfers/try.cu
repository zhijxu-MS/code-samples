/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <iostream>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
using namespace std;

typedef struct callBackData {
  void *src;
  void *dst;
  void *expected;
  unsigned int bytes;
} callBackData_t;
void check_accuracy(int *data, int* expected, unsigned int num)
{
    for(int i=0; i<num; i++)
    {
        if(data[i] != expected[i])
        {
            printf("something is wrong, total num is %d, index is %d, %d vs %d \n", num, i, data[i], expected[i]);
            return;
        }
    }
    printf("data is equal!\n");
}
void CUDART_CB memcpyHostToHost(void *ptr) {
    printf("in memcpyHostToHost\n");
    auto *info = (callBackData_t*) ptr;
    auto start = chrono::steady_clock::now();
    // checkCuda(cudaMemcpy(info->dst, info->src, info->bytes, cudaMemcpyHostToHost));
    memcpy(info->dst, info->src, info->bytes);
    cout << "H2H time: " << chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count()/1000.0 << "ms" << endl;
    // check_accuracy((int*)info->dst, (int*)info->expected, info->bytes/4);
    // memcmp(info->dst, info->src, info->bytes);
}

int main()
{
  unsigned int nElements = 512*1024*1024;
  const unsigned int bytes = nElements * sizeof(float);

  // 2 stream
  int num_streams = 2;
  cudaStream_t streams[num_streams];
  for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
  }
  // device array
  float *d_a;
  checkCuda( cudaMalloc((void**)&d_a, bytes) );
  float *h_expected = (float*)malloc(bytes);
  for(int i=0; i<nElements; i++){
    h_expected[i] = i;
  }
  checkCuda(cudaMemcpy(d_a, h_expected, bytes, cudaMemcpyHostToDevice));
  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;
  h_aPageable = (float*)malloc(bytes);                    // host pageable
  h_bPageable = (float*)malloc(bytes);                    // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
  //
  for (int j=0; j<6; j++){
    //stream 0
    cudaMemcpyAsync(h_aPinned, d_a, bytes, cudaMemcpyDeviceToHost, streams[0]);
    callBackData_t *host_args1 = new callBackData_t;
    host_args1->src = (void*)h_aPinned;
    host_args1->dst = (void*)h_aPageable;
    host_args1->expected = (void*)h_expected;
    host_args1->bytes = bytes;
    cudaLaunchHostFunc(streams[0], memcpyHostToHost, host_args1);
    //stream 1
    cudaMemcpyAsync(h_bPinned, d_a, bytes, cudaMemcpyDeviceToHost, streams[1]);
    callBackData_t *host_args2 = new callBackData_t;
    host_args2->src = (void*)h_bPinned;
    host_args2->dst = (void*)h_bPageable;
    host_args2->expected = (void*)h_expected;
    host_args2->bytes = bytes;
    cudaLaunchHostFunc(streams[1], memcpyHostToHost, host_args2);
  }

    // cleanup
    checkCuda(cudaDeviceSynchronize());
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
    free(h_expected);
    printf("finish main function, everyhing is done\n");
  return 0;
}
