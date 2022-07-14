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
#include <cstdlib>
#include <unistd.h>
#include <mpi.h>
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
  size_t bytes;
} callBackData_t;
void check_accuracy(int *data, int* expected, size_t num)
{
    for(size_t i=0; i<num; i++)
    {
        if(data[i] != expected[i])
        {
            printf("something is wrong, total num is %ld, index is %ld, %d vs %d \n", num, i, data[i], expected[i]);
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
    // memcmp(info->dst, info->src, info->bytes);
    // check_accuracy((int*)info->dst, (int*)info->expected, info->bytes/4);
}

int main(int argc, char *argv [])
{
  size_t nElements = 1024*1024*1024;
  size_t bytes = nElements * sizeof(float);

  // to xiaoran:  preparation work
  // select gpu device >> each rank will work on its gpu device, its gpu data, ...
  // int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm comm;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  cudaSetDevice(rank);
  cout << "i am rank " << rank << endl;
  // 2 stream
  int num_streams = 2;
  cudaStream_t streams[num_streams];
  for (size_t i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
  }
  // device array
  float *d_a;
  checkCuda( cudaMalloc((void**)&d_a, 2*bytes) );
  float *h_expected = (float*)malloc(2*bytes);
  memset(h_expected, 0, 2*bytes);
  for(size_t i=0; i<2*nElements; i++){
    h_expected[i] = i;
  }
  checkCuda(cudaMemcpy(d_a, h_expected, 2*bytes, cudaMemcpyHostToDevice));
  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;
  h_aPageable = (float*)malloc(bytes);
  h_bPageable = (float*)malloc(2*bytes);
  memset(h_aPageable, 0, bytes); // make sure malloc memory is alloc in OS
  memset(h_bPageable, 0, 2*bytes);
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) );
  checkCuda( cudaMallocHost((void**)&h_bPinned, 2*bytes) );

  sleep(5); // above work can be done in advance.


  //to xiaoran:  each loop iteration is a nebula.save
  for (int j=0; j<6; j++){
    auto res = MPI_Barrier (MPI_COMM_WORLD); // make sure all rank are sync, so they can nebula.save at the same time
    //stream 0
    cudaMemcpyAsync(h_aPinned, d_a, bytes, cudaMemcpyDeviceToHost, streams[0]);
    callBackData_t *host_args1 = new callBackData_t;
    host_args1->src = (void*)h_aPinned;
    host_args1->dst = (void*)h_aPageable;
    host_args1->expected = (void*)h_expected;
    host_args1->bytes = bytes;
    cudaLaunchHostFunc(streams[0], memcpyHostToHost, host_args1);
    //stream 1
    cudaMemcpyAsync(h_bPageable, d_a, 2*bytes, cudaMemcpyDeviceToHost, streams[1]); // actually this async is sync.

     // to xiaoran:  the following code of the loop is just to check accuracy, it's not necessary to happen in customer's real run
    // check accuracy
    // memset(h_aPageable, 0, bytes); # if this statement is uncomment, then the following check will be wrong >> just to make sure the check is woking its job
    checkCuda(cudaDeviceSynchronize());
    if (memcmp(h_aPageable, h_expected, bytes) != 0 or memcmp(h_bPageable, h_expected, 2*bytes) != 0)
    {
       printf("something is wrong, copied data is not same as expected");
       return -1;
    }
    else
    {
      printf("copied data is expected!");
    }
    // cudaMemcpy(h_bPageable, h_bPinned, 2*bytes, cudaMemcpyHostToHost);
    memset(h_aPageable, 0, bytes); // just to make sure later copy is doing their job
    memset(h_bPageable, 0, 2*bytes);
    sleep(10);
    continue;
    callBackData_t *host_args2 = new callBackData_t;
    host_args2->src = (void*)h_bPinned;
    host_args2->dst = (void*)h_bPageable;
    host_args2->expected = (void*)h_expected;
    host_args2->bytes = bytes;
    cudaLaunchHostFunc(streams[1], memcpyHostToHost, host_args2);

  }

    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
    free(h_expected);
    printf("finish main function, everyhing is done\n");
    MPI_Finalize();
  return 0;
}
