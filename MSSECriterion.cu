#include "utils.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct msse_functor
{
  msse_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      float z = x-y;
      return z;
  }
};


static int extracunn_MSSECriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, target));

  luaL_argcheck(L, THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");

  float sum;

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), msse_functor());

  sum *= sum;
  sum /= (-2*size*size);

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


static int extracunn_MSSECriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  luaL_argcheck(L, THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
                "input and target need to have the same number of elements");
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));

  long size = THCudaTensor_nElement(state, input);
  float norm = -1./(size*size);
  float sum = 0.0;

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

   sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), msse_functor());

  sum *= norm;
  thrust::fill(gradInput_data,gradInput_data+size,sum);

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}

#define MSSECRITERION_THREADS 128

__global__ void extracunn_MSSECriterion_updateOutput_kernel(float* output, float *input, float *target, int nframe, int dim)
{
  __shared__ float buffer[MSSECRITERION_THREADS];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // msse
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i] - target_k[i];
    buffer[threadIdx.x] += z;
  }
  __syncthreads();


  //reduce
  if (threadIdx.x == 0)
  {
    *output = 0;
    for (int i=0; i<blockDim.x; i++)
    {
      *output += buffer[i];
    }
    *output *= (*output);
    *output /= (-2*dim*dim);
  }
}

__global__ void extracunn_MSSECriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, float norm, int nframe, int dim)
{
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*dim;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;

  __shared__ float buffer[MSSECRITERION_THREADS];

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;
  float sum = 0.0;
  // msse
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i] - target_k[i];
    buffer[threadIdx.x] += z;
  }
  __syncthreads();


  //reduce
  if (threadIdx.x == 0)
  {
    sum = 0;
    for (int i=0; i<blockDim.x; i++)
    {
      sum += buffer[i];
    }
  }

  // gradInput
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = norm*sum;
}

static int extracunn_MSSECriterion_updateOutput2(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaStorage *output = THCudaStorage_newWithSize(state, 1);

  dim3 blocks(1);
  dim3 threads(MSSECRITERION_THREADS);

  extracunn_MSSECriterion_updateOutput_kernel<<<blocks,threads,
    0, THCState_getCurrentStream(state)>>>(output->data,
                                           THCudaTensor_data(state, input),
                                           THCudaTensor_data(state, target),
                                           1, size);

  lua_pushnumber(L, THCudaStorage_get(state, output, 0));

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  THCudaStorage_free(state, output);

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  return 1;
}

static int extracunn_MSSECriterion_updateGradInput2(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, input);
  float norm = -1./(size*size);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  dim3 blocks(1);
  dim3 threads(MSSECRITERION_THREADS);

  extracunn_MSSECriterion_updateGradInput_kernel<<<blocks,threads,
    0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradInput),
                                           THCudaTensor_data(state, input),
                                           THCudaTensor_data(state, target),
                                           norm,
                                           1, size);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}


static const struct luaL_Reg extracunn_MSSECriterion__ [] = {
  {"MSSECriterion_updateOutput", extracunn_MSSECriterion_updateOutput},
  {"MSSECriterion_updateGradInput", extracunn_MSSECriterion_updateGradInput},
  {"MSSECriterion_updateOutput2", extracunn_MSSECriterion_updateOutput2},
  {"MSSECriterion_updateGradInput2", extracunn_MSSECriterion_updateGradInput2},
  {NULL, NULL}
};

void extracunn_MSSECriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, extracunn_MSSECriterion__, "nn");
  lua_pop(L,1);
}
                                      
