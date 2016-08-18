#include "THCUNN.h"

#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"
#include "THCReduceApplyUtils.cuh"

#include "utils.h"
#include "common.h"

__global__ void SpatialMirrorDiagonal_updateOutput(
  THCDeviceTensor<float, 4> input,
  THCDeviceTensor<float, 4> output) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= output.getSize(2) * output.getSize(3)) {
    return;
  }
  int outputPointX = outputPointId % output.getSize(3);
  int outputPointY = outputPointId / output.getSize(3);

  int inputPointX = output.getSize(3) - outputPointX - 1;
  int inputPointY = output.getSize(2) - outputPointY - 1;

  float valueToCopy = input[batch][plane][inputPointY][inputPointX];
  output[batch][plane][outputPointY][outputPointX] = valueToCopy;
}

static int extracunn_SpatialMirrorDiagonal_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int numInputDims = THCudaTensor_nDimension(state, input);
  THArgCheck(numInputDims == 3 || numInputDims == 4, 2,
             "input must be 3 or 4-dimensional");

  if (numInputDims == 4) {
    numBatch = THCudaTensor_size(state, input, 0);
    planeDim++;
    dimh++;
    dimw++;
  }

  int numPlanes = THCudaTensor_size(state, input, planeDim);
  int inputH    = THCudaTensor_size(state, input, dimh);
  int inputW    = THCudaTensor_size(state, input, dimw);
  int outputH   = inputH;
  int outputW   = inputW;

  THCDeviceTensor<float, 4> devInput;
  THCDeviceTensor<float, 4> devOutput;

  if (numInputDims == 3) {
    THCudaTensor_resize3d(state, output, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 3>(state, input).upcastOuter<4>();
    devOutput = toDeviceTensor<float, 3>(state, output).upcastOuter<4>();
  } else {
    THCudaTensor_resize4d(state, output, numBatch, numPlanes, outputH, outputW);

    devInput = toDeviceTensor<float, 4>(state, input);
    devOutput = toDeviceTensor<float, 4>(state, output);
  }

  int outputPlaneSize = devOutput.getSize(2) * devOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devOutput.getSize(1),
            devOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);
  SpatialMirrorDiagonal_updateOutput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devInput, devOutput);

  return 1;
}

__global__ void SpatialMirrorDiagonal_updateGradInput(
  THCDeviceTensor<float, 4> gradInput,
  THCDeviceTensor<float, 4> gradOutput) {

  int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
  int plane = blockIdx.y;
  int batch = blockIdx.z;
  if (outputPointId >= gradOutput.getSize(2) * gradOutput.getSize(3)) {
    return;
  }
  int outputPointX = outputPointId % gradOutput.getSize(3);
  int outputPointY = outputPointId / gradOutput.getSize(3);
  
  int inputPointX = gradOutput.getSize(3) - outputPointX - 1;
  int inputPointY = gradOutput.getSize(2) - outputPointY - 1;

  float valueToCopy = gradOutput[batch][plane][outputPointY][outputPointX];
  //atomicAdd(&gradInput[batch][plane][inputPointY][inputPointX], valueToCopy);
  gradInput[batch][plane][inputPointY][inputPointX] = valueToCopy;
}

static int extracunn_SpatialMirrorDiagonal_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;

  int numInputDims = THCudaTensor_nDimension(state, input);
  if (numInputDims == 4) {
    planeDim++;
    dimh++;
    dimw++;
  }

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  THCDeviceTensor<float, 4> devGradInput;
  THCDeviceTensor<float, 4> devGradOutput;

  if (numInputDims == 3) {
    devGradInput = toDeviceTensor<float, 3>(state, gradInput).upcastOuter<4>();
    devGradOutput = toDeviceTensor<float, 3>(state, gradOutput).upcastOuter<4>();
  } else {
    devGradInput = toDeviceTensor<float, 4>(state, gradInput);
    devGradOutput = toDeviceTensor<float, 4>(state, gradOutput);
  }

  int outputPlaneSize = devGradOutput.getSize(2) * devGradOutput.getSize(3);
  dim3 gridSize(THCCeilDiv(outputPlaneSize, 256),
            devGradOutput.getSize(1),
            devGradOutput.getSize(0));
  dim3 blockSize(outputPlaneSize > 256 ? 256 : outputPlaneSize);

  SpatialMirrorDiagonal_updateGradInput<<<gridSize, blockSize, 0, THCState_getCurrentStream(state)>>>(
    devGradInput, devGradOutput);

  return 1;
}

static const struct luaL_Reg extracunn_SpatialMirrorDiagonal__ [] = {
  {"SpatialMirrorDiagonal_updateOutput", extracunn_SpatialMirrorDiagonal_updateOutput},
  {"SpatialMirrorDiagonal_updateGradInput", extracunn_SpatialMirrorDiagonal_updateGradInput},
  {NULL, NULL}
};

void extracunn_SpatialMirrorDiagonal_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, extracunn_SpatialMirrorDiagonal__, "nn");
  lua_pop(L,1);
}
