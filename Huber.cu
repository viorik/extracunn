#include "THCApply.cuh"
#include "utils.h"

struct Huber {
  const float threshold_;

  Huber(float threshold): threshold_(threshold) {}

  __device__ __forceinline__ void operator()(float* x) {
    if (*x > threshold_) *x = threshold_;
    else if ( *x < -threshold_) *x = -threshold_;
    else *x = *x;
  }
};

static int extracunn_Huber(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double threshold = luaL_checknumber(L,2);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 1, "torch.CudaTensor");

  THC_pointwiseApply1(state, input,
                               Huber(threshold));
  
  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg extracunn_Huber__ [] = {
  {"Huber", extracunn_Huber},
  {NULL, NULL}
};

void extracunn_Huber_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaL_register(L,NULL, extracunn_Huber__);
  lua_pop(L,1);
}

