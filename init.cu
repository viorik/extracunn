#include "luaT.h"
#include "TH.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libextracunn(lua_State *L);

int luaopen_libextracunn(lua_State *L)
{
  lua_newtable(L);
  extracunn_SpatialConvolutionMMNoBias_init(L);
  extracunn_Huber_init(L);

  return 1;
}
