#ifndef EXTRACUNN_UTILS_H
#define EXTRACUNN_UTILS_H

extern "C"
{
#include <lua.h>
}
#include <luaT.h>
#include <THC/THC.h>

THCState* getCutorchState(lua_State* L);
void extracunn_SpatialConvolutionMMNoBias_init(lua_State *L);
void extracunn_Huber_init(lua_State *L);
void extracunn_MSSECriterion_init(lua_State *L);
void extracunn_SpatialSkew_init(lua_State *L);
void extracunn_SpatialUnskew_init(lua_State *L);
void extracunn_SpatialMirrorHorizontal_init(lua_State *L);
void extracunn_SpatialMirrorVertical_init(lua_State *L);
void extracunn_SpatialMirrorDiagonal_init(lua_State *L);

#endif
