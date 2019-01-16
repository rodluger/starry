/**
Defines the surface map class.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include "utils.h"
#include "errors.h"
#include "solver.h"
#include "limbdark.h"
#include "basis.h"
#include "rotation.h"

namespace starry2 { 
namespace maps {

using namespace utils;

#include "maps/cache.h"
#include "maps/Map.h"
#include "maps/io.h"
#include "maps/oper.h"
#include "maps/flux.h"
#include "maps/python_interface.h"

} // namespace maps
} // namespace starry2
#endif