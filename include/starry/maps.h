/**
\file maps.h
\brief Defines the surface map classes.

*/

#ifndef _STARRY_MAPS_H_
#define _STARRY_MAPS_H_

#include "utils.h"
#include "solvers/emitted.h"
#include "solvers/reflected.h"
#include "solvers/limbdarkened.h"
#include "basis.h"
#include "wigner.h"

namespace starry { 
namespace maps {

using namespace utils;

#include "data.h"
#include "Map/Map.h"

} // namespace maps
} // namespace starry
#endif