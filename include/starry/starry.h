/**
\file starry.h
\brief Entry point to the `starry` C++ interface.

*/

/**
\mainpage starry documentation

Welcome to the starry C++ API documentation. This
page is still under development, and docstrings are
still missing for several routines.

*/

#ifndef _STARRY_H_
#define _STARRY_H_

#include "utils.h"
#include "maps.h"
#include "extensions/extensions.h"

namespace starry {

    using maps::Map;
    using utils::Multi;
    using utils::MapType;

} // namespace starry

#endif