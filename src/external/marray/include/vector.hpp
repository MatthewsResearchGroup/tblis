#ifndef _MARRAY_VECTOR_HPP_
#define _MARRAY_VECTOR_HPP_

#include <complex>
#include "utility.hpp"

#if __GNUC__ >= 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace MArray
{

template <typename T, typename=void>
struct vector_traits
{
    constexpr static unsigned vector_width = 1;
    constexpr static unsigned alignment = 1;
    typedef T vector_type;
};

}

#if defined(__AVX512F__)

#include "vector_avx512.hpp"

#elif defined(__AVX__)

#include "vector_avx.hpp"

#elif defined(__SSE4_1__)

#include "vector_sse41.hpp"

#else

#include "vector_generic.hpp"

#endif

#if __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

#endif
