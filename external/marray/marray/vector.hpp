#ifndef MARRAY_VECTOR_HPP
#define MARRAY_VECTOR_HPP

#include <complex>

#include "../marray/utility.hpp"

#if __GNUC__ >= 6
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace MArray
{

template <typename T, typename=void>
struct vector_traits
{
    constexpr static int vector_width = 1;
    constexpr static int alignment = 1;
    typedef T vector_type;
};

}

#if defined(__AVX512F__)

#include "../marray/vector_avx512.hpp"

#elif defined(__AVX__)

#include "../marray/vector_avx.hpp"

#elif defined(__SSE4_1__)

#include "../marray/vector_sse41.hpp"

#endif

#if __GNUC__ >= 6
#pragma GCC diagnostic pop
#endif

#endif //MARRAY_VECTOR_HPP
