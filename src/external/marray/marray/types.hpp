#ifndef MARRAY_TYPES_HPP
#define MARRAY_TYPES_HPP

/** @file */

#include <cstddef>
#include <cstdint>

#include "short_vector.hpp"

/** @defgroup macros Macros */
/** @defgroup constants Constants */
/** @defgroup types Typedefs */
/** @defgroup classes Tensors and Tensor Views */
/** @defgroup range Range Functions */
/** @defgroup util Utilities */
/** @defgroup funcs Global Functions */

/** Main MArray namespace. */
namespace MArray
{

struct all_t { constexpr all_t() {} };
struct bcast_t { constexpr bcast_t() {} };

/** Namespace including slicing constants. */
namespace slice
{
    /**
     * Token used to select all indices along a dimension.
     *
     * @ingroup constants
     */
    constexpr all_t all;
    /**
     * Token used to create a new dimension along which the tensor will be replicated.
     *
     * This can be used to broadcast data into a destination tensor of larger
     * dimensionality, e.g.:
     *
     * @code{.cxx}
     * marray<3> A{3,5,8};
     * marray<4> B{3,6,5,8};
     * B = A[all][bcast][all][all];
     * //now B[i][j][k][l] == A[i][k][l] for all j
     * @endcode
     *
     * @ingroup constants
     */
    constexpr bcast_t bcast;
}

#ifndef MARRAY_LEN_TYPE
/**
 * User-definable macro defining the type of tensor indices.
 *
 * Default is `ptrdiff_t`.
 *
 * @note This type *must* be signed.
 *
 * @ingroup macros
 */
#define MARRAY_LEN_TYPE ptrdiff_t
#endif

/**
 * The integral type used for index lengths.
 *
 * Typically the same as `ptrdiff_t`.
 *
 * @ingroup types
 */
typedef MARRAY_LEN_TYPE len_type;

#ifndef MARRAY_STRIDE_TYPE
/**
 * User-definable macro defining the type of tensor strides.
 *
 * Default is `ptrdiff_t`.
 *
 * @note This type *must* be signed.
 *
 * @ingroup macros
 */
#define MARRAY_STRIDE_TYPE ptrdiff_t
#endif

/**
 * The integral type used for index strides.
 *
 * Typically the same as `ptrdiff_t`.
 *
 * @ingroup types
 */
typedef MARRAY_STRIDE_TYPE stride_type;

#ifndef MARRAY_OPT_NDIM
/**
 * User-definable macro which indicates the highest number of tensor dimensions likely to be encountered.
 *
 * Tensors with a larger number of dimensions may always be created, but dynamic allocation will be used for
 * and dimension-specific data.
 *
 * @ingroup macros
 */
#define MARRAY_OPT_NDIM 8
#endif

using len_vector         = short_vector<len_type,MARRAY_OPT_NDIM>;
using stride_vector      = short_vector<stride_type,MARRAY_OPT_NDIM>;
using dpd_len_vector     = short_vector<std::array<len_type,8>,MARRAY_OPT_NDIM>;
using dpd_stride_vector  = short_vector<std::array<stride_type,8>,MARRAY_OPT_NDIM>;
using dpd_len_vector2    = short_vector<std::array<len_type,8>,2*MARRAY_OPT_NDIM>;
using dpd_stride_vector2 = short_vector<std::array<stride_type,8>,2*MARRAY_OPT_NDIM>;
using dim_vector         = short_vector<int,MARRAY_OPT_NDIM>;
using dim_vector2        = short_vector<int,2*MARRAY_OPT_NDIM>;
using len_vector2        = short_vector<len_type,2*MARRAY_OPT_NDIM>;
using stride_vector2     = short_vector<stride_type,2*MARRAY_OPT_NDIM>;
using index_vector       = short_vector<len_type,MARRAY_OPT_NDIM>;
using irrep_vector       = short_vector<int,MARRAY_OPT_NDIM>;
template <typename T>
using ptr_vector         = short_vector<T*,MARRAY_OPT_NDIM>;

#if MARRAY_DOXYGEN
/**
 * User-definable macro requesting error checking (including full bounds checking).
 *
 * @ingroup macros
 */
#define MARRAY_ENABLE_ASSERTS
#endif

#ifndef MARRAY_DEFAULT_LAYOUT
/**
 * User-definable macro specifying the default tensor layout.
 *
 * `#define` this macro to either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR)
 * before including any MArray headers. Otherwise, the default is [ROW_MAJOR](@ref MArray::ROW_MAJOR).
 * The default can be also overriden when constructing a tensor or view.
 *
 * @ingroup macros
 */
#define MARRAY_DEFAULT_LAYOUT ROW_MAJOR
#endif

#ifndef MARRAY_DEFAULT_BASE
/**
 * User-definable macro specifying the default base for indices.
 *
 * `#define` this macro to either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE)
 * before including any MArray headers. Otherwise, the default is [BASE_ZERO](@ref MArray::BASE_ZERO).
 * The default can be also overriden when constructing a tensor or view.
 *
 * @ingroup macros
 */
#define MARRAY_DEFAULT_BASE BASE_ZERO
#endif

#define MARRAY_PASTE_(x,y) x##y
#define MARRAY_PASTE(x,y) MARRAY_PASTE_(x,y)

#define MARRAY_DEFAULT_DPD_LAYOUT_(type) \
    MARRAY_PASTE(MARRAY_PASTE(type,_),MARRAY_DEFAULT_LAYOUT)

#ifndef MARRAY_DEFAULT_DPD_LAYOUT
#define MARRAY_DEFAULT_DPD_LAYOUT PREFIX
#endif

struct uninitialized_t { constexpr uninitialized_t() {} };

/**
 * A token which indicates not to initialize allocated memory.
 *
 * This special value is used to construct an array which
 * does not default- or value-initialize its elements (useful for avoiding
 * redundant memory operations for scalar types).
 *
 * @ingroup constants
 */
constexpr uninitialized_t uninitialized;

/**
 * Special value which indicates that the number of dimensions is not known at compile time.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr int DYNAMIC;
#else
constexpr int DYNAMIC = -1;
#endif

}

#endif //MARRAY_TYPES_HPP
