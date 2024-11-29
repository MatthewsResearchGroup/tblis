#ifndef MARRAY_MARRAY_FWD_HPP
#define MARRAY_MARRAY_FWD_HPP

#include <complex>
#include <memory>

#include "../types.hpp"

namespace MArray
{
/**
 * A partially-indexed tensor.
 *
 * This type cannot be constructed directly, but is returned by indexing a tensor
 * or tensor view. Provides a limited API for:
 *
 * - Further indexing, slicing, or broadcasting
 * - Obtaining the data pointer
 * - Creating a view from the currently-indexed portion
 * - Participating in element-wise expressions (including as the left-hand side)
 *
 * In the latter two cases, the resulting view or expression leaves any unindexed dimensions intact, i.e. it
 * is as if the remaining dimensions were indexed with `[slice::all]`.
 *
 * @ingroup classes
 */
template <typename Type, int NDim, int NIndexed, typename... Dims>
class marray_slice;

/**
 * Tensor base class.
 *
 * This class should not be used directly. Use @ref marray and @ref marray_view instead.
 *
 * @ingroup classes
 */
template <typename Type, int NDim, typename Derived, bool Owner>
class marray_base;

/**
 * A tensor (multi-dimensional array) view, which may either be mutable or immutable.
 *
 * @tparam Type     The type of the tensor elements. The view is immutable if this is const-qualified.
 *
 * @tparam NDim     The number of tensor dimensions, must be positive or DYNAMIC. Default is DYNAMIC.
 *
 * @ingroup classes
 */
template <typename Type, int NDim=DYNAMIC>
class marray_view;

/**
 * A tensor (multi-dimensional array) container.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @tparam NDim         The number of tensor dimensions, must be positive.
 *
 * @tparam Allocator    An allocator. If not specified, `std::allocator<Type>` is used.
 *
 * @ingroup classes
 */
template <typename Type, int NDim=DYNAMIC, typename Allocator=std::allocator<Type>>
class marray;

/**
 * Alias for a 1-dimensional tensor view.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @ingroup types
 */
template <typename Type> using row_view = marray_view<Type, 1>;

/**
 * Alias for a 1-dimensional tensor.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @tparam Allocator    An allocator. If not specified, `std::allocator<Type>` is used.
 *
 * @ingroup types
 */
template <typename Type, typename Allocator=std::allocator<Type>> using row = marray<Type, 1, Allocator>;

/**
 * Alias for a 2-dimensional tensor view.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @ingroup types
 */
template <typename Type> using matrix_view = marray_view<Type, 2>;

/**
 * Alias for a 2-dimensional tensor.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @tparam Allocator    An allocator. If not specified, `std::allocator<Type>` is used.
 *
 * @ingroup types
 */
template <typename Type, typename Allocator=std::allocator<Type>> using matrix = marray<Type, 2, Allocator>;

/**
 * Type specifying one of the pre-defined tensor layouts. The pre-defined layouts place tensor elements
 * in continguous locations in memory; more flexible layouts can be achieved by explicitly specifying the
 * strides of the tensor indices.
 *
 * @ingroup types
 */
struct layout
{
    struct construct {};

    int type;

    constexpr explicit layout(int type, construct) : type(type) {}

    bool operator==(const layout&) const = default;
};

/**
 * Specifies that elements should be laid out in column-major order.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr layout COLUMN_MAJOR;
#else
constexpr layout COLUMN_MAJOR{0, layout::construct{}};
#endif

/**
 * Specifies that elements should be laid out in row-major order.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr layout ROW_MAJOR;
#else
constexpr layout ROW_MAJOR{1, layout::construct{}};
#endif

/**
 * The default layout for tensors (either row- or column-major).
 * This value is controlled by the macro @ref MARRAY_DEFAULT_LAYOUT.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr layout DEFAULT_LAYOUT;
#else
constexpr layout DEFAULT_LAYOUT{MARRAY_DEFAULT_LAYOUT};
#endif

/**
 * Type specifying one of the pre-defined tensor bases. The pre-defined bases index all tensor dimensions starting
 * at either zero or one. More flexible bases can be obtained by explicitly specifying the base for each tensor
 * dimension.
 *
 * @ingroup types
 */
struct index_base
{
    struct construct {};

    int type;

    constexpr explicit index_base(int type, construct) : type(type) {}

    bool operator==(const index_base&) const = default;
};

/**
 * Specifies that indices should start at 0.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr index_base BASE_ZERO;
#else
constexpr index_base BASE_ZERO{0, index_base::construct{}};
#endif

/**
 * Specifies that indices should start at 1.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr index_base BASE_ONE;
#else
constexpr index_base BASE_ONE{1, index_base::construct{}};
#endif

struct c_cxx_t
{
    operator layout() { return ROW_MAJOR; }
    operator index_base() { return BASE_ZERO; }
};

struct fortran_t
{
    operator layout() { return COLUMN_MAJOR; }
    operator index_base() { return BASE_ONE; }
};

/**
 * Specifies row-major layout and base-0 indices. Convertible to either [layout](@ref MArray::layout) or [index_base](@ref MArray::index_base).
 * Same as @ref CXX.
 *
 * @ingroup constants
 */
constexpr c_cxx_t C;

/**
 * Specifies row-major layout and base-0 indices. Convertible to either [layout](@ref MArray::layout) or [index_base](@ref MArray::index_base).
 * Same as @ref C.
 *
 * @ingroup constants
 */
constexpr c_cxx_t CXX;

/**
 * Specifies column-major layout and base-1 indices. Convertible to either [layout](@ref MArray::layout) or [index_base](@ref MArray::index_base).
 * Same as [MATLAB](@ref MArray::MATLAB).
 *
 * @ingroup constants
 */
constexpr fortran_t FORTRAN;

/**
 * Specifies column-major layout and base-1 indices. Convertible to either [layout](@ref MArray::layout) or [index_base](@ref MArray::index_base).
 * Same as [FORTRAN](@ref MArray::FORTRAN).
 *
 * @ingroup constants
 */
constexpr fortran_t MATLAB;

/**
 * The default base for indices (either 0 or 1). This value is controlled by the macro @ref MARRAY_DEFAULT_BASE.
 *
 * @ingroup constants
 */
#if MARRAY_DOXYGEN
constexpr index_base DEFAULT_BASE;
#else
constexpr index_base DEFAULT_BASE{MARRAY_DEFAULT_BASE};
#endif

}

#endif //MARRAY_MARRAY_FWD_HPP
