#ifndef MARRAY_MARRAY_VIEW_HPP
#define MARRAY_MARRAY_VIEW_HPP

#include "detail/utility.hpp"
#include "marray_base.hpp"

namespace MArray
{

template <typename Type, int NDim>
class marray_view : public marray_base<Type, NDim, marray_view<Type, NDim>, false>
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
    template <typename, int, int, typename...> friend class marray_slice;

    protected:
        typedef marray_base<Type, NDim, marray_view, false> base_class;

        using base_class::base_;
        using base_class::len_;
        using base_class::stride_;
        using base_class::data_;

#ifdef MARRAY_ENABLE_ASSERTS
        using base_class::bbox_len_;
        using base_class::bbox_off_;
        using base_class::bbox_stride_;
        using base_class::bbox_data_;
#endif

        using typename base_class::base_like;
        using typename base_class::layout_like;

    public:
        using typename base_class::value_type;
        using typename base_class::pointer;
        using typename base_class::const_pointer;
        using typename base_class::reference;
        using typename base_class::const_reference;

        /***********************************************************************
         *
         * @name Constructors
         *
         **********************************************************************/
        /** @{ */

        /**
         * Construct an empty view.
         */
        marray_view() {}

        /**
         * Construct a view of the given tensor, view, or partially-indexed tensor.
         *
         * @param other     The tensor, view, or partially-indexed tensor to view.
         *                  If this is a mutable view (the value type is not
         *                  const-qualified), then `other` may not be a const-
         *                  qualified tensor instance or a view with a const-
         *                  qualified value type. May be either an lvalue- or
         *                  rvalue-reference.
         */
#if MARRAY_DOXYGEN
        marray_view(tensor_or_view other);
#else
        marray_view(const marray_view& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray_view(const marray_base<U, N, D, O>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray_view(marray_base<U, N, D, O>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray_view(marray_base<U, N, D, O>&& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        marray_view(const marray_slice<U, N, I, D...>& other)
        {
            reset(other);
        }
#endif

        /**
         * Construct a view that wraps an immutable std::vector of compatible type.
         *
         * @note        Only enable for one-dimensional views.
         *
         * @tparam T    Type such that `const T*` is convertible to `Type*`.
         *
         * @param  v    The std::vector to wrap.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        marray_view(const std::vector<T>& v, std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<const T*,pointer>>* = nullptr)
#else
        marray_view(const std::vector<T>& v)
#endif
        {
            reset(v);
        }

        /**
         * Construct a view that wraps a mutable std::vector of compatible type.
         *
         * @note        Only enable for one-dimensional views.
         *
         * @tparam T    Type such that `T*` is convertible to `Type*`.
         *
         * @param v     The std::vector to wrap.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        marray_view(std::vector<T>& v, std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<T*,pointer>>* = nullptr)
#else
        marray_view(std::vector<T>& v)
#endif
        {
            reset(v);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided shape, and the default base and layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         */
        marray_view(const array_1d<len_type>& len, pointer ptr)
        {
            reset(len, ptr);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided shape and base, and the default layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param base  One of [BASE_ZERO](@ref MArray::BASE_ZERO), [BASE_ONE](@ref MArray::BASE_ONE), [FORTRAN](@ref MArray::FORTRAN), or [MATLAB](@ref MArray::MATLAB).
         */
        marray_view(const array_1d<len_type>& len, pointer ptr, const index_base& base)
        {
            reset(len, ptr, base);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided shape and layout, and the default base.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param stride    The strides along each dimension, or a layout (one of [ROW_MAJOR](@ref MArray::ROW_MAJOR), [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR),
         *                  [FORTRAN](@ref MArray::FORTRAN), [MATLAB](@ref MArray::MATLAB)). The stride is the distance
         *                  in memory (in units of the value type) between successive
         *                  elements along this direction. In general, the strides need
         *                  not be defined such that elements have unique locations,
         *                  although such a view should not be written into. Strides may
         *                  also be negative. In this case, `ptr` still refers to the
         *                  location of the "first" element (`index == base`), although this
         *                  is not the lowest address of any tensor element.
         */
        marray_view(const array_1d<len_type>& len, pointer ptr, const layout_like& stride)
        {
            reset(len, ptr, stride);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided
         * shape, base, and layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param base  The base for each index, or a standard base (one of [BASE_ZERO](@ref MArray::BASE_ZERO), [BASE_ONE](@ref MArray::BASE_ONE),
         *              [FORTRAN](@ref MArray::FORTRAN), [MATLAB](@ref MArray::MATLAB)). The base is the minimum value of the index. For example,
         *              a tensor with lengths (2,3,4) and base (1,6,0) can have valid indices along the three
         *              dimensions in the ranges [1,3), [6,9), and [0,4), respectively.
         *
         * @param stride    The strides along each dimension, or a layout (one of [ROW_MAJOR](@ref MArray::ROW_MAJOR), [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR),
         *                  [FORTRAN](@ref MArray::FORTRAN), [MATLAB](@ref MArray::MATLAB)). The stride is the distance
         *                  in memory (in units of the value type) between successive
         *                  elements along this direction. In general, the strides need
         *                  not be defined such that elements have unique locations,
         *                  although such a view should not be written into. Strides may
         *                  also be negative. In this case, `ptr` still refers to the
         *                  location of the "first" element (`index == base`), although this
         *                  is not the lowest address of any tensor element.
         */
        marray_view(const array_1d<len_type>& len, pointer ptr, const base_like& base,
                    const layout_like& stride)
        {
            reset(len, ptr, base, stride);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided shape and standard FORTRAN/MATLAB indexing.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param fortran   Either the token [FORTRAN](@ref MArray::FORTRAN) or [MATLAB](@ref MArray::MATLAB).
         */
 #if MARRAY_DOXYGEN
        marray_view(const array_1d<len_type>& len, pointer ptr, fortran_t fortran)
#else
        marray_view(const array_1d<len_type>& len, pointer ptr, fortran_t)
#endif
        {
            reset(len, ptr, FORTRAN);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided extents and the default layout.
         *
         * @note This constructor provides functionality similar to the multidimensional array declarations in
         *       FORTRAN, e.g. `real, dimension(begin1:end1, begin2:end2, ...) :: array`, except that the upper
         *       bounds (`end`) are one greater than in the corresponding FORTRAN statement. This is by design such
         *       that the basic constructors with only `len` are equivalent to `begin = [0,...]` and `end = len`.
         *
         * @param begin   The smallest values of each index. May be any one-
         *                dimensional container whose elements are convertible to
         *                tensor lengths, including initializer lists. These values are the same
         *                as the tensor @ref base().
         *
         * @param end   One plus the largest values of each index. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists. The length of index `i` is
         *              equal to `end[i]-begin[i]`.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         */
        marray_view(const array_1d<len_type>& begin, const array_1d<len_type>& end, pointer ptr)
        {
            reset(begin, end, ptr);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided extents and layout.
         *
         * @note This constructor provides functionality similar to the multidimensional array declarations in
         *       FORTRAN, e.g. `real, dimension(begin1:end1, begin2:end2, ...) :: array`, except that the upper
         *       bounds (`end`) are one greater than in the corresponding FORTRAN statement. This is by design such
         *       that the basic constructors with only `len` are equivalent to `begin = [0,...]` and `end = len`.
         *
         * @param begin   The smallest values of each index. May be any one-
         *                dimensional container whose elements are convertible to
         *                tensor lengths, including initializer lists. These values are the same
         *                as the tensor @ref base().
         *
         * @param end   One plus the largest values of each index. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists. The length of index `i` is
         *              equal to `end[i]-begin[i]`.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param stride    The strides along each dimension, or a layout (one of [ROW_MAJOR](@ref MArray::ROW_MAJOR), [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR),
         *                  [FORTRAN](@ref MArray::FORTRAN), [MATLAB](@ref MArray::MATLAB)). The stride is the distance
         *                  in memory (in units of the value type) between successive
         *                  elements along this direction. In general, the strides need
         *                  not be defined such that elements have unique locations,
         *                  although such a view should not be written into. Strides may
         *                  also be negative. In this case, `ptr` still refers to the
         *                  location of the "first" element (`index == begin`), although this
         *                  is not the lowest address of any tensor element.
         */
        marray_view(const array_1d<len_type>& begin, const array_1d<len_type>& end, pointer ptr, const layout_like& stride)
        {
            reset(begin, end, ptr, stride);
        }

        /** @} */
        /***********************************************************************
         *
         * @name Reset
         *
         **********************************************************************/
        /** @{ */

        using base_class::reset;

        /** @} */
        /***********************************************************************
         *
         * @name Static helper functions
         *
         **********************************************************************/
        /** @{ */

        using base_class::size;
        using base_class::is_contiguous;

        /** @} */
        /***********************************************************************
         *
         * @name Operators
         *
         **********************************************************************/
        /** @{ */

        marray_view& operator=(const marray_view& other)
        {
            return base_class::operator=(other);
        }

        using base_class::operator=;
        using base_class::operator+=;
        using base_class::operator-=;
        using base_class::operator*=;
        using base_class::operator/=;
        using base_class::operator==;

        /** @} */
        /***********************************************************************
         *
         * @name Views
         *
         **********************************************************************/
        /** @{ */

        using base_class::cview;
        using base_class::view;

        /** @} */
        /***********************************************************************
         *
         * @name Iterators
         *
         **********************************************************************/
        /** @{ */

        using base_class::cbegin;
        using base_class::begin;
        using base_class::cend;
        using base_class::end;
        using base_class::crbegin;
        using base_class::rbegin;
        using base_class::crend;
        using base_class::rend;

        /** @} */
        /***********************************************************************
         *
         * @name Shift operations
         *
         **********************************************************************/
        /** @{ */

        using base_class::shifted;
        using base_class::shifted_up;
        using base_class::shifted_down;
        using base_class::rebased;

        /** @} */
        /***********************************************************************
         *
         * @name Permutation
         *
         **********************************************************************/
        /** @{ */

        using base_class::permuted;
        using base_class::transposed;
        using base_class::T;

        /** @} */
        /***********************************************************************
         *
         * @name Dimension change
         *
         **********************************************************************/
        /** @{ */

        using base_class::lowered;

        /** @} */
        /***********************************************************************
         *
         * @name Reversal
         *
         **********************************************************************/
        /** @{ */

        using base_class::reversed;

        /** @} */
        /***********************************************************************
         *
         * @name Slices
         *
         **********************************************************************/
        /** @{ */

        using base_class::cslice;
        using base_class::slice;
        using base_class::cfront;
        using base_class::front;
        using base_class::cback;
        using base_class::back;

        /** @} */
        /***********************************************************************
         *
         * @name Indexing
         *
         **********************************************************************/
        /** @{ */

        using base_class::operator[];
        using base_class::operator();

        /** @} */
        /***********************************************************************
         *
         * @name Element-wise iteration
         *
         **********************************************************************/
        /** @{ */

        using base_class::for_each_element;

        /** @} */
        /***********************************************************************
         *
         * @name Basic getters
         *
         **********************************************************************/
        /** @{ */

        using base_class::corigin;
        using base_class::origin;
        using base_class::cdata;
        using base_class::data;
        using base_class::base;
        using base_class::bases;
        using base_class::length;
        using base_class::lengths;
        using base_class::stride;
        using base_class::strides;
        using base_class::dimension;

        /** @} */
        /***********************************************************************
         *
         * @name In-place shift operations
         *
         **********************************************************************/
        /** @{ */

        /**
         * Shift this view along each dimension by the amount specified.
         *
         * The `k`th index `i` in the shifted view is equivalent to an index `i+n[k]`
         * in the original view.
         *
         * @param n The amount by which to shift for each dimension. May be any
         *          one-dimensional container type whose elements are convertible
         *          to a tensor length, including initializer lists.
         */
        void shift(const array_1d<len_type>& n)
        {
            MARRAY_ASSERT(n.size() == dimension());

            std::array<len_type, NDim> n_;
            n.slurp(n_);

            for (auto dim : range(dimension()))
                shift(dim, n_[dim]);
        }

        /**
         * Shift this view by the given amount along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view.
         *
         * @param dim   The dimension along which to shift the view.
         *
         * @param n     The amount by which to shift.
         */
        void shift(int dim, len_type n)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            data_ += n * stride(dim);

#ifdef MARRAY_ENABLE_ASSERTS
            bbox_off_[dim] += n * stride(dim) / bbox_stride_[dim];
            MARRAY_ASSERT(bbox_off_[dim] >= 0);
            MARRAY_ASSERT(bbox_off_[dim] + length(dim) * std::abs(stride(dim)) / bbox_stride_[dim] <= bbox_len_[dim]);
#endif
        }

        /**
         * Shift this view by the given amount along one dimension and then resize.
         *
         * The effect is the same as a combination of @ref shift(int, len_type)
         * and @ref length(int, len_type), but the view may take on an invalid
         * intermediate state.
         *
         * @param dim   The dimension along which to shift the view.
         *
         * @param n     The amount by which to shift.
         *
         * @param len   The new size along the indicated dimension.
         */
        void shift_and_resize(int dim, len_type n, len_type len)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(len >= 0);
            data_ += n * stride(dim);
            len_[dim] = len;

#ifdef MARRAY_ENABLE_ASSERTS
            bbox_off_[dim] += n * stride(dim) / bbox_stride_[dim];
            MARRAY_ASSERT(bbox_off_[dim] >= 0);
            MARRAY_ASSERT(bbox_off_[dim] + length(dim) * std::abs(stride(dim)) / bbox_stride_[dim] <= bbox_len_[dim]);
#endif
        }

        /**
         * Shift this view "down" along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim  The dimension along which to shift the view.
         */
        void shift_down(int dim)
        {
            shift(dim, length(dim));
        }

        /**
         * Shift this view "up" along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim  The dimension along which to shift the view.
         */
        void shift_up(int dim)
        {
            shift(dim, -length(dim));
        }

        /**
         * Shift this view "down" along one dimension and resize.
         *
         * The effect is the same as a combination of @ref shift_down(int)
         * and @ref length(int, len_type), but the view may take on an invalid
         * intermediate state.
         *
         * @param dim  The dimension along which to shift the view.
         *
         * @param len  The new length along the indicated dimension.
         */
        void next(int dim, len_type len)
        {
            shift_and_resize(dim, length(dim), len);
        }

        /**
         * Shift this view "up" along one dimension and resize.
         *
         * The effect is the same as a combination of @ref shift(int, len_type)
         * (by an amount equal to `-len`)
         * and @ref length(int, len_type), but the view may take on an invalid
         * intermediate state.
         *
         * @param dim  The dimension along which to shift the view.
         *
         * @param len  The new length along the indicated dimension.
         */
        void prev(int dim, len_type len)
        {
            shift_and_resize(dim, -len, len);
        }

        /** @} */
        /***********************************************************************
         *
         * @name In-place permutation
         *
         **********************************************************************/
        /** @{ */

        /**
         * Permute this view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original view.
         *
         * @param perm  The permutation vector. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists. The values must form
         *              a permutation of `[0,NDim)`, where `NDim` is the number of
         *              tensor dimensions.
         */
        void permute(const array_1d<int>& perm)
        {
            MARRAY_ASSERT(perm.size() == dimension());

            auto len = len_;
            auto stride = stride_;
            dim_vector perm_;
            perm.slurp(perm_);

#ifdef MARRAY_ENABLE_ASSERTS
            auto bbox_len = bbox_len_;
            auto bbox_off = bbox_off_;
            auto bbox_stride = bbox_stride_;
#endif

            for (auto i : range(dimension()))
            {
                MARRAY_ASSERT(perm_[i] < dimension());
                for (auto j : range(i+1,dimension()))
                    MARRAY_ASSERT(perm_[i] != perm_[j]);

                len_[i] = len[perm_[i]];
                stride_[i] = stride[perm_[i]];

#ifdef MARRAY_ENABLE_ASSERTS
                bbox_len_[i] = bbox_len[perm_[i]];
                bbox_off_[i] = bbox_off[perm_[i]];
                bbox_stride_[i] = bbox_stride[perm_[i]];
#endif
            }
        }

        /**
         * Permute this view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original view.
         *
         * @param perm  The permutation vector. May be any
         *              set of integral types convertible
         *              to `int`. The values must form
         *              a permutation of `[0,NDim)`, where `NDim` is the number of
         *              tensor dimensions.
         */
#if MARRAY_DOXYGEN
        void permute(const Perm&... perm)
#else
        template <typename... Perm>
        std::enable_if_t<detail::are_convertible<int,Perm...>::value>
        permute(const Perm&... perm)
#endif
        {
            static_assert(sizeof...(Perm) == NDim || NDim == DYNAMIC);
            permute({(int)perm...});
        }

        /**
         * Transpose this view.
         *
         * This overload is only available for matrix views.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=std::enable_if_t<N==2>>
#endif
        void transpose()
        {
            permute({1, 0});
        }

        /** @} */
        /***********************************************************************
         *
         * @name In-place dimension change
         *
         **********************************************************************/
        /** @{ */

        /**
         * Reduce to a view of lower dimensionality.
         *
         * The values along each lowered dimension (which corresponds to one or
         * more dimensions in the original tensor or tensor view) must have a
         * consistent stride (i.e. those dimensions must be contiguous). The base of an index which is
         * formed by combining multiple indices in the original view is equal to the base of index
         * with smallest stride.
         *
         * This overload is only available for dynamic views (`NDim == ` [DYNAMIC](@ref MArray::DYNAMIC)).
         *
         * @param split The "split" or "pivot" vector. The number of split points/pivots
         *              must be equal to the number of dimensions in the lowered view
         *              minus one. Dimensions `[0,split[0])` correspond to the
         *              first dimension of the return view, dimensions `[split[K-1],N)`
         *              correspond to the last dimension of the returned view, and
         *              dimensions `[split[i-1],split[i])` correspond to the `i`th
         *              dimension of the return view otherwise, where `N` is the
         *              dimensionality of the original tensor and `K` is the number of dimensions in the
         *              lower-dimensional view. The split points must be
         *              in increasing order. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=std::enable_if_t<N==DYNAMIC>>
#endif
        void lower(const array_1d<int>& split)
        {
            reset(lowered(split));
        }

        /** @} */
        /***********************************************************************
         *
         * @name In-place reversal
         *
         **********************************************************************/
        /** @{ */

        /**
         * Reverse the order of the indices along each dimension.
         *
         * An index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original view, where `n` is the tensor length along
         * that dimension.
         */
        void reverse()
        {
            for (auto i : range(dimension()))
                reverse(i);
        }

        /**
         * Reverse the order of the indices along the specified dimension.
         *
         * For the indicated dimension, an index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original view, where `n` is the tensor length along
         * that dimension.
         *
         * @param dim  The dimension along which to reverse the indices.
         */
        void reverse(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            data_ += (length(dim)-1) * stride(dim);
            stride_[dim] = -stride_[dim];
        }

        /** @} */
        /***********************************************************************
         *
         * @name Basic setters
         *
         **********************************************************************/
        /** @{ */

        /**
         * Set the tensor data pointer.
         *
         * @param ptr  The new data pointer.
         *
         * @returns    The old data pointer.
         */
        pointer data(pointer ptr)
        {
            std::swap(ptr, data_);
            return ptr;
        }

        /**
         * Set the tensor length along the specified dimension.
         *
         * @param dim   The dimension whose length to change.
         *
         * @param len   The new length.
         *
         * @returns     The old length.
         */
        len_type length(int dim, len_type len)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(len >= 0);
            std::swap(len, len_[dim]);
#ifdef MARRAY_ENABLE_ASSERTS
            MARRAY_ASSERT(bbox_off_[dim] + length(dim) * std::abs(stride(dim)) / bbox_stride_[dim] <= bbox_len_[dim]);
#endif
            return len;
        }

        /**
         * Prepare the view for iteration "down" one dimension.
         *
         * @param dim   The dimension to be iterated over.
         */
        void first(int dim)
        {
            length(dim, 0);
        }

        /**
         * Prepare the view for iteration "up" one dimension.
         *
         * @param dim   The dimension to be iterated over.
         */
        void last(int dim)
        {
            next(dim, 0);
        }

        /**
         * Set the tensor stride along the specified dimension.
         *
         * @param dim   The dimension whose stride to change.
         *
         * @param str   The new stride.
         *
         * @returns     The old stride.
         */
        stride_type stride(int dim, stride_type str)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            std::swap(str, stride_[dim]);
#ifdef MARRAY_ENABLE_ASSERTS
            MARRAY_ASSERT(std::abs(stride(dim)) % bbox_stride_[dim] == 0);
            MARRAY_ASSERT(bbox_off_[dim] + length(dim) * std::abs(stride(dim)) / bbox_stride_[dim] <= bbox_len_[dim]);
#endif
            return str;
        }

        /** @} */
        /***********************************************************************
         *
         * @name Swap
         *
         **********************************************************************/
        /** @{ */

        /**
         * Swap this view with another.
         *
         * @param other The view to swap with.
         */
        void swap(marray_view& other)
        {
            base_class::swap(other);
        }

        /** @} */
};

/**
 * Swap two views.
 *
 * @param a      The first view to swap.
 *
 * @param b      The second view to swap.
 *
 * @ingroup funcs
 */
template <typename Type, int NDim>
void swap(marray_view<Type, NDim>& a, marray_view<Type, NDim>& b)
{
    a.swap(b);
}

}

#endif //MARRAY_MARRAY_VIEW_HPP
