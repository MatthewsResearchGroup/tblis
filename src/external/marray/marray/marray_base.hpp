#ifndef MARRAY_MARRAY_BASE_HPP
#define MARRAY_MARRAY_BASE_HPP

#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

#include "array_1d.hpp"
#include "detail/utility.hpp"
#include "marray_slice.hpp"
#include "index_iterator.hpp"

#include "fwd/expression_fwd.hpp"

namespace MArray
{

namespace detail
{

inline stride_vector strides(const array_1d<len_type>& len, layout layout = DEFAULT_LAYOUT)
{
    len_vector len_;
    len.slurp(len_);

    MARRAY_ASSERT(len_.size() > 0);

    int ndim = len_.size();
    stride_vector stride(ndim);

    if (layout == ROW_MAJOR)
    {
        stride[ndim-1] = 1;
        for (auto i : reversed_range(ndim-1))
            stride[i] = stride[i+1]*len_[i+1];
    }
    else
    {
        stride[0] = 1;
        for (auto i : range(1,ndim))
            stride[i] = stride[i-1]*len_[i-1];
    }

    return stride;
}

template <typename Type, int NDim>
struct layout_like : protected array_1d<stride_type>
{
    struct no_layout : layout { constexpr no_layout() : layout(-1, construct{}) {} };

    layout layout_ = no_layout{};

    layout_like(layout layout) : layout_(layout) {}

    using array_1d<stride_type>::array_1d;
    using array_1d<stride_type>::size;

    void stride(const detail::array_type_t<len_type, NDim>& len,
                detail::array_type_t<stride_type, NDim>& strides) const
    {
        if (*this)
        {
            slurp(strides);
        }
        else
        {
            detail::assign(strides, detail::strides(len, layout_));
        }
    }

    detail::array_type_t<stride_type, NDim> stride(const detail::array_type_t<len_type, NDim>& len) const
    {
        detail::array_type_t<stride_type, NDim> strides;
        stride(len, strides);
        return strides;
    }

    template <typename Type_, int NDim_>
    operator layout_like<Type_, NDim_>&() const
    {
        return reinterpret_cast<const layout_like<Type_, NDim_>&>(*this);
    }

    explicit operator bool() const
    {
        return layout_ == no_layout{};
    }
};

template <typename Type, int NDim>
struct base_like : protected array_1d<len_type>
{
    struct no_base : index_base { constexpr no_base() : index_base(-1, construct{}) {} };

    index_base base_ = no_base{};

    base_like(index_base base) : base_(base) {}

    using array_1d<len_type>::array_1d;
    using array_1d<len_type>::size;

    void base(const detail::array_type_t<len_type, NDim>& len,
              detail::array_type_t<len_type, NDim>& base) const
    {
        if (*this)
        {
            slurp(base);
        }
        else
        {
            if constexpr (NDim == DYNAMIC) base.resize(len.size());
            std::fill(base.begin(), base.end(), base_ == BASE_ZERO ? 0 : 1);
        }
    }

    detail::array_type_t<len_type, NDim> base(const detail::array_type_t<len_type, NDim>& len) const
    {
        detail::array_type_t<len_type, NDim> bases;
        base(len, bases);
        return bases;
    }

    template <typename Type_, int NDim_>
    operator base_like<Type_, NDim_>&() const
    {
        return reinterpret_cast<const base_like<Type_, NDim_>&>(*this);
    }

    explicit operator bool() const
    {
        return base_ == no_base{};
    }
};

}

template <typename Type, int NDim, typename Derived, bool Owner>
class marray_base
{
    static_assert(NDim > 0 || NDim == DYNAMIC, "NDim must be positive or DYNAMIC");

    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
    template <typename, int, int, typename...> friend class marray_slice;

    public:
        /***********************************************************************
         *
         * @name Typedefs
         *
         **********************************************************************/
        /** @{ */

        /** The type of tensor elements. */
        typedef Type value_type;
        /** The type of a pointer to a tensor element. Maybe mutable or immutable depending
         *  on the const-qualification of `Type`. */
        typedef Type* pointer;
        /** The type of an immutable pointer to a tensor element. */
        typedef const Type* const_pointer;
        /** The type of a reference to a tensor element. Maybe mutable or immutable depending
         *  on the const-qualification of `Type`. */
        typedef Type& reference;
        /** The type of an immutable reference to a tensor element. */
        typedef const Type& const_reference;

        /** @} */

        typedef typename detail::initializer_type<Type, NDim>::type
            initializer_type;
        typedef std::conditional_t<NDim == DYNAMIC,void,
            marray_iterator<marray_base>> iterator;
        typedef std::conditional_t<NDim == DYNAMIC,void,
            marray_iterator<const marray_base>> const_iterator;
        typedef std::conditional_t<NDim == DYNAMIC,void,
            std::reverse_iterator<iterator>> reverse_iterator;
        typedef std::conditional_t<NDim == DYNAMIC,void,
            std::reverse_iterator<const_iterator>> const_reverse_iterator;

        typedef typename std::conditional<Owner,const Type,Type>::type ctype;
        typedef ctype& cref;
        typedef ctype* cptr;

    protected:
        using layout_like = detail::layout_like<Type, NDim>;
        using base_like = detail::base_like<Type, NDim>;
        detail::array_type_t<len_type, NDim> base_ = {};
        detail::array_type_t<len_type, NDim> len_ = {};
        detail::array_type_t<stride_type, NDim> stride_ = {};
        pointer data_ = nullptr;

#ifdef MARRAY_ENABLE_ASSERTS

        detail::array_type_t<len_type, NDim> bbox_len_ = {};
        detail::array_type_t<len_type, NDim> bbox_off_ = {};
        detail::array_type_t<stride_type, NDim> bbox_stride_ = {};
        const_pointer bbox_data_ = nullptr;

        template <typename U, int N, typename D, bool O>
        void set_bbox_(const marray_base<U,N,D,O>& other)
        {
            /*
             * The bounding box may have already been set by inherit_bbox_. If so, we shouldn't overwrite it.
             */
            if (bbox_data_)
                return;

            bbox_data_ = other.data();
            detail::assign(bbox_len_, other.lengths());
            bbox_off_ = bbox_len_;
            std::fill_n(bbox_off_.begin(), bbox_off_.size(), 0);
            detail::assign(bbox_stride_, other.strides());
            for (auto& s : bbox_stride_)
                s = std::max(std::abs(s), stride_type(1));
        }

        template <typename U, int N, typename D, bool O>
        void inherit_bbox_(const marray_base<U,N,D,O>& other)
        {
            bbox_data_ = other.bbox_data_;
            detail::assign(bbox_len_, other.bbox_len_);
            detail::assign(bbox_off_, other.bbox_off_);
            detail::assign(bbox_stride_, other.bbox_stride_);
        }

#else

        template <typename U, int N, typename D, bool O>
        void set_bbox_(const marray_base<U,N,D,O>& other) {}

        template <typename U, int N, typename D, bool O>
        void inherit_bbox_(const marray_base<U,N,D,O>& other) {}

#endif

        /***********************************************************************
         *
         * @name Reset
         *
         **********************************************************************/
        /** @{ */

        /**
         * Reset to an empty view.
         */
        void reset()
        {
            data_ = nullptr;
            base_.clear();
            len_.clear();
            stride_.clear();
        }

        /**
         * Reset to a view that wraps an immutable std::vector of compatible type.
         *
         * @note        Only enabled for one-dimensional views.
         *
         * @tparam T    Type such that `const T*` is convertible to `Type*`.
         *
         * @param v     The std::vector to wrap.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<const T*,pointer>>
#else
        void
#endif
        reset(const std::vector<T>& v)
        {
            reset({v.size()}, v.data(), {1});
        }

        /**
         * Reset to a view that wraps a mutable std::vector of compatible type.
         *
         * @note        Only enabled for one-dimensional views.
         *
         * @tparam T    Type such that `T*` is convertible to `Type*`.
         *
         * @param v     The std::vector to wrap.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<T*,pointer>>
#else
        void
#endif
        reset(std::vector<T>& v)
        {
            reset({v.size()}, v.data(), {1});
        }

        /**
         * Reset to a view of the given tensor, view, or partially-indexed tensor.
         *
         * @param other     The tensor, view, or partially-indexed tensor to view.
         *                  If this is a mutable view (the value type is not
         *                  const-qualified), then `other` may not be a const-
         *                  qualified tensor instance or a view with a const-
         *                  qualified value type. May be either an lvalue- or
         *                  rvalue-reference.
         */
#if MARRAY_DOXYGEN
        void reset(tensor_or_view other);
#else
        template <typename U, int N, bool O, typename D>
        void reset(marray_base<U, N, D, O>& other)
        {
            static_assert(NDim == DYNAMIC || N == DYNAMIC || NDim == N);
            inherit_bbox_(other);
            reset(other.lengths(), other.data(), other.bases(), other.strides());
        }

        template <typename U, int N, bool O, typename D>
        void reset(marray_base<U, N, D, O>&& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, bool O, typename D>
        void reset(const marray_base<U, N, D, O>& other)
        {
            static_assert(NDim == DYNAMIC || N == DYNAMIC || NDim == N);
            inherit_bbox_(other);
            reset(other.lengths(), other.data(), other.bases(), other.strides());
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        void reset(const marray_slice<U, N, I, D...>& other)
        {
            reset(other.view());
        }
#endif

        /**
         * Reset to a view that wraps a raw data pointer, using the provided shape, and the default base and layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              [len_type](@ref MArray::len_type), including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param layout_and_indexing
         *              Information specifying the memory layout and indexing for this view. May be one of
         *              the following:
         *                -# Nothing. The default layout and indexing are used.
         *                -# A pre-defined index base, layout, or combined base/layout specifier: one of
         *                   [BASE_ZERO](@ref MArray::BASE_ZERO), [BASE_ONE](@ref MArray::BASE_ONE),
         *                   [ROW_MAJOR](@ref MArray::ROW_MAJOR), [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR),
         *                   [C](@ref MArray::C), [CXX](@ref MArray::CXX), [FORTRAN](@ref MArray::FORTRAN),
         *                   or [MATLAB](@ref MArray::MATLAB).
         *                -# The strides of the tensor dimensions. May be any one-dimensional container type whose elements
         *                   are convertible to [stride_type](@ref MArray::stride_type), including initializer lists.
         *                -# The index base for each tensor dimension, followed by the strides of each dimension. May be any
         *                   combination of either pre-defined index bases (e.g. [BASE_ZERO](@ref MArray::BASE_ZERO)), pre-defined
         *                   layout (e.g. [ROW_MAJOR](@ref MArray::ROW_MAJOR)), combined base/layout specifier
         *                   (e.g. [FORTRAN](@ref MArray::FORTRAN)), or one-dimensional containers whose elements are
         *                   convertible to [len_type](@ref MArray::len_type) (index bases) or
         *                   [stride_type](@ref MArray::stride_type) (strides), including initializer lists.
         *
         *              Examples:
         *                    - `1: view.reset(len, ptr) #all defaults are used`
         *                    - `2: view.reset(len, ptr, BASE_ONE) #default layout`
         *                    - `2: view.reset(len, ptr, COLUMN_MAJOR) #default index base`
         *                    - `2: view.reset(len, ptr, FORTRAN)`
         *                    - `3: view.reset(len, ptr, {12, 1, 6}) #strides are specified explicitly, default base`
         *                    - `4: view.reset(len, ptr, BASE_ONE, ROW_MAJOR)`
         *                    - `4: view.reset(len, ptr, BASE_ZERO, {1, 120, 40, 10})`
         *                    - `4: view.reset(len, ptr, {0, 0, 10}, COLUMN_MAJOR)`
         *                    - `4: view.reset(len, ptr, {-1, 0, 2}, {144, 12, 1})`
         *                    - `4: view.reset(len, ptr, FORTRAN, std_vector_of_strides)`
         *                    - `4: view.reset(len, ptr, std_list_of_bases, CXX)`
         */
 #if MARRAY_DOXYGEN
        void reset(shape len, pointer ptr, base_and_or_layout layout_and_indexing);
 #else
        void reset(const array_1d<len_type>& len, pointer ptr)
        {
            reset(len, ptr, DEFAULT_BASE, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& len, pointer ptr, const index_base& base)
        {
            reset(len, ptr, base, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& len, pointer ptr, const layout_like& stride)
        {
            reset(len, ptr, DEFAULT_BASE, stride);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& len, pointer ptr, const base_like& base,
                   const layout_like& stride)
        {
            MARRAY_ASSERT(len.size() > 0);
            MARRAY_ASSERT(NDim == DYNAMIC || NDim == len.size());
            MARRAY_ASSERT(!base || len.size() == base.size());
            MARRAY_ASSERT(!stride || len.size() == stride.size());

            data_ = ptr;
            len.slurp(len_);
            base.base(len_, base_);
            stride.stride(len_, stride_);

            for (auto i : range(dimension()))
                MARRAY_ASSERT(len_[i] >= 0);

            set_bbox_(*this);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& len, pointer ptr, c_cxx_t)
        {
            reset(len, ptr, CXX, CXX);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& len, pointer ptr, fortran_t)
        {
            reset(len, ptr, FORTRAN, FORTRAN);
        }
#endif

        /**
         * Reset to a view that wraps a raw data pointer, using the provided extents and the specified layout.
         *
         * @note This overload provides functionality similar to the multidimensional array declarations in
         *       FORTRAN, e.g. `real, dimension(begin1:end1, begin2:end2, ...) :: array`, except that the upper
         *       bounds (`end`) are one greater than in the corresponding FORTRAN statement. This is by design such
         *       that the basic overloads with only `len` are equivalent to `begin = [0,...]` and `end = len`.
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
         * @param stride    The strides along each dimension, or one of [ROW_MAJOR](@ref MArray::ROW_MAJOR),
         *                  [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR), [C](@ref MArray::C), [CXX](@ref MArray::CXX),
         *                  [FORTRAN](@ref MArray::FORTRAN), [MATLAB](@ref MArray::MATLAB). If not specified,
         *                  the default layout is used.
         */
#if MARRAY_DOXYGEN
        void reset(indices begin, indices end, pointer ptr, layout_or_strides stride = DEFAULT_LAYOUT);
#else
        void reset(const array_1d<len_type>& begin, const array_1d<len_type>& end, pointer ptr)
        {
            reset(begin, end, ptr, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        void reset(const array_1d<len_type>& begin, const array_1d<len_type>& end, pointer ptr, const layout_like& stride)
        {
            MARRAY_ASSERT(begin.size() == end.size());

            detail::array_type_t<len_type,NDim> base;
            detail::array_type_t<len_type,NDim> len;
            begin.slurp(base);
            end.slurp(len);
            for (auto i : range(len.size()))
                len[i] -= base[i];

            reset(len, ptr, base, stride);
        }
#endif

        /** @} */
        /* *********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename Ptr, typename Func, int... I>
        void for_each_element_(Func& f, std::integer_sequence<int, I...>) const
        {
            auto it = make_iterator(lengths(), strides());
            Ptr ptr = const_cast<Ptr>(data());
            while (it.next(ptr))
                detail::call(f, *ptr, it.position()[I]+base(I)...);
        }

        template <typename Ptr, typename Func>
        void for_each_element_(Func& f) const
        {
            auto it = make_iterator(lengths(), strides());
            Ptr ptr = const_cast<Ptr>(data());
            detail::array_type_t<len_type,NDim> idx(dimension());
            while (it.next(ptr))
            {
                for (auto i : range(dimension()))
                    idx[i] = base(i)+it.position()[i];
                detail::call(f, *ptr, std::as_const(idx));
            }
        }

        void set_lengths_(int i, detail::array_type_t<len_type, NDim>& len,
                          std::initializer_list<Type> data)
        {
            MARRAY_ASSERT(i < len.size());
            len[i] = data.size();
        }

        template <typename U>
        void set_lengths_(int i, detail::array_type_t<len_type, NDim>& len,
                          std::initializer_list<U> data)
        {
            MARRAY_ASSERT(i < len.size());
            len[i] = data.size();
            set_lengths_(i+1, len, *data.begin());
        }

        void set_data_(int i, pointer ptr, std::initializer_list<Type> data)
        {
            auto it = data.begin();
            for (len_type j = 0;j < length(i);j++)
            {
                ptr[j * stride(i)] = *it;
                ++it;
            }
        }

        template <typename U>
        void set_data_(int i, pointer ptr, std::initializer_list<U> data)
        {
            auto it = data.begin();
            for (len_type j = 0;j < length(i);j++)
            {
                set_data_(i+1, ptr + j * stride(i), *it);
                ++it;
            }
        }

        template <size_t... I, typename... Args>
        reference get_reference_(std::index_sequence<I...>, const Args&... args) const
        {
            return *(const_cast<marray_base&>(*this).data() + ... + [&](int dim, len_type idx)
            {
                idx -= base(dim);
                MARRAY_ASSERT(idx >= 0 && idx < length(dim));
                return idx * stride(dim);
            }(I, args));
        }

        template <typename U, int N, typename D, bool O>
        void copy_(const marray_base<U, N, D, O>& other) const
        {
            static_assert(NDim == DYNAMIC || N == DYNAMIC || NDim == N);
            MARRAY_ASSERT(dimension() == other.dimension());
            MARRAY_ASSERT(std::equal(lengths().begin(), lengths().end(), other.lengths().begin()));

            if (!dimension()) return;

            auto a = const_cast<pointer>(data());
            auto b = other.data();
            auto [contiguous, size] = is_contiguous(lengths(), strides());

            if (contiguous && std::equal(strides().begin(), strides().end(), other.strides().begin()))
            {
                std::copy_n(b, size, a);
            }
            else
            {
                auto it = make_iterator(lengths(), strides(), other.strides());
                while (it.next(a, b))
                    *a = *b;
            }
        }

        void copy_(const Type& value) const
        {
            if (!dimension()) return;

            pointer a = const_cast<pointer>(data());
            auto [contiguous, size] = is_contiguous(lengths(), strides());

            if (contiguous)
            {
                std::fill_n(a, size, value);
            }
            else
            {
                auto it = make_iterator(lengths(), strides());
                while (it.next(a))
                    *a = value;
            }
        }

        template <typename T, typename Arg, typename... Args>
        auto index_(const Arg& arg, const Args&... args)
        {
            constexpr auto N = detail::count_dimensions<Arg, Args...>::value;
            static_assert(NDim == DYNAMIC || N == NDim);
            MARRAY_ASSERT(N == dimension());

            if constexpr (std::is_same_v<std::decay_t<Arg>,bcast_t>)
            {
                return marray_slice<T, N, 0, bcast_dim>{*this, slice::bcast}(args...);
            }
            else if constexpr (std::is_same_v<std::decay_t<Arg>,all_t>)
            {
                return marray_slice<T, N, 1, slice_dim>{*this, range(length(0))}(args...);
            }
            else if constexpr (std::is_convertible_v<Arg,len_type>)
            {
                return marray_slice<T, N, 1>{*this, arg}(args...);
            }
            else
            {
                return marray_slice<T, N, 1, slice_dim>{*this, arg}(args...);
            }
        }

        void swap(marray_base& other)
        {
            using std::swap;
#ifdef MARRAY_ENABLE_ASSERTS
            swap(bbox_data_, other.bbox_data_);
            swap(bbox_len_, other.bbox_len_);
            swap(bbox_off_, other.bbox_off_);
            swap(bbox_stride_, other.bbox_stride_);
#endif
            swap(data_, other.data_);
            swap(base_, other.base_);
            swap(len_, other.len_);
            swap(stride_, other.stride_);
        }

    public:
        /***********************************************************************
         *
         * @name Static helper functions
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return the strides for a hypothetical tensor with the given lengths and layout.
         *
         * @param len       The lengths of the hypothetical tensor.
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *                  If omitted, the default layout is used.
         *
         * @returns         The set of strides for the given lengths and layout.
         */
        static stride_vector strides(const array_1d<len_type>& len, layout layout = DEFAULT_LAYOUT)
        {
            return detail::strides(len, layout);
        }

        /**
         * Return the number of elements in a hypothetical tensor with the given lengths.
         *
         * @param len       The lengths of the hypothetical tensor.
         *
         * @return          The number of elements, which is equal to the product of the lengths.
         */
        static stride_type size(const array_1d<len_type>& len)
        {
            //TODO: add alignment option

            len_vector len_;
            len.slurp(len_);

            stride_type s = 1;
            for (auto i : range(len_.size()))
            {
                MARRAY_ASSERT(len_[i] >= 0);
                s *= len_[i];
            }
            return s;
        }

        /**
         * Return whether or not a tensor with the given lengths and strides would have contiguous storage.
         *
         * @param len       The lengths of the hypothetical tensor.
         *
         * @param stride    The strides of the hypothetical tensor.
         *
         * @return          A `std::pair`, whose first member is a boolean value indicating
         *                  contiguous storage, and whose second member is the size of the contiguous
         *                  storage, or 0.
         */
        static std::pair<bool,stride_type> is_contiguous(const array_1d<len_type>& len,
                                                         const array_1d<stride_type>& stride)
        {
            len_vector len_;
            len.slurp(len_);

            stride_vector stride_;
            stride.slurp(stride_);

            auto ndim = len_.size();
            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(ndim == stride_.size());

            stride_type size = 1;
            auto rng = range(ndim);
            if (stride_.front() > stride_.back())
                rng.reverse();

            for (auto i : rng)
            {
                MARRAY_ASSERT(len_[i] >= 0);
                if (stride_[i] != size)
                    return std::make_pair(false, stride_type());

                size *= len_[i];
            }

            return std::make_pair(true, size);
        }

        /** @} */
        /***********************************************************************
         *
         * @name Operators
         *
         **********************************************************************/
        /** @{ */

        /**
         * Set the tensor data using a nested initializer list.
         *
         * @param data  A nested initializer list. The number of levels must be
         *              equal to the number of dimensions, and the supplied initializer
         *              lists must be "dense", i.e. every element must be specified.
         *
         * @note Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC).
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        Derived& operator=(initializer_type data)
#else
        tensor_or_view& operator=(nested_initializer_list data)
#endif
        {
            detail::array_type_t<len_type, NDim> len(dimension());
            set_lengths_(0, len, data);
            MARRAY_ASSERT(len == lengths());
            set_data_(0, this->data(), data);
            return static_cast<Derived&>(*this);
        }

        /**
         * Set the tensor elements to the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     An expression; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator=(expression other);
#else
        template <typename Expression>
        std::enable_if_t<NDim != DYNAMIC && is_expression_arg<Expression>::value,Derived&>
        operator=(const Expression& other)
        {
            assign_expr(*this, other);
            return static_cast<Derived&>(*this);
        }

        /* Inherit docs */
        template <typename Expression>
        std::enable_if_t<NDim != DYNAMIC && is_expression_arg<Expression>::value,const Derived&>
        operator=(const Expression& other) const
        {
            assign_expr(*this, other);
            return static_cast<const Derived&>(*this);
        }
#endif

        /**
         * Set all of the tensor elements to the given scalar.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     A scalar value.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator=(const Type& other);
#else
        Derived& operator=(const Type& other)
        {
            copy_(other);
            return static_cast<Derived&>(*this);
        }

        /* Inherit docs */
        const Derived& operator=(const Type& other) const
        {
            copy_(other);
            return static_cast<const Derived&>(*this);
        }
#endif

        /**
         * Set the tensor elements to those of the given tensor or tensor view.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     A tensor, tensor view, or partially indexed tensor. The dimensions of
         *                  the tensor must match those of this tensor.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator=(tensor_or_view other);
#else
        template <typename U, int N, typename D, bool O>
        std::enable_if_t<NDim == DYNAMIC || N == DYNAMIC || NDim == N, Derived&>
        operator=(const marray_base<U, N, D, O>& other)
        {
            copy_(other);
            return static_cast<Derived&>(*this);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        std::enable_if_t<NDim == DYNAMIC || N == DYNAMIC || NDim == N, const Derived&>
        operator=(const marray_base<U, N, D, O>& other) const
        {
            copy_(other);
            return static_cast<const Derived&>(*this);
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        std::enable_if_t<NDim == DYNAMIC || NDim == marray_slice<U, N, I, D...>::NewNDim, Derived&>
        operator=(const marray_slice<U, N, I, D...>& other)
        {
            return *this = other.view();
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        std::enable_if_t<NDim == DYNAMIC || NDim == marray_slice<U, N, I, D...>::NewNDim, const Derived&>
        operator=(const marray_slice<U, N, I, D...>& other) const
        {
            return *this = other.view();
        }

        /* Inherit docs */
        Derived& operator=(const marray_base& other)
        {
            return operator=<>(other);
        }
#endif

        /**
         * Increment the elements by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     An expression; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator+=(expression other);
#else
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,Derived&>
        operator+=(const Expression& other)
        {
            return *this = *this + other;
        }

        /* Inherit docs */
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,const Derived&>
        operator+=(const Expression& other) const
        {
            return *this = *this + other;
        }
#endif

        /**
         * Decrement the elements by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     An expression; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator-=(expression other);
#else
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,Derived&>
        operator-=(const Expression& other)
        {
            return *this = *this - other;
        }

        /* Inherit docs */
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,const Derived&>
        operator-=(const Expression& other) const
        {
            return *this = *this - other;
        }
#endif

        /**
         * Perform an element-wise multiplication by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     An expression; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator*=(expression other);
#else
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,Derived&>
        operator*=(const Expression& other)
        {
            return *this = *this * other;
        }

        /* Inherit docs */
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,const Derived&>
        operator*=(const Expression& other) const
        {
            return *this = *this * other;
        }
#endif

        /**
         * Perform an element-wise division by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view ([marray_view](@ref MArray::marray_view)),
         * the value type must not be const-qualified.
         *
         * @param other     An expression; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return          *this
         */
#if MARRAY_DOXYGEN
        tensor_or_view& operator/=(expression other);
#else
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,Derived&>
        operator/=(const Expression& other)
        {
            return *this = *this / other;
        }

        /* Inherit docs */
        template <typename Expression>
        std::enable_if_t<is_expression_arg_or_scalar<Expression>::value,const Derived&>
        operator/=(const Expression& other) const
        {
            return *this = *this / other;
        }
#endif

        /**
         * Return true if this tensor is the same size and shape and has the same elements
         * as another tensor.
         *
         * @param other     A tensor or tensor view against which to check.
         *
         * @return          True if all elements match, false otherwise. If the tensors
         *                  are not the same size and shape, then false.
         */
#if MARRAY_DOXYGEN
        bool operator==(tensor_or_view other) const;
#else
        template <typename U, int N, typename D, bool O>
        bool operator==(const marray_base<U, N, D, O>& other) const
        {
            if (lengths() != other.lengths() || !dimension())
                return false;

            auto it = make_iterator(lengths(), strides(), other.strides());
            auto a = data();
            auto b = other.data();
            while (it.next(a, b))
            {
                if (*a == *b) continue;
                return false;
            }

            return true;
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        bool operator==(const marray_slice<U, N, I, D...>& other) const
        {
            return *this == other.view();
        }
#endif

        /**
         * Return false if this tensor is the same size and shape and has the same elements
         * as another tensor.
         *
         * @param other     A tensor or tensor view against which to check.
         *
         * @return          False if all elements match, true otherwise. If the tensors
         *                  are not the same size and shape, then true.
         */
#if MARRAY_DOXYGEN
        bool operator!=(tensor_or_view other) const;
#else
        template <typename U, int N, typename D, bool O>
        bool operator!=(const marray_base<U, N, D, O>& other) const
        {
            return !(*this == other);
        }

        /* Inherit docs */
        template <typename U, int N, int I, typename... D>
        bool operator!=(const marray_slice<U, N, I, D...>& other) const
        {
            return *this != other.view();
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Views
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return an immutable view of this tensor.
         *
         * @tparam N The number of dimensions in the view, or [DYNAMIC](@ref MArray::DYNAMIC). The default is `NDim`.
         *
         * @return An immutable view.
         */
        template <int N=NDim>
#if MARRAY_DOXYGEN
        immutable_view
#else
        marray_view<const Type, N>
#endif
        cview() const
        {
            return *this;
        }

        /**
         * Return a view of this tensor.
         *
         * @tparam N The number of dimensions in the view, or [DYNAMIC](@ref MArray::DYNAMIC). The default is `NDim`.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
        template <int N=NDim>
#if MARRAY_DOXYGEN
        possibly_mutable_view view();
#else
        marray_view<ctype, N> view() const
        {
            return *this;
        }

        /* Inherit docs */
        template <int N=NDim>
        marray_view<Type, N> view()
        {
            return *this;
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Iterators
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a possibly-const-qualified iterator to the first leading face or element.
         *
         * The returned iterator selects leading faces of the tensor, i.e. a subtensor formed by specifying the
         * value of the first index. For a one-dimensional tensor (vector), the iterator selects vector elements.
         * The iterator supports random access. The iterator is const-qualified if the object is a tensor
         * ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified iterator over the leading faces of the tensor.
         */
        const_iterator cbegin(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return const_iterator{*this, dim, base(dim)};
        }

        /**
         * Return a possibly-const-qualified iterator to the first leading face or element.
         *
         * The returned iterator selects leading faces of the tensor, i.e. a subtensor formed by specifying the
         * value of the first index. For a one-dimensional tensor (vector), the iterator selects vector elements.
         * The iterator supports random access. The iterator is const-qualified if the object is a const-qualitied
         * tensor ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified iterator over the leading faces of the tensor.
         */
#if MARRAY_DOXYGEN
        possibly_const_iterator begin(int dim=0);
#else
        const_iterator begin(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return const_iterator{*this, dim, base(dim)};
        }

        iterator begin(int dim=0)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return iterator{*this, dim, base(dim)};
        }
#endif

        /**
         * Return a possibly-const-qualified iterator to one past the last leading face or element.
         *
         * The one-past-the-end iterator should not be dereferenced unless it is decremented to a valid position.
         * The returned iterator selects leading faces of the tensor, i.e. a subtensor formed by specifying the
         * value of the first index. For a one-dimensional tensor (vector), the iterator selects vector elements.
         * The iterator supports random access. The iterator is const-qualified if the object is a tensor
         * ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified iterator over the leading faces of the tensor.
         */
        const_iterator cend(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return const_iterator{*this, dim, base(dim) + length(dim)};
        }

        /**
         * Return a possibly-const-qualified iterator to one past the last leading face or element.
         *
         * The one-past-the-end iterator should not be dereferenced unless it is decremented to a valid position.
         * The returned iterator selects leading faces of the tensor, i.e. a subtensor formed by specifying the
         * value of the first index. For a one-dimensional tensor (vector), the iterator selects vector elements.
         * The iterator supports random access. The iterator is const-qualified if the object is a const-qualified
         * tensor ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified iterator over the leading faces of the tensor.
         */
#if MARRAY_DOXYGEN
        possibly_const_iterator end(int dim=0);
#else
        const_iterator end(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return const_iterator{*this, dim, base(dim) + length(dim)};
        }

        iterator end(int dim=0)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return iterator{*this, dim, base(dim) + length(dim)};
        }
#endif

        /**
         * Return a possibly-const-qualified reverse iterator to the last leading face or element.
         *
         * The returned iterator selects leading faces of the tensor in reverse order, i.e. a subtensor formed by
         * specifying the value of the first index. For a one-dimensional tensor (vector), the iterator selects
         * vector elements. The iterator supports random access. The iterator is const-qualified if the object is a
         * tensor ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified reverse iterator over the leading faces of the tensor.
         */
        const_reverse_iterator crbegin(int dim=0) const
        {
            return const_reverse_iterator{end(dim)};
        }

        /**
         * Return a possibly-const-qualified reverse iterator to the last leading face or element.
         *
         * The returned iterator selects leading faces of the tensor in reverse order, i.e. a subtensor formed by
         * specifying the value of the first index. For a one-dimensional tensor (vector), the iterator selects
         * vector elements. The iterator supports random access. The iterator is const-qualified if the object is a
         * const-qualified tensor ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view))
         * with a const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified reverse iterator over the leading faces of the tensor.
         */
#if MARRAY_DOXYGEN
        possibly_const_reverse_iterator rbegin(int dim=0);
#else
        const_reverse_iterator rbegin(int dim=0) const
        {
            return const_reverse_iterator{end(dim)};
        }

        reverse_iterator rbegin(int dim=0)
        {
            return reverse_iterator{end(dim)};
        }
#endif

        /**
         * Return a possibly-const-qualified reverse iterator to one before the first leading face or element.
         *
         * The one-past-the-end iterator should not be dereferenced unless it is decremented to a valid position.
         * The returned iterator selects leading faces of the tensor in reverse order, i.e. a subtensor formed by
         * specifying the value of the first index. For a one-dimensional tensor (vector), the iterator selects vector
         * elements. The iterator supports random access. The iterator is const-qualified if the object is a tensor
         * ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view)) with a
         * const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified reverse iterator over the leading faces of the tensor.
         */
        const_reverse_iterator crend(int dim=0) const
        {
            return const_reverse_iterator{begin(dim)};
        }

        /**
         * Return a possibly-const-qualified reverse iterator to one before the first leading face or element.
         *
         * The one-past-the-end iterator should not be dereferenced unless it is decremented to a valid position.
         * The returned iterator selects leading faces of the tensor in reverse order, i.e. a subtensor formed by
         * specifying the value of the first index. For a one-dimensional tensor (vector), the iterator selects vector
         * elements. The iterator supports random access. The iterator is const-qualified if the object is a
         * const-qualified tensor ([marray](@ref MArray::marray)), or a view ([marray_view](@ref MArray::marray_view))
         * with a const-qualified value type.
         *
         * @param dim The dimension along which to iterator. Default is the leading dimension.
         *
         * @returns A possibly-const-qualified reverse iterator over the leading faces of the tensor.
         */
#if MARRAY_DOXYGEN
        possibly_const_reverse_iterator rend(int dim=0);
#else
        const_reverse_iterator rend(int dim=0) const
        {
            return const_reverse_iterator{begin(dim)};
        }

        reverse_iterator rend(int dim=0)
        {
            return reverse_iterator{begin(dim)};
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Shift operations
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount along each dimension.
         *
         * The `k`th index `i` in the shifted view is equivalent to an index `i+n[k]`
         * in the original tensor or tensor view.
         *
         * @param n The amount by which to shift for each dimension. May be any
         *          one-dimensional container type whose elements are convertible
         *          to a tensor length, including initializer lists.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view shifted(const array_1d<len_type>& n);
#else
        marray_view<Type, NDim> shifted(const array_1d<len_type>& n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(n);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype, NDim> shifted(const array_1d<len_type>& n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }
#endif

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @param n     The amount by which to shift.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view shifted(int dim, len_type n);
#else
        marray_view<Type, NDim> shifted(int dim, len_type n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(dim, n);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype, NDim> shifted(int dim, len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted(dim, n);
        }
#endif

        /**
         * Return a view that references elements whose indices are
         * shifted "down" along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view shifted_down(int dim);
#else
        marray_view<Type,NDim> shifted_down(int dim)
        {
            return shifted(dim, length(dim));
        }

        /* Inherit docs */
        marray_view<ctype,NDim> shifted_down(int dim) const
        {
            return const_cast<marray_base&>(*this).shifted_down(dim);
        }
#endif

        /**
         * Return a view that references elements whose indices are
         * shifted "up" along one dimension.
         *
         * For the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view shifted_up(int dim);
#else
        marray_view<Type,NDim> shifted_up(int dim)
        {
            return shifted(dim, -length(dim));
        }

        /* Inherit docs */
        marray_view<ctype,NDim> shifted_up(int dim) const
        {
            return const_cast<marray_base&>(*this).shifted_up(dim);
        }
#endif

        /**
         * Return a view that references the same data, but with a different base.
         *
         * @param new_base   The base of the new view. Either a container type with elements convertible to
         *                   [len_type](@ref MArray::len_type) (including initializer lists), or one of the tokens [BASE_ZERO](@ref MArray::BASE_ZERO),
         *                   [BASE_ONE](@ref MArray::BASE_ONE), [FORTRAN](@ref MArray::FORTRAN), or [MATLAB](@ref MArray::MATLAB).
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view rebased(const base_like& new_base);
#else
        marray_view<Type,NDim> rebased(const base_like& new_base)
        {
            marray_view<Type,NDim> r(*this);
            r.rebase(new_base);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype,NDim> rebased(const base_like& new_base) const
        {
            return const_cast<marray_base&>(*this).rebased(new_base);
        }
#endif

        /**
         * Return a view that references the same data, but with a different base along one dimension.
         *
         * @param dim   The dimension along which to rebase the returned view.
         *
         * @param new_base   The base of the new view along the indicated dimension.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view rebased(int dim, len_type new_base);
#else
        marray_view<Type,NDim> rebased(int dim, len_type new_base)
        {
            marray_view<Type,NDim> r(*this);
            r.rebase(dim, new_base);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype,NDim> rebased(int dim, len_type new_base) const
        {
            return const_cast<marray_base&>(*this).rebased(dim, new_base);
        }
#endif

        /**
         * Change the base for the indices.
         *
         * @param new_base   The new base, either a container with elements convertible to [len_type](@ref MArray::len_type) or one
         *                   of the tokens [BASE_ZERO](@ref MArray::BASE_ZERO), [BASE_ONE](@ref MArray::BASE_ONE), [FORTRAN](@ref MArray::FORTRAN), or [MATLAB](@ref MArray::MATLAB).
         */
        void rebase(const base_like& new_base)
        {
            MARRAY_ASSERT(!new_base || new_base.size() == dimension());

            auto base = new_base.base(lengths());
            for (auto i : range(dimension()))
                rebase(i, base[i]);
        }

        /**
         * Change the base along one dimension.
         *
         * @param dim   The dimension along which to rebase.
         *
         * @param new_base   The new base along the indicated dimension.
         *
         */
        void rebase(int dim, len_type new_base)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            base_[dim] = new_base;
        }

        /** @} */
        /***********************************************************************
         *
         * @name Permutation
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a permuted view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original tensor or tensor
         * view.
         *
         * @param perm  The permutation vector. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists. The values must form
         *              a permutation of `[0,N)`, where `N` is the number of
         *              tensor dimensions.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view permuted(const array_1d<int>& perm);
#else
        marray_view<Type,NDim> permuted(const array_1d<int>& perm)
        {
            marray_view<Type,NDim> r(*this);
            r.permute(perm);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype,NDim> permuted(const array_1d<int>& perm) const
        {
            return const_cast<marray_base&>(*this).permuted(perm);
        }
#endif

        /**
         * Return a permuted view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original tensor or tensor
         * view.
         *
         * @param perm  The permutation vector. May be any
         *              set of integral types convertible
         *              to `int`. The values must form
         *              a permutation of `[0,NDim)`, where `NDim` is the number of
         *              tensor dimensions.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view permuted(const Perm&... perm);
#else
        template <typename... Perm>
        std::enable_if_t<detail::are_convertible<int,Perm...>::value,marray_view<Type,NDim>>
        permuted(const Perm&... perm)
        {
            marray_view<Type,NDim> r(*this);
            r.permute(perm...);
            return r;
        }

        /* Inherit docs */
        template <typename... Perm>
        std::enable_if_t<detail::are_convertible<int,Perm...>::value,marray_view<ctype,NDim>>
        permuted(const Perm&... perm) const
        {
            return const_cast<marray_base&>(*this).permuted(perm...);
        }
#endif

        /**
         * Return a transposed view.
         *
         * This overload is only available for matrices and matrix views (`NDim == 2`).
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view transposed();
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N==2,marray_view<Type, NDim>>
        transposed()
        {
            return permuted({1, 0});
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==2,marray_view<ctype, NDim>>
        transposed() const
        {
            return const_cast<marray_base&>(*this).transposed();
        }
#endif

        /**
         * Return a transposed view.
         *
         * This overload is only available for matrices and matrix views (`NDim == 2`).
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view T();
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N==2,marray_view<Type, NDim>>
        T()
        {
            return transposed();
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==2,marray_view<ctype, NDim>>
        T() const
        {
            return const_cast<marray_base&>(*this).T();
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Dimension change
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a view of lower dimensionality.
         *
         * The values along each lowered dimension (which corresponds to one or
         * more dimensions in the original tensor or tensor view) must have a
         * consistent stride (i.e. those dimensions must be contiguous). The base of an index which is
         * formed by combining multiple indices in the original view is equal to the base of index
         * with smallest stride.
         *
         * @tparam NewNDim  The number of dimensions in the lowered view or [DYNAMIC](@ref MArray::DYNAMIC).
         *                  If `split` is a comma-separated list of split points (e.g. `t.split(1, 3, 4)`),
         *                  then `NewNDim` is deduced as the number of split points plus one.
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
         *              in increasing order and in the range `[1,N)`. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists, or a comma-separated list of
         *              split points (i.e. multiple arguments of any integral type, one
         *              for each split point).
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int NewNDim=DYNAMIC>
#if MARRAY_DOXYGEN
        possibly_mutable_view lowered(const array_1d<int>& split);
#else
        marray_view<Type, NewNDim> lowered(const array_1d<int>& split)
        {
            static_assert(NewNDim == DYNAMIC || NewNDim > 0,
                          "Cannot split into this number of dimensions");
            static_assert(NDim == DYNAMIC || NewNDim == DYNAMIC || NewNDim <= NDim,
                          "Cannot split into this number of dimensions");

            MARRAY_ASSERT(NewNDim == DYNAMIC || split.size() == NewNDim-1);
            MARRAY_ASSERT(split.size() < dimension());
            auto nsplit = split.size();

            dim_vector split_;
            split.slurp(split_);

            for (auto i : range(nsplit))
            {
                MARRAY_ASSERT(split_[i] > 0 && split_[i] < dimension());
                if (i != 0) MARRAY_ASSERT(split_[i-1] < split_[i]);
            }

            len_vector newbase(nsplit+1);
            len_vector newlen(nsplit+1);
            stride_vector newstride(nsplit+1);

            for (auto i : range(nsplit+1))
            {
                auto begin = (i == 0 ? 0 : split_[i-1]);
                auto end = (i == nsplit ? dimension()-1 : split_[i]-1);
                if (begin > end) continue;

                if (stride(begin) < stride(end) ||
                    (stride(begin) == stride(end) && length(begin) == 1))
                {
                    newbase[i] = base(begin);
                    newlen[i] = length(end);
                    newstride[i] = stride(begin);
                    for (auto j : range(begin,end))
                    {
                        MARRAY_ASSERT(stride(j+1) == stride(j) * length(j));
                        newlen[i] *= length(j);
                    }
                }
                else
                {
                    newbase[i] = base(end);
                    newlen[i] = length(end);
                    newstride[i] = stride(end);
                    for (auto j : range(begin,end))
                    {
                        MARRAY_ASSERT(stride(j) == stride(j+1) * length(j+1));
                        newlen[i] *= length(j);
                    }
                }
            }

            //TODO: bbox? right now it is "shrunk" to the new view

            return {newlen, data(), newbase, newstride};
        }

        /* Inherit docs */
        template <int NewNDim=DYNAMIC>
        auto lowered(const array_1d<int>& split) const
        {
            return const_cast<marray_base&>(*this).lowered<NewNDim>(split);
        }

        /* Inherit docs */
        template <typename... Splits>
        std::enable_if_t<detail::are_convertible<int,Splits...>::value,marray_view<Type,sizeof...(Splits)+1>>
        lowered(const Splits... splits)
        {
            return lowered<sizeof...(Splits)+1>({(int)splits...});
        }

        /* Inherit docs */
        template <typename... Splits>
        std::enable_if_t<detail::are_convertible<int,Splits...>::value,marray_view<ctype,sizeof...(Splits)+1>>
        lowered(const Splits... splits) const
        {
            return lowered<sizeof...(Splits)+1>({(int)splits...});
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Reversal
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a view where the order of the indices along each dimension has
         * been reversed.
         *
         * An index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original tensor or view, where `n` is the tensor length along
         * that dimension.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view reversed();
#else
        marray_view<Type, NDim> reversed()
        {
            marray_view<Type,NDim> r(*this);
            r.reverse();
            return r;
        }

        /* Inherit docs */
        marray_view<ctype, NDim> reversed() const
        {
            return const_cast<marray_base&>(*this).reversed();
        }
#endif

        /**
         * Return a view where the order of the indices along the given dimension has
         * been reversed.
         *
         * For the indicated dimension, an index of `i` in the reversed tensor corresponds to an index of
         * `n-1-i` in the original tensor, where `n` is the tensor length along
         * that dimension.
         *
         * @param dim   The dimension along which to reverse the indices.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view ([marray_view](@ref MArray::marray_view)),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view reversed(int dim);
#else
        marray_view<Type, NDim> reversed(int dim)
        {
            marray_view<Type,NDim> r(*this);
            r.reverse(dim);
            return r;
        }

        /* Inherit docs */
        marray_view<ctype, NDim> reversed(int dim) const
        {
            return const_cast<marray_base&>(*this).reversed(dim);
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Slices
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return an immutable reference or view to an element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @param dim   The dimension along which to extract the reference or face. Default is the leading dimension.
         *
         * @param i     The index of the element or face to extract.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference_or_view cslice(int dim, len_type i);
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<const Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        cslice(int dim, len_type i) const
        {
            return const_cast<marray_base&>(*this).slice(dim, i);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, const_reference>
        cslice(int dim, len_type i) const
        {
            return const_cast<marray_base&>(*this).slice(dim, i);
        }
#endif

        /**
         * Return a reference or view to an element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face. Default is the leading dimension.
         *
         * @param i     The index of the element or face to extract.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference_or_view slice(int dim, len_type i);
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        slice(int dim, len_type i)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());

            i -= base(dim);
            MARRAY_ASSERT(i >= 0 && i < length(dim));

            len_vector base;
            len_vector len;
            stride_vector stride;
            base.insert(base.end(), bases().begin(), bases().begin()+dim);
            base.insert(base.end(), bases().begin()+dim+1, bases().end());
            len.insert(len.end(), lengths().begin(), lengths().begin()+dim);
            len.insert(len.end(), lengths().begin()+dim+1, lengths().end());
            stride.insert(stride.end(), strides().begin(), strides().begin()+dim);
            stride.insert(stride.end(), strides().begin()+dim+1, strides().end());

            return {len, data() + i * this->stride(dim), base, stride};
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<ctype, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        slice(int dim, len_type i) const
        {
            return const_cast<marray_base&>(*this).slice(dim, i);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, reference>
        slice(int dim, len_type i)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            return (*this)[i];
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, cref>
        slice(int dim, len_type i) const
        {
            return const_cast<marray_base&>(*this).slice(dim, i);
        }
#endif

        /**
         * Return an immutable reference or view to the first element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @param dim   The dimension along which to extract the reference or face. Default is the leading dimension.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference_or_view cfront(int dim=0) const;
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<const Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        cfront(int dim=0) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, const_reference>
        cfront(int dim=0) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }
#endif

        /**
         * Return a reference or view to the first element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face. Default is the leading dimension.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference_or_view front(int dim=0);
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        front(int dim=0)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return slice(dim, base(dim));
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<ctype, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        front(int dim=0) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, reference>
        front(int dim=0)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            return (*this)[base()];
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, cref>
        front(int dim=0) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }
#endif

        /**
         * Return an immutable reference or view to the last element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @param dim   The dimension along which to extract the reference or face. Default is the leading dimension.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference_or_view cback(int dim=0) const;
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<const Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        cback(int dim=0) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, const_reference>
        cback(int dim=0) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }
#endif

        /**
         * Return a reference or view to the last element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face. Default is the leading dimension.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference_or_view back(int dim=0);
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<Type, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        back(int dim=0)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return slice(dim, base(dim) + length(dim) - 1);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1, marray_view<ctype, NDim == DYNAMIC ? DYNAMIC : NDim-1>>
        back(int dim=0) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, reference>
        back(int dim=0)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            return (*this)[base() + length() - 1];
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, cref>
        back(int dim=0) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Indexing
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return a reference or subtensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload specifies a particular index along a dimension, and so
         * reduces the dimensionality of the resulting view by one. If specific
         * indices are given for all dimensions then the result is a reference to
         * the specified tensor element. If this overload is mixed with others
         * that specify ranges of indices, then the result is a subtensor view.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final view or reference is mutable if the instance is not const-qualified.
         * For a tensor view ([marray_view](@ref MArray::marray_view)), the final
         * view or reference is mutable if the value type is not const-qualified.
         *
         * @note Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC). Otherwise, use [operator()](@ref operator()(const array_1d<len_type>&)).
         *
         * @param i     The specified index. The dimension to which this index
         *              refers depends on how many [] operators have been applied.
         *              The first [] refers to the first dimension and so on.
         *
         * @return      If all indices have been explicitly specified, a reference
         *              to the indicated tensor element. Otherwise, a temporary
         *              indexing object which can be converted to a tensor view.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_reference_or_view operator[](len_type i);
#else
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, reference>
        operator[](len_type i)
        {
            i -= base();
            MARRAY_ASSERT(i >= 0 && i < length());
            return data()[i * stride()];
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N==1, cref>
        operator[](len_type i) const
        {
            return const_cast<marray_base&>(*this)[i];
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1&&N!=DYNAMIC, marray_slice<Type, NDim, 1>>
        operator[](len_type i)
        {
            i -= base(0);
            MARRAY_ASSERT(i >= 0 && i < length(0));
            return {*this, i};
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=1&&N!=DYNAMIC, marray_slice<ctype, NDim, 1>>
        operator[](len_type i) const
        {
            i -= base(0);
            MARRAY_ASSERT(i >= 0 && i < length(0));
            return {*this, i};
        }
#endif

        /**
         * Return a subtensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload specifies a range along a dimension, and does not reduce the
         * dimensionality of the view.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final view is mutable if the instance is not const-qualified.
         * For a tensor view ([marray_view](@ref MArray::marray_view)), the final
         * view is mutable if the value type is not const-qualified.
         *
         * @note Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC). Otherwise, use @ref operator()().
         *
         * @param x     The specified range (either the result of [range](@ref MArray::range) or
         *              [all](@ref MArray::slice::all)). The dimension to which this range
         *              refers depends on how many [] operators have been applied.
         *              The first [] refers to the first dimension and so on.
         *
         * @return      A temporary indexing object which can be converted to a tensor view.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view operator[](range x);
#else
        template <typename I, typename=void, int N=NDim>
        std::enable_if_t<N!=DYNAMIC,marray_slice<Type, NDim, 1, slice_dim>>
        operator[](range_t<I> x)
        {
            x -= base(0);
            MARRAY_ASSERT_RANGE_IN(x, 0, length(0));
            return {*this, x};
        }

        /* Inherit docs */
        template <typename I, typename=void, int N=NDim>
        std::enable_if_t<N!=DYNAMIC,marray_slice<ctype, NDim, 1, slice_dim>>
        operator[](range_t<I> x) const
        {
            x -= base(0);
            MARRAY_ASSERT_RANGE_IN(x, 0, length(0));
            return {*this, x};
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=DYNAMIC,marray_slice<Type, NDim, 1, slice_dim>>
        operator[](all_t)
        {
            return {*this, range(length(0))};
        }

        /* Inherit docs */
        template <typename=void, int N=NDim>
        std::enable_if_t<N!=DYNAMIC,marray_slice<ctype, NDim, 1, slice_dim>>
        operator[](all_t) const
        {
            return {*this, range(length(0))};
        }
#endif

        /**
         * Return a broadcasted tensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload specifies that the tensor data should be repeated some number of times.
         * The indexing object returned by this overload cannot be converted to a view or used on the left-hand
         * side of an assignment, but it can be used in the right-hand side of any tensor expression. Explicitly
         * broadcasting allows the dimensions of different tensors in the expression to be lined up as desired.
         *
         * @note Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC). Otherwise, use @ref operator()().
         *
         * @param bcast  The special token [bcast](@ref MArray::slice::bcast).
         *
         * @return      A temporary indexing object.
         */
#if MARRAY_DOXYGEN
        indexing_object operator[](bcast_t bcast) const;
#else
        marray_slice<const Type, NDim, 0, bcast_dim>
        operator[](bcast_t) const
        {
            return {*this, slice::bcast};
        }
#endif

        /**
         * Return a reference to the indicated element.
         *
         * @param idx   The indices of the desired element. The number of indices must be
         *              equal to the number of dimensions. May be any
         *              one-dimensional container type whose elements are convertible
         *              to a tensor length, including initializer lists.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view ([marray_view](@ref MArray::marray_view)),
         *          the returned reference is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_reference operator()(const array_1d<len_type>& idx);
#else
        reference operator()(const array_1d<len_type>& idx)
        {
            MARRAY_ASSERT(idx.size() == dimension());

            std::array<len_type,NDim> idx_;
            idx.slurp(idx_);

            auto ptr = data();
            for (auto i : range(dimension()))
            {
                idx_[i] -= base(i);
                MARRAY_ASSERT(idx_[i] >= 0 && idx_[i] < length(i));
                ptr += idx_[i] * stride(i);
            }

            return *ptr;
        }

        /* Inherit docs */
        cref operator()(const array_1d<len_type>& idx) const
        {
            return const_cast<marray_base&>(*this)(idx);
        }
#endif

        /**
         * Return a reference or subtensor.
         *
         * The result of `tensor(arg1, arg2, arg2, ...)` is exactly the same as
         * `tensor[arg1][arg2][arg2]...`, except that it is also applicable to tensors
         * and views with `NDim == ` [DYNAMIC](@ref MArray::DYNAMIC). In that case, the result is the same as
         * `tensor.view<N>()[arg1][arg2][arg3]...` where `N` is the number of non-broadcast
         * arguments, and must be the same as the number of tensor dimensions.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final view or reference is mutable if the instance is not const-qualified.
         * For a tensor view ([marray_view](@ref MArray::marray_view)), the final
         * view or reference is mutable if the value type is not const-qualified.
         *
         * @param idx  Parameter pack of indices, ranges, [all](@ref MArray::slice::all), or
         *             [bcast](@ref MArray::slice::bcast).
         *
         * @return      If all indices have been explicitly specified, a reference
         *              to the indicated tensor element. Otherwise, a temporary
         *              indexing object which can be converted to a tensor view (unless
         *              one or more broadcast dimensions have been introduced).
         */
#if MARRAY_DOXYGEN
        template <typename... Indices>
        possibly_mutable_reference_or_view operator()(Indices... idx);
#else
        template <typename Arg, typename... Args, typename=
            std::enable_if_t<detail::are_indices_or_slices<Arg, Args...>::value &&
                            !detail::are_convertible<len_type, Arg, Args...>::value>>
        auto operator()(const Arg& arg, const Args&... args)
        {
            return index_<Type>(arg, args...);
        }

        /* Inherit docs */
        template <typename Arg, typename... Args, typename=
            std::enable_if_t<detail::are_indices_or_slices<Arg, Args...>::value &&
                            !detail::are_convertible<len_type, Arg, Args...>::value>>
        auto operator()(const Arg& arg, const Args&... args) const
        {
            return const_cast<marray_base&>(*this).index_<ctype>(arg, args...);
        }

        /* Inherit docs */
        template <typename... Args>
        std::enable_if_t<detail::are_convertible<len_type, Args...>::value,reference>
        operator()(const Args&... args)
        {
            static_assert(NDim == DYNAMIC || NDim == sizeof...(Args));
            MARRAY_ASSERT(sizeof...(Args) == dimension());
            return get_reference_(std::index_sequence_for<Args...>{}, args...);
        }

        /* Inherit docs */
        template <typename... Args>
        std::enable_if_t<detail::are_convertible<len_type, Args...>::value,cref>
        operator()(const Args&... args) const
        {
            return const_cast<marray_base&>(*this)(args...);
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Element-wise iteration
         *
         **********************************************************************/
        /** @{ */

        /**
         * Iterate over the elements and call a function.
         *
         * @tparam N  The number of dimensions which will be iterated over, or [DYNAMIC](@ref MArray::DYNAMIC).
         *            If `NDim` and `N` are both not [DYNAMIC](@ref MArray::DYNAMIC), then they must be equal.
         *
         * @param f
         * @parblock  A function or functor callable as either `f(e)`, where `e` is a reference to a tensor element,
         *            or
         *
         *            - if `N != ` [DYNAMIC](@ref MArray::DYNAMIC) : `f(e, i0, i1, ...)` where `i0, i1, ...` are indices, one for
         *              each dimension, or
         *            - if `N == ` [DYNAMIC](@ref MArray::DYNAMIC) : `f(e, idx)` where `idx` is an unspecified container
         *              type of indices, one for each dimension.
         *
         *            For a tensor ([marray](@ref MArray::marray)),
         *            the elements are mutable if the instance is not const-qualified.
         *            For a tensor view ([marray_view](@ref MArray::marray_view)),
         *            the elements are mutable if the value type is not
         *            const-qualified.
         * @endparblock
         */
        template <int N=NDim, typename Func>
#if MARRAY_DOXYGEN
        void for_each_element(Func&& f);
#else
        void for_each_element(Func&& f)
        {
            static_assert(N == DYNAMIC || NDim == DYNAMIC || N == NDim);

            if constexpr (N == DYNAMIC)
            {
                for_each_element_<pointer>(f);
            }
            else
            {
                MARRAY_ASSERT(N == dimension());
                for_each_element_<pointer>(f, std::make_integer_sequence<int, N>{});
            }
        }

        template <int N=NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            static_assert(N == DYNAMIC || NDim == DYNAMIC || N == NDim);

            if constexpr (N == DYNAMIC)
            {
                for_each_element_<cptr>(f);
            }
            else
            {
                MARRAY_ASSERT(N == dimension());
                for_each_element_<cptr>(f, std::make_integer_sequence<int, N>{});
            }
        }
#endif

        /** @} */
        /***********************************************************************
         *
         * @name Basic getters
         *
         **********************************************************************/
        /** @{ */

        /**
         * Return an immutable pointer to the tensor origin.
         *
         * @return An immutable pointer that points to the element with
         *         all zero indices. In general this pointer should not be dereferenced unless is has
         *         been offset by the product of the tensor strides with a valid set of indices.
         */
#if MARRAY_DOXYGEN
        immutable_pointer
#else
        const_pointer
#endif
        corigin() const
        {
            return const_cast<marray_base&>(*this).origin();
        }

        /**
         * Return an immutable pointer to the tensor origin.
         *
         * @return A pointer that points to the element with
         *         all zero indices. In general this pointer should not be dereferenced unless is has
         *         been offset by the product of the tensor strides with a valid set of indices.
         *         For a tensor ([marray](@ref MArray::marray)),
         *            the returned pointer is immutable if the instance is const-qualified.
         *            For a tensor view ([marray_view](@ref MArray::marray_view)),
         *            the returned pointer is immutable if the value type is
         *            const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_immutable_pointer origin();
#else
        pointer origin()
        {
            return const_cast<marray_base&>(*this).origin();
        }

        /* Inherit docs */
        cptr origin() const
        {
            auto ptr = data();
            for (auto i : range(dimension()))
                ptr -= base(i) * stride(i);
            return ptr;
        }
#endif

        /**
         * Return an immutable pointer to the tensor data.
         *
         * @return An immutable pointer that points to the element with `index == base`.
         */
#if MARRAY_DOXYGEN
        immutable_pointer
#else
        const_pointer
#endif
        cdata() const
        {
            return const_cast<marray_base&>(*this).data();
        }

        /**
         * Return a pointer to the tensor data.
         *
         * @return A pointer that points to the element with `index == base`.
         *         For a tensor ([marray](@ref MArray::marray)),
         *            the returned pointer is immutable if the instance is const-qualified.
         *            For a tensor view ([marray_view](@ref MArray::marray_view)),
         *            the returned pointer is immutable if the value type is
         *            const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_pointer data();
#else
        pointer data()
        {
            return data_;
        }

        /* Inherit docs */
        cptr data() const
        {
            return const_cast<marray_base&>(*this).data();
        }
#endif

        /**
         * Return the tensor base along the specified dimension.
         *
         * @param dim   A dimension.
         *
         * @return      The base of the specified dimension.
         */
        len_type base(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return base_[dim];
        }

        /**
         * Return the tensor base.
         *
         * @return The base of the tensor; immutable.
         */
        auto& bases() const
        {
            return base_;
        }

        /**
         * Return the tensor length along the specified dimension.
         *
         * @param dim   A dimension.
         *
         * @return      The length of the specified dimension.
         */
        len_type length(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return len_[dim];
        }

        /**
         * Return the tensor lengths.
         *
         * @return The lengths of the tensor; immutable.
         */
        auto& lengths() const
        {
            return len_;
        }

        /**
         * Return the tensor stride along the specified dimension.
         *
         * @param dim   A dimension.
         *
         * @return      The stride of the specified dimension.
         */
        stride_type stride(int dim=0) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return stride_[dim];
        }

        /**
         * Return the tensor strides.
         *
         * @return The strides of the tensor; immutable.
         */
        auto& strides() const
        {
            return stride_;
        }

        /**
         * Return the number of dimensions.
         *
         * @note The return value is only truly a constant expresson if `N` is not [DYNAMIC](@ref MArray::DYNAMIC).
         *
         * @return The number of dimensions.
         */
        constexpr int dimension() const
        {
            return lengths().size();
        }

        /** @} */
};

/**
 * Return an immutable view of the given tensor.
 *
 * @param x The tensor to view.
 *
 * @tparam N The number of dimensions in the resulting view or [DYNAMIC](@ref MArray::DYNAMIC). The default is the same number of
 *           dimensions in the tensor (if fixed), or [DYNAMIC](@ref MArray::DYNAMIC) (if not).
 *
 * @return  An immutable view.
 *
 * @ingroup funcs
 */
#if MARRAY_DOXYGEN
template <int N>
immutable_view cview(tensor_or_view x);
#else
template <typename Type, int NDim, typename Derived, bool Owner>
auto cview(const marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.cview();
}

template <typename Type, int NDim, int NIndexed, typename... Dims>
auto cview(const marray_slice<Type, NDim, NIndexed, Dims...>& x)
{
    return x.cview();
}

template <int N, typename Type, int NDim, typename Derived, bool Owner>
auto cview(const marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.template cview<N>();
}

template <int N, typename Type, int NDim, int NIndexed, typename... Dims>
auto cview(const marray_slice<Type, NDim, NIndexed, Dims...>& x)
{
    return x.template cview<N>();
}
#endif

/**
 * Return a view of the given tensor.
 *
 * @param x The tensor to view.
 *
 * @tparam N The number of dimensions in the resulting view or [DYNAMIC](@ref MArray::DYNAMIC). The default is the same number of
 *           dimensions in the tensor (if fixed), or [DYNAMIC](@ref MArray::DYNAMIC) (if not).
 *
 * @return  A possibly-mutable tensor view. For a tensor
 *          ([marray](@ref MArray::marray)), the returned view is immutable if the instance is const-qualified.
 *          For a tensor view ([marray_view](@ref MArray::marray_view)),
 *          the returned view is mutable if the value type is not
 *          const-qualified.
 *
 * @ingroup funcs
 */
#if MARRAY_DOXYGEN
template <int N>
possibly_mutable_view view(tensor_or_view x);
#else
template <typename Type, int NDim, typename Derived, bool Owner>
auto view(const marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.view();
}

template <typename Type, int NDim, typename Derived, bool Owner>
auto view(marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.view();
}

template <typename Type, int NDim, int NIndexed, typename... Dims>
auto view(const marray_slice<Type, NDim, NIndexed, Dims...>& x)
{
    return x.view();
}

template <int N, typename Type, int NDim, typename Derived, bool Owner>
auto view(const marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.template view<N>();
}

template <int N, typename Type, int NDim, typename Derived, bool Owner>
auto view(marray_base<Type, NDim, Derived, Owner>& x)
{
    return x.template view<N>();
}

template <int N, typename Type, int NDim, int NIndexed, typename... Dims>
auto view(const marray_slice<Type, NDim, NIndexed, Dims...>& x)
{
    return x.template view<N>();
}
#endif

/**
 * Write out a representation of the tensor or tensor view to the given output stream.
 * The format resembles the following (i.e., a valid C/C++ nested array initializer):
 *
 * @code
 * {
 *  {0, 1, 2},
 *  {3, 4, 5},
 *  {6, 7, 9}
 * }
 * @endcode
 *
 * @param os The output stream to write to.
 *
 * @param x  The tensor or tensor view to write.
 *
 * @returns  The output stream `os`.
 *
 * @ingroup funcs
 */
#if MARRAY_DOXYGEN
std::ostream& operator<<(std::ostream& os, tensor_or_view x)
#else
template <typename Type, int NDim, typename Derived, bool Owner>
std::ostream& operator<<(std::ostream& os, const marray_base<Type, NDim, Derived, Owner>& x)
#endif
{
    auto N = x.dimension();

    for (auto i : range(N-1))
        os << std::string(i, ' ') << "{\n";

    len_vector idx(N-1);
    auto data = x.data();
    auto& len = x.lengths();
    auto& stride = x.strides();

    for (bool done = false;!done;)
    {
        os << std::string(N-1, ' ') << '{';
        auto n = len[N-1];
        if (n > 0)
        {
            for (auto i : range(n-1))
                os << data[i*stride[N-1]] << ", ";
            os << data[(n-1)*stride[N-1]];
        }
        os << "}";

        for (auto i : reversed_range(N-1))
        {
            idx[i]++;
            data += stride[i];

            if (idx[i] >= len[i])
            {
                data -= idx[i]*stride[i];
                idx[i] = 0;
                os << "\n" << std::string(i, ' ') << '}';
                if (i == 0) done = true;
            }
            else
            {
                os << ",\n";
                for (auto j : range(i+1,N-1))
                    os << std::string(j, ' ') << "{\n";
                break;
            }
        }

        if (N == 1) break;
    }

    return os;
}

/* Inherit docs */
template <typename Type, int NDim, int NIndexed, typename... Dims>
std::ostream& operator<<(std::ostream& os, const marray_slice<Type, NDim, NIndexed, Dims...>& x)
{
    return os << x.view();
}

}

#endif //MARRAY_MARRAY_BASE_HPP
