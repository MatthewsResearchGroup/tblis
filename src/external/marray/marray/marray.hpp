#ifndef MARRAY_MARRAY_HPP
#define MARRAY_MARRAY_HPP

#include "marray_view.hpp"

namespace MArray
{

template <typename Type, int NDim, typename Allocator>
class marray : public marray_base<Type, NDim, marray<Type, NDim, Allocator>, true>
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
    template <typename, int, int, typename...> friend class marray_slice;

    protected:
        typedef marray_base<Type, NDim, marray, true> base_class;
        typedef std::allocator_traits<Allocator> alloc_traits;

        using base_class::base_;
        using base_class::len_;
        using base_class::stride_;
        using base_class::data_;
        struct storage_s : Allocator { stride_type size = 0; } storage_;
        index_base initial_base_ = DEFAULT_BASE;
        layout layout_ = DEFAULT_LAYOUT;

    public:
        using typename base_class::value_type;
        using typename base_class::pointer;
        using typename base_class::const_pointer;
        using typename base_class::reference;
        using typename base_class::const_reference;
        using typename base_class::initializer_type;

        /***********************************************************************
         *
         * @name Constructors
         *
         **********************************************************************/
        /** @{ */

        /**
         * Construct a tensor with all zero lengths, and which has no elements.
         */
        marray() {}

        /**
         * Construct a tensor by copying a std::vector of compatible type.
         *
         * @note        Only enabled for one-dimensional tensors.
         *
         * @tparam T    Type such that `T` is convertible to `Type`.
         *
         * @param v     The std::vector to copy.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        marray(const std::vector<T>& v, std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<T,value_type>>* = nullptr)
#else
        marray(const std::vector<T>& v)
#endif
        {
            reset(v);
        }

        /**
         * Copy constructor.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its base and layout are inherited.
         * Otherwise the default base and layout are used.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         */
#if MARRAY_DOXYGEN
        marray(tensor_or_view other);
#else
        template <typename U, int N, int I, typename... D>
        marray(const marray_slice<U, N, I, D...>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray(const marray_base<U, N, D, O>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray(const marray<U, N, A>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        marray(const marray& other)
        {
            reset(other);
        }
#endif

        /**
         * Copy constructor with specified layout.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its base is inherited.
         * Otherwise the default base is used.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         *
         * @param layout    The layout to use for the copied data.
         */
#if MARRAY_DOXYGEN
        marray(tensor_or_view other, layout layout);
#else
        template <typename U, int N, int I, typename... D>
        marray(const marray_slice<U, N, I, D...>& other, layout layout)
        {
            reset(other, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray(const marray_base<U, N, D, O>& other, layout layout)
        {
            reset(other, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray(const marray<U, N, A>& other, layout layout)
        {
            reset(other, layout);
        }
#endif

        /**
         * Copy constructor with specified base.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its layout is inherited.
         * Otherwise the default layout is used.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         *
         * @param base      The base to use for the copied data.
         */
#if MARRAY_DOXYGEN
        marray(tensor_or_view other, base base);
#else
        template <typename U, int N, int I, typename... D>
        marray(const marray_slice<U, N, I, D...>& other, index_base base)
        {
            reset(other, base);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray(const marray_base<U, N, D, O>& other, index_base base)
        {
            reset(other, base);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray(const marray<U, N, A>& other, index_base base)
        {
            reset(other, base);
        }
#endif

        /**
         * Copy constructor with specified base and layout.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         *
         * @param base      The base to use for the copied data.
         *
         * @param layout    The layout to use for the copied data.
         */
#if MARRAY_DOXYGEN
        marray(tensor_or_view other, base base, layout layout);
#else
        template <typename U, int N, int I, typename... D>
        marray(const marray_slice<U, N, I, D...>& other, index_base base, layout layout)
        {
            reset(other, base, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray(const marray_base<U, N, D, O>& other, index_base base, layout layout)
        {
            reset(other, base, layout);
        }
#endif

        /**
         * Copy constructor with FORTRAN/MATLAB layout.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         *
         * @param fortran   The token [FORTRAN](@ref MArray::FORTRAN) or [MATLAB](@ref MArray::MATLAB).
         */
#if MARRAY_DOXYGEN
        marray(tensor_or_view other, fortran_t fortran);
#else
        template <typename U, int N, int I, typename... D>
        marray(const marray_slice<U, N, I, D...>& other, fortran_t)
        {
            reset(other, FORTRAN);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray(const marray_base<U, N, D, O>& other, fortran_t)
        {
            reset(other, FORTRAN);
        }
#endif

        /**
         * Move constructor.
         *
         * @param other The tensor to move from. It is left in the same state as if @ref reset() was called.
         */
        marray(marray&& other)
        {
            reset(std::move(other));
        }

        /**
         * Create a tensor of the specified shape.
         *
         * This constructor should be called using uniform initialization syntax:
         *
         * @code{.cpp}
         * marray<double,4> my_tensor{3, 9, 2, 10};
         * @endcode
         *
         * The default layout is used.
         *
         * @param len       An initializer list of lengths, one for each dimension.
         */
#if MARRAY_DOXYGEN
        explicit marray(std::initializer_list<len_type> len)
#else
        template <typename Type_=Type>
        explicit marray(detail::len_type_init len)
#endif
        {
            reset(len);
        }

        /**
         * Create a tensor of the specified shape and fill value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists, except when
         *              `val` is not given. In this case, use
         *              uniform initialization syntax.
         *
         * @param val       Initialize all elements to this value. If not specified,
         *                  use value-initialization.
         */
        explicit marray(const array_1d<len_type>& len, const Type& val=Type())
        {
            reset(len, val);
        }

        /**
         * Create a tensor of the specified shape and fill value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, uninitialized_t uninitialized)
#else
        explicit marray(const array_1d<len_type>& len, uninitialized_t)
#endif
        {
            reset(len, uninitialized);
        }

        /**
         * Create a tensor of the specified shape and with the given layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
        explicit marray(const array_1d<len_type>& len, layout layout)
        {
            reset(len, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given fill value and layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param val       Initialize all elements to this value.
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
        explicit marray(const array_1d<len_type>& len, const Type& val, layout layout)
        {
            reset(len, val, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given layout and
         * without initializing tensor elements.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, uninitialized_t uninitialized, layout layout)
#else
        explicit marray(const array_1d<len_type>& len, uninitialized_t, layout layout)
#endif
        {
            reset(len, uninitialized, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given base.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         */
        explicit marray(const array_1d<len_type>& len, index_base base)
        {
            reset(len, base);
        }

        /**
         * Create a tensor of the specified shape and with the given fill value and base.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param val       Initialize all elements to this value.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         */
        explicit marray(const array_1d<len_type>& len, const Type& val, index_base base)
        {
            reset(len, val, base);
        }

        /**
         * Create a tensor of the specified shape and with the given base and
         * without initializing tensor elements.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, uninitialized_t uninitialized, base base)
#else
        explicit marray(const array_1d<len_type>& len, uninitialized_t, index_base base)
#endif
        {
            reset(len, uninitialized, base);
        }

        /**
         * Create a tensor of the specified shape and with the given base and layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
        explicit marray(const array_1d<len_type>& len, index_base base, layout layout)
        {
            reset(len, base, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given fill value, base, and layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param val       Initialize all elements to this value.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
        explicit marray(const array_1d<len_type>& len, const Type& val, index_base base, layout layout)
        {
            reset(len, val, base, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given base and layout, and
         * without initializing tensor elements.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout    The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, uninitialized_t uninitialized, base base, layout layout)
#else
        explicit marray(const array_1d<len_type>& len, uninitialized_t, index_base base, layout layout)
#endif
        {
            reset(len, uninitialized, base, layout);
        }

        /**
         * Create a tensor of the specified shape and FORTRAN/MATLAB layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param fortran    The token [FORTRAN](@ref MArray::FORTRAN).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, fortran_t fortran)
#else
        explicit marray(const array_1d<len_type>& len, fortran_t)
#endif
        {
            reset(len, FORTRAN);
        }

        /**
         * Create a tensor of the specified shape in FORTRAN/MATLAB layout, with the given fill value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param val       Initialize all elements to this value.
         *
         * @param fortran    The token [FORTRAN](@ref MArray::FORTRAN).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, const Type& val, fortran_t fortran)
#else
        explicit marray(const array_1d<len_type>& len, const Type& val, fortran_t)
#endif
        {
            reset(len, val, FORTRAN);
        }

        /**
         * Create a tensor of the specified shape in FORTRAN/MATLAB layout,
         * without initializing tensor elements.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         *
         * @param fortran    The token [FORTRAN](@ref MArray::FORTRAN).
         */
#if MARRAY_DOXYGEN
        explicit marray(const array_1d<len_type>& len, uninitialized_t uninitialized, fortran_t fortran)
#else
        explicit marray(const array_1d<len_type>& len, uninitialized_t, fortran_t)
#endif
        {
            reset(len, uninitialized, FORTRAN);
        }

        /**
         * Create a tensor with the specified data.
         *
         * @param data   A nested initializer list containing the tensor data.
         *               The level of nesting must match the number of dimensions.
         *               The lengths of the dimensions are inferred from the
         *               initializer; the provided data must be "dense", i.e. there
         *               cannot be missing values.
         *
         * @param layout The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *               If not specified, use the default layout.
         *
         * @note Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC).
         */
#if MARRAY_DOXYGEN
        marray(initializer data, layout layout = DEFAULT_LAYOUT)
#else
        template <int NDim_=NDim>
        marray(initializer_type data, layout layout = DEFAULT_LAYOUT, std::enable_if_t<(NDim_>1)>* = nullptr)
    #endif
        {
            reset(data, layout);
        }

        /**
         * Create a tensor from the given expression.
         *
         * @param other   A mathematical expression of one or more tensors.
         *                The shape of the new tensor is deduced from the tensors
         *                involved in the expression, and so should be well-defined.
         *                This means, in particular, that an implicitly broadcasted
         *                expression ([bcast](@ref MArray::slice::bcast)) is not valid. The default
         *                layout and base are used.
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression<Expression>::value>>
#endif
        marray(const Expression& other);

        ~marray()
        {
            reset();
        }

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

        marray& operator=(const marray& other)
        {
            return base_class::operator=(other);
        }

        marray& operator=(initializer_type other)
        {
            return base_class::operator=(other);
        }

        using base_class::operator=;
        using base_class::operator+=;
        using base_class::operator-=;
        using base_class::operator*=;
        using base_class::operator/=;
        using base_class::operator==;
        using base_class::operator!=;

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
         * @name Reset
         *
         **********************************************************************/
        /** @{ */

        /**
         * Reset the tensor to an empty state, with all lengths zero.
         *
         * @return      A reference to the marray.
         */
        marray& reset()
        {
            if (storage_.size)
            {
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_+i);
                }
                alloc_traits::deallocate(storage_, data_, storage_.size);
                storage_.size = 0;
            }

            base_class::reset();
            initial_base_ = DEFAULT_BASE;
            layout_ = DEFAULT_LAYOUT;

            return *this;
        }

        /**
         * Re-initialize the tensor by moving the data from another tensor.
         *
         * @param other The tensor from which to move. It is left in the state as if @ref reset() were called on it.
         *
         * @return      A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(marray&& other)
#else
        // The following is necessary to work around a defect in C++ whereby explicit
        // constructors are NOT disregarded in overload resolution, leading to an
        // ambiguity between this constructor and reset(array_1d) in something like
        // x.reset({1,2})
        template <typename T, typename=std::enable_if_t<std::is_same_v<T,marray>>>
        marray& reset(T&& other)
#endif
        {
            swap(other);
            return *this;
        }

        /**
         * Re-initialize the tensor by copying a std::vector of compatible type.
         *
         * @note        Only enabled for one-dimensional tensors.
         *
         * @tparam T    Type such that `T` is convertible to `Type`.
         *
         * @param v     The std::vector to copy.
         *
         * @return      A reference to the marray.
         */
        template <typename T>
#if !MARRAY_DOXYGEN
        std::enable_if_t<(NDim == 1 || NDim == DYNAMIC) && std::is_convertible_v<T,value_type>,marray&>
#else
        marray&
#endif
        reset(const std::vector<T>& v)
        {
            return reset(marray_view<const T,1>{v});
        }

        /**
         * Re-initialize the tensor by copying another tensor or tensor view.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its base and layout are inherited.
         * Otherwise the default base and layout are used.
         *
         * @param other The tensor, view, or partially-indexed tensor to copy.
         *
         * @return      A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(tensor_or_view other);
#else
        template <typename U, int N, int I, typename... D>
        marray& reset(const marray_slice<U, N, I, D...>& other)
        {
            return reset(other.view(), DEFAULT_BASE, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray& reset(const marray_base<U, N, D, O>& other)
        {
            return reset(other, DEFAULT_BASE, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray& reset(const marray<U, N, A>& other)
        {
            return reset(other, other.initial_base_, other.layout_);
        }
#endif

        /**
         * Re-initialize the tensor by copying another tensor or tensor view, with specified layout.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its base is inherited.
         * Otherwise the default base is used.
         *
         * @param other  The tensor, view, or partially-indexed tensor to copy.
         *
         * @param layout The layout to use for the copied data.
         *
         * @return       A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(tensor_or_view other, layout layout);
#else
        template <typename U, int N, int I, typename... D>
        marray& reset(const marray_slice<U, N, I, D...>& other, layout layout)
        {
            return reset(other.view(), DEFAULT_BASE, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray& reset(const marray_base<U, N, D, O>& other, layout layout)
        {
            return reset(other, DEFAULT_BASE, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray& reset(const marray<U, N, A>& other, layout layout)
        {
            return reset(other, other.initial_base_, layout);
        }
#endif

        /**
         * Re-initialize the tensor by copying another tensor or tensor view, with specified base.
         *
         * If `other` is a tensor ([marray](@ref MArray::marray)), then its layout is inherited.
         * Otherwise the default layout is used.
         *
         * @param other The tensor, view, or partially-indexed tensor to copy.
         *
         * @param base  The base to use for the copied data.
         *
         * @return      A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(tensor_or_view other, base base);
#else
        template <typename U, int N, int I, typename... D>
        marray& reset(const marray_slice<U, N, I, D...>& other, MArray::index_base base)
        {
            return reset(other.view(), base, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray& reset(const marray_base<U, N, D, O>& other, MArray::index_base base)
        {
            return reset(other, base, DEFAULT_LAYOUT);
        }

        /* Inherit docs */
        template <typename U, int N, typename A>
        marray& reset(const marray<U, N, A>& other, MArray::index_base base)
        {
            return reset(other, base, other.layout_);
        }
#endif

        /**
         * Re-initialize the tensor by copying another tensor or tensor view, with specified base and layout.
         *
         * @param other  The tensor, view, or partially-indexed tensor to copy.
         *
         * @param base   The base to use for the copied data.
         *
         * @param layout The layout to use for the copied data.
         *
         * @return       A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(tensor_or_view other, base base, layout layout);
#else
        template <typename U, int N, int I, typename... D>
        marray& reset(const marray_slice<U, N, I, D...>& other, MArray::index_base base, layout layout)
        {
            return reset(other.view(), base, layout);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray& reset(const marray_base<U, N, D, O>& other, MArray::index_base base, layout layout)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.lengths(), uninitialized, base, layout);
            }
            else
            {
                reset(other.lengths(), base, layout);
            }

            base_class::template operator=<>(other);

            return *this;
        }
#endif

        /**
         * Re-initialize the tensor by copying another tensor or tensor view, with FORTRAN/MATLAB layout.
         *
         * @param other   The tensor, view, or partially-indexed tensor to copy.
         *
         * @param fortran The token [FORTRAN](@ref MArray::FORTRAN) or [MATLAB](@ref MArray::MATLAB).
         *
         * @return        A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(tensor_or_view other, fortran_t fortran);
#else
        template <typename U, int N, int I, typename... D>
        marray& reset(const marray_slice<U, N, I, D...>& other, fortran_t)
        {
            return reset(other.view(), BASE_ONE, COLUMN_MAJOR);
        }

        /* Inherit docs */
        template <typename U, int N, typename D, bool O>
        marray& reset(const marray_base<U, N, D, O>& other, fortran_t)
        {
            return reset(other, BASE_ONE, COLUMN_MAJOR);
        }
#endif

        /**
         * Re-initialize to a tensor of the specified shape and fill value.
         *
         * @param len The length of each dimension. May be any one-dimensional
         *            container type with elements convertible to tensor
         *            lengths, including initializer lists, except when
         *            `val` is not given. In this case, use
         *            uniform initialization syntax.
         *
         * @param val Initialize all elements to this value. If not specified,
         *            use value-initialization.
         *
         * @return    A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, const Type& val=Type())
        {
            return reset(len, val, DEFAULT_BASE, DEFAULT_LAYOUT);
        }

        /**
         * Re-initialize to a tensor of the specified shape and fill value.
         *
         * @param len           The length of each dimension. May be any one-dimensional
         *                      container type with elements convertible to tensor
         *                      lengths, including initializer lists.
         *
         * @param uninitialized The token @ref uninitialized.
         *
         * @return              A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, uninitialized_t uninitialized)
#else
        marray& reset(const array_1d<len_type>& len, uninitialized_t)
#endif
        {
            return reset(len, uninitialized, DEFAULT_BASE, DEFAULT_LAYOUT);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given layout.
         *
         * @param len    The length of each dimension. May be any one-dimensional
         *               container type with elements convertible to tensor
         *               lengths, including initializer lists.
         *
         * @param layout The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *               [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return       A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, layout layout)
        {
            return reset(len, Type(), DEFAULT_BASE, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given fill value and layout.
         *
         * @param len    The length of each dimension. May be any one-dimensional
         *               container type with elements convertible to tensor
         *               lengths, including initializer lists.
         *
         * @param val    Initialize all elements to this value.
         *
         * @param layout The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *               [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return       A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, const Type& val, layout layout)
        {
            return reset(len, val, DEFAULT_BASE, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given layout and
         * without initializing tensor elements.
         *
         * @param len           The length of each dimension. May be any one-dimensional
         *                      container type with elements convertible to tensor
         *                      lengths, including initializer lists.
         *
         * @param uninitialized The token @ref uninitialized.
         *
         * @param layout        The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *                      [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return              A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, uninitialized_t uninitialized, layout layout)
#else
        marray& reset(const array_1d<len_type>& len, uninitialized_t, layout layout)
#endif
        {
            return reset(len, uninitialized, DEFAULT_BASE, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given base.
         *
         * @param len  The length of each dimension. May be any one-dimensional
         *             container type with elements convertible to tensor
         *             lengths, including initializer lists.
         *
         * @param base The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *             [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *             [MATLAB](@ref MArray::MATLAB)).
         *
         * @return     A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, MArray::index_base base)
        {
            return reset(len, Type(), base, DEFAULT_LAYOUT);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given fill value and base.
         *
         * @param len  The length of each dimension. May be any one-dimensional
         *             container type with elements convertible to tensor
         *             lengths, including initializer lists.
         *
         * @param val  Initialize all elements to this value.
         *
         * @param base The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *             [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *             [MATLAB](@ref MArray::MATLAB)).
         *
         * @return     A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, const Type& val, MArray::index_base base)
        {
            return reset(len, val, base, DEFAULT_LAYOUT);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given base and
         * without initializing tensor elements.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param uninitialized   The token @ref uninitialized.
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, uninitialized_t uninitialized, base base)
#else
        marray& reset(const array_1d<len_type>& len, uninitialized_t, MArray::index_base base)
#endif
        {
            return reset(len, uninitialized, base, DEFAULT_LAYOUT);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given base and layout.
         *
         * @param len    The length of each dimension. May be any one-dimensional
         *               container type with elements convertible to tensor
         *               lengths, including initializer lists.
         *
         * @param base   The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *               [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *               [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *               [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return       A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, MArray::index_base base, layout layout)
        {
            return reset(len, Type(), base, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given fill value, base, and layout.
         *
         * @param len    The length of each dimension. May be any one-dimensional
         *               container type with elements convertible to tensor
         *               lengths, including initializer lists.
         *
         * @param val    Initialize all elements to this value.
         *
         * @param base   The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *               [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *               [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *               [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return       A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, const Type& val, MArray::index_base base, layout layout)
        {
            reset(len, uninitialized, base, layout);
            std::uninitialized_fill_n(data(), size(), val);
            return *this;
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given base and layout, and
         * without initializing tensor elements.
         *
         * @param len           The length of each dimension. May be any one-dimensional
         *                      container type with elements convertible to tensor
         *                      lengths, including initializer lists.
         *
         * @param uninitialized The token @ref uninitialized.
         *
         * @param base          The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *                      [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                      [MATLAB](@ref MArray::MATLAB)).
         *
         * @param layout        The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *                      [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @return              A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, uninitialized_t uninitialized, base base, layout layout)
#else
        marray& reset(const array_1d<len_type>& len, uninitialized_t, MArray::index_base base, layout layout)
#endif
        {
            reset();

            initial_base_ = base;
            layout_ = layout;
            storage_.size = size(len);
            base_class::reset(len, alloc_traits::allocate(storage_, storage_.size), base, layout);

            return *this;
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given base and layout.
         *
         * @param len     The length of each dimension. May be any one-dimensional
         *                container type with elements convertible to tensor
         *                lengths, including initializer lists.
         *
         * @param layout  The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or
         *                [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @param base    The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *                [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *                [MATLAB](@ref MArray::MATLAB)).
         *
         * @return        A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, layout layout, MArray::index_base base)
        {
            return reset(len, Type(), base, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and with the given fill value, base, and layout.
         *
         * @param len    The length of each dimension. May be any one-dimensional
         *               container type with elements convertible to tensor
         *               lengths, including initializer lists.
         *
         * @param val    Initialize all elements to this value.
         *
         * @param layout The layout to use, either [ROW_MAJOR](@ref MArray::ROW_MAJOR) or [COLUMN_MAJOR](@ref MArray::COLUMN_MAJOR).
         *
         * @param base   The base to use, either [BASE_ZERO](@ref MArray::BASE_ZERO) or
         *               [BASE_ONE](@ref MArray::BASE_ONE) (a.k.a. [FORTRAN](@ref MArray::FORTRAN) or
         *               [MATLAB](@ref MArray::MATLAB)).
         *
         * @return       A reference to the marray.
         */
        marray& reset(const array_1d<len_type>& len, const Type& val, layout layout, MArray::index_base base)
        {
            return reset(len, val, base, layout);
        }

        /**
         * Re-initialize to a tensor of the specified shape and FORTRAN/MATLAB layout.
         *
         * @param len     The length of each dimension. May be any one-dimensional
         *                container type with elements convertible to tensor
         *                lengths, including initializer lists.
         *
         * @param fortran The token [FORTRAN](@ref MArray::FORTRAN).
         *
         * @return        A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, fortran_t fortran)
#else
        marray& reset(const array_1d<len_type>& len, fortran_t)
#endif
        {
            return reset(len, Type(), BASE_ONE, COLUMN_MAJOR);
        }

        /**
         * Re-initialize to a tensor of the specified shape in FORTRAN/MATLAB layout, with the given fill value.
         *
         * @param len     The length of each dimension. May be any one-dimensional
         *                container type with elements convertible to tensor
         *                lengths, including initializer lists.
         *
         * @param val     Initialize all elements to this value.
         *
         * @param fortran The token [FORTRAN](@ref MArray::FORTRAN).
         *
         * @return        A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, const Type& val, fortran_t fortran)
#else
        marray& reset(const array_1d<len_type>& len, const Type& val, fortran_t)
#endif
        {
            return reset(len, val, BASE_ONE, COLUMN_MAJOR);
        }

        /**
         * Re-initialize to a tensor of the specified shape in FORTRAN/MATLAB layout,
         * without initializing tensor elements.
         *
         * @param len           The length of each dimension. May be any one-dimensional
         *                      container type with elements convertible to tensor
         *                      lengths, including initializer lists.
         *
         * @param uninitialized The token @ref uninitialized.
         *
         * @param fortran       The token [FORTRAN](@ref MArray::FORTRAN).
         *
         * @return              A reference to the marray.
         */
#if MARRAY_DOXYGEN
        marray& reset(const array_1d<len_type>& len, uninitialized_t uninitialized, fortran_t fortran)
#else
        marray& reset(const array_1d<len_type>& len, uninitialized_t, fortran_t)
#endif
        {
            return reset(len, uninitialized, BASE_ONE, COLUMN_MAJOR);
        }

        /**
         * Re-inititialize a tensor from the specified data.
         *
         * @param data   A nested initializer list containing the tensor data.
         *               The level of nesting must match the number of dimensions.
         *               The lengths of the dimensions are inferred from the
         *               initializer; the provided data must be "dense", i.e. there
         *               cannot be missing values.
         *
         * @param layout The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *               If not specified, use the default layout.
         *
         * @return       A reference to the marray.
         *
         * @note         Only available when `NDim != ` [DYNAMIC](@ref MArray::DYNAMIC) or 1.
         */
#if !MARRAY_DOXYGEN
        template <int NDim_=NDim, typename=std::enable_if_t<NDim_!=1 && NDim_ != DYNAMIC>,marray&>
#endif
        marray& reset(initializer_type data, layout layout = DEFAULT_LAYOUT)
        {
            detail::array_type_t<len_type, NDim> len(NDim);
            base_class::set_lengths_(0, len, data);
            reset(len, layout);
            base_class::set_data_(0, data_, data);
            return *this;
        }

        /** @} */
        /***********************************************************************
         *
         * @name Resize
         *
         **********************************************************************/
        /** @{ */

        /**
         * Resize the tensor.
         *
         * After resizing, any elements whose indicies fall within the bounds of
         * the new shape are retained. Any new elements are initialized as requested.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param len The lengths of the dimensions are resizing.
         *
         * @param val The value to use for initializing new element (those
         *            whose indices fall outside the bounds of the original
         *            shape). If not specified, value-initialization is used.
         *
         * @return    A reference to the marray.
         */
        marray& resize(const array_1d<len_type>& len, const Type& val=Type())
        {
            detail::array_type_t<len_type, NDim> new_len;
            len.slurp(new_len);

            if (new_len == len_)
                return *this;

            marray a(std::move(*this));

            reset(len, val, a.initial_base_, a.layout_);
            base_ = a.base_;

            auto b = view();

            /*
             * It is OK to change the geometry of 'a' even if it is not
             * a view since it is about to go out of scope.
             */
            for (auto i : range(dimension()))
            {
                auto l = std::min(a.length(i), b.length(i));
                a.len_[i] = l;
                b.length(i, l);
            }

            b = a;

            return *this;
        }

        /** @} */
        /***********************************************************************
         *
         * @name Push/pop
         *
         **********************************************************************/
        /** @{ */

        /**
         * Append a new value to the end of the vector.
         *
         * This overload is only available for vectors.
         *
         * The length of the vector is increased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param x     The value to append.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=std::enable_if_t<N==1>>
#endif
        void push_back(const Type& x)
        {
            push_back(0, x);
        }

        /**
         * Append a new element or face to the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is increased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param dim   The dimension along which to append.
         *
         * @param x     The element (if this is a vector) or face to append.
         */
#if MARRAY_DOXYGEN
        void push_back(int dim, element_or_tensor_or_view x);
#else
        template <typename=void, int M=NDim, typename=std::enable_if_t<M==1>>
        void push_back(int dim, const Type& x)
        {
            MARRAY_ASSERT(dim == 0);
            resize({length()+1});
            back() = x;
        }

        template <typename U, int N, typename D, bool O, int M=NDim,
            typename=std::enable_if_t<M!=1>>
        void push_back(int dim, const marray_base<U, N, D, O>& x)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            auto len = len_;
            len[dim]++;
            resize(len);
            back(dim) = x;
        }
#endif

        /**
         * Remove the element at the end of the vector.
         *
         * This overload is only available for vectors.
         *
         * The length of the vector is decresed by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=std::enable_if_t<N==1>>
#endif
        void pop_back()
        {
            resize({length()-1});
        }

        /**
         * Remove the element or face at the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is decreased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param dim   The dimension along which to remove an element or face.
         */
        void pop_back(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(length(dim) > 0);

            auto len = len_;
            len[dim]--;
            resize(len);
        }

        /** @} */
        /***********************************************************************
         *
         * @name Basic setters
         *
         **********************************************************************/
        /** @{ */

        /**
         * The number of elements in the tensor.
         *
         * Since the elements are arranged contiguously, this is also the size
         * of the underlying storage.
         *
         * @return The number of elements in the tensor.
         */
        stride_type size() const
        {
            return storage_.size;
        }

        /** @} */
        /***********************************************************************
         *
         * @name Swap
         *
         **********************************************************************/
        /** @{ */

        /**
         * Swap the shape, data, and layout of this tensor with those of another.
         *
         * @param other     The tensor to swap with.
         */
        void swap(marray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(initial_base_, other.initial_base_);
            swap(layout_, other.layout_);
            base_class::swap(other);
        }

        /** @} */
};

/**
 * Swap the shape, data, and layout of two tensors
 *
 * @param a     The first tensor to swap.
 *
 * @param b     The second tensor to swap.
 *
 * @ingroup funcs
*/
template <typename Type, int NDim, typename Allocator>
void swap(marray<Type, NDim, Allocator>& a, marray<Type, NDim, Allocator>& b)
{
    a.swap(b);
}

}

#endif //MARRAY_MARRAY_HPP
