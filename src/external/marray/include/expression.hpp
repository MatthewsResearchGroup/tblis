#ifndef _MARRAY_EXPRESSION_HPP_
#define _MARRAY_EXPRESSION_HPP_

#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE4_1__)

#include <x86intrin.h>

#endif

#include "utility.hpp"
#include "vector.hpp"

namespace MArray
{

struct bcast_dim
{
    len_type len;

    bcast_dim(len_type len) : len(len) {}
};

struct slice_dim
{
    len_type len;
    stride_type stride;

    slice_dim(len_type len, stride_type stride)
    : len(len), stride(stride) {}
};

}

#include "marray.hpp"

namespace MArray
{

template <typename T>
struct is_scalar : std::is_arithmetic<T> {};

template <typename T>
struct is_scalar<std::complex<T>> : std::is_floating_point<T> {};

template <typename T, typename... Dims>
struct array_expr
{
    typedef T& result_type;
    typedef vector_traits<typename std::remove_cv<T>::type> vec_traits;
    typedef typename vec_traits::vector_type vector_type;

    T* data;
    std::tuple<Dims...> dims;

    template <size_t I>
    using dim_type = typename std::tuple_element<I, std::tuple<Dims...>>::type;

    array_expr(T* data, const Dims&... dims)
    : data(data), dims{dims...} {}

    result_type eval() const
    {
        return *data;
    }

    template <unsigned NDim, unsigned Dim>
    detail::enable_if_t<(Dim < NDim-sizeof...(Dims)),result_type>
    eval_at(len_type) const
    {
        return *data;
    }

    template <unsigned NDim, unsigned Dim>
    detail::enable_if_t<(Dim >= NDim-sizeof...(Dims)),result_type>
    eval_at(len_type i) const
    {
        return eval_at(i, std::get<Dim-(NDim-sizeof...(Dims))>(dims));
    }

    result_type eval_at(len_type i, const slice_dim&) const
    {
        return data[i];
    }

    result_type eval_at(len_type, const bcast_dim&) const
    {
        return *data;
    }

    template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
    detail::enable_if_t<(Dim < NDim-sizeof...(Dims)),vector_type>
    eval_vec(len_type) const
    {
        return vec_traits::load1(data);
    }

    template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
    detail::enable_if_t<(Dim >= NDim-sizeof...(Dims)),vector_type>
    eval_vec(len_type i) const
    {
        return eval_vec<Width,Aligned>(i, std::get<Dim-(NDim-sizeof...(Dims))>(dims));
    }

    template <unsigned Width, bool Aligned>
    vector_type eval_vec(len_type i, const slice_dim&) const
    {
        return vec_traits::template load<Width,Aligned>(data+i);
    }

    template <unsigned Width, bool Aligned>
    vector_type eval_vec(len_type, const bcast_dim&) const
    {
        return vec_traits::load1(data);
    }

    template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
    void store_vec(len_type i, vector_type v) const
    {
        vec_traits::template store<Width,Aligned>(v, data+i);
    }
};

template <typename Expr, typename=void> struct is_array_expression;

template <typename Expr, typename=void> struct is_expression;

template <typename Expr, typename=void> struct is_unary_expression;

template <typename Expr, typename=void> struct is_binary_expression;

template <typename Expr, typename=void> struct expr_result_type;

template <typename Expr>
struct expr_result_type<Expr, detail::enable_if_t<is_scalar<detail::decay_t<Expr>>::value>>
{
    typedef detail::decay_t<Expr> type;
};

template <typename Expr>
struct expr_result_type<Expr, detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value>>
{
    typedef typename detail::decay_t<Expr>::result_type type;
};

template <typename Expr>
detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value,
                    typename expr_result_type<detail::decay_t<Expr>>::type>
eval(Expr&& expr)
{
    return expr.eval();
}

template <typename Expr>
detail::enable_if_t<is_scalar<detail::decay_t<Expr>>::value,Expr>
eval(Expr&& expr)
{
    return expr;
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value,
                    typename expr_result_type<detail::decay_t<Expr>>::type>
eval_at(Expr&& expr, len_type i)
{
    return expr.template eval_at<NDim, Dim>(i);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_scalar<detail::decay_t<Expr>>::value,Expr>
eval_at(Expr&& expr, len_type)
{
    return expr;
}

template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned, typename Expr>
detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value,
                    typename detail::decay_t<Expr>::vector_type>
eval_vec(Expr&& expr, len_type i)
{
    return expr.template eval_vec<NDim, Dim, Width, Aligned>(i);
}

template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned, typename Expr>
detail::enable_if_t<is_scalar<detail::decay_t<Expr>>::value,
                    typename vector_traits<detail::decay_t<Expr>>::vector_type>
eval_vec(Expr&& expr, len_type)
{
    return vector_traits<detail::decay_t<Expr>>::set1(expr);
}

namespace operators
{

template <typename T>
using vector_type = typename vector_traits<T>::vector_type;

template <typename T, typename U, typename=void>
struct binary_result_type;

template <typename T, typename U>
struct binary_result_type<T, U,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef typename std::common_type<T, U>::type type;
};

template <typename T, typename U>
struct binary_result_type<T, std::complex<U>,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef T type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, U,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, std::complex<U>,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, U,
    detail::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef std::complex<T> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    detail::enable_if_t<std::is_integral<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef U type;
};

template <typename T, typename U>
struct binary_result_type<T, std::complex<U>,
    detail::enable_if_t<std::is_integral<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<U> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    detail::enable_if_t<std::is_integral<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef typename std::common_type<T,U>::type type;
};

struct plus
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        typename binary_result_type<T, U>::type>
    {
        typedef typename binary_result_type<T, U>::type V;
        return V(a)+V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
                        decltype(a+b)>
    {
        return a+b;
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a, vector_type<T> b) const
    {
        return vector_traits<T>::add(a, b);
    }
};

struct minus
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        typename binary_result_type<T, U>::type>
    {
        typedef typename binary_result_type<T, U>::type V;
        return V(a)-V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
                        decltype(a-b)>
    {
        return a-b;
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a, vector_type<T> b) const
    {
        return vector_traits<T>::sub(a, b);
    }
};

struct multiplies
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        typename binary_result_type<T, U>::type>
    {
        typedef typename binary_result_type<T, U>::type V;
        return V(a)*V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
                        decltype(a*b)>
    {
        return a*b;
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a, vector_type<T> b) const
    {
        return vector_traits<T>::mul(a, b);
    }
};

struct divides
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        typename binary_result_type<T, U>::type>
    {
        typedef typename binary_result_type<T, U>::type V;
        return V(a)/V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    detail::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
                        decltype(a/b)>
    {
        return a/b;
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a, vector_type<T> b) const
    {
        return vector_traits<T>::div(a, b);
    }
};

struct pow
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const -> decltype(std::pow(a,b))
    {
        return std::pow(a,b);
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a, vector_type<T> b) const
    {
        return vector_traits<T>::pow(a, b);
    }
};

struct negate
{
    template <typename T>
    auto operator()(const T& a) const -> decltype(-a)
    {
        return -a;
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a) const
    {
        return vector_traits<T>::negate(a);
    }
};

struct exp
{
    template <typename T>
    auto operator()(const T& a) const -> decltype(std::exp(a))
    {
        return std::exp(a);
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a) const
    {
        return vector_traits<T>::exp(a);
    }
};

struct sqrt
{
    template <typename T>
    auto operator()(const T& a) const -> decltype(std::sqrt(a))
    {
        return std::sqrt(a);
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a) const
    {
        return vector_traits<T>::sqrt(a);
    }
};

}

template <typename LHS, typename RHS, typename Op>
struct binary_expr
{
    typedef LHS first_type;
    typedef RHS second_type;
    typedef detail::decay_t<typename expr_result_type<LHS>::type> first_result_type;
    typedef detail::decay_t<typename expr_result_type<RHS>::type> second_result_type;
    typedef decltype(std::declval<Op>()(
        std::declval<first_result_type>(),
        std::declval<second_result_type>())) result_type;
    typedef typename vector_traits<result_type>::vector_type vector_type;

    LHS first;
    RHS second;
    Op op;

    binary_expr(const LHS& first, const RHS& second, const Op& op = Op())
    : first(first), second(second), op(op) {}

    result_type eval() const
    {
        return op(MArray::eval(first), MArray::eval(second));
    }

    template <unsigned NDim, unsigned Dim>
    result_type eval_at(len_type i) const
    {
        return op(MArray::eval_at<NDim, Dim>(first, i),
                  MArray::eval_at<NDim, Dim>(second, i));
    }

    template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
    vector_type eval_vec(len_type i) const
    {
        return op.template vec<result_type>(
            vector_traits<first_result_type>::template convert<result_type>(
                MArray::eval_vec<NDim, Dim, Width, Aligned>(first, i)),
            vector_traits<second_result_type>::template convert<result_type>(
                MArray::eval_vec<NDim, Dim, Width, Aligned>(second, i)));
    }
};

template <typename LHS, typename RHS>
using add_expr = binary_expr<LHS, RHS, operators::plus>;

template <typename LHS, typename RHS>
using sub_expr = binary_expr<LHS, RHS, operators::minus>;

template <typename LHS, typename RHS>
using mul_expr = binary_expr<LHS, RHS, operators::multiplies>;

template <typename LHS, typename RHS>
using div_expr = binary_expr<LHS, RHS, operators::divides>;

template <typename Base, typename Exponent>
using pow_expr = binary_expr<Base, Exponent, operators::pow>;

template <typename Expr, typename Op>
struct unary_expr
{
    typedef Expr expr_type;
    typedef detail::decay_t<typename expr_result_type<Expr>::type> input_result_type;
    typedef decltype(std::declval<Op>()(
        std::declval<input_result_type>())) result_type;
    typedef typename vector_traits<result_type>::vector_type vector_type;

    Expr expr;
    Op op;

    unary_expr(const Expr& expr, const Op& op = Op())
    : expr(expr), op(op) {}

    result_type eval() const
    {
        return op(MArray::eval(expr));
    }

    template <unsigned NDim, unsigned Dim>
    result_type eval_at(len_type i) const
    {
        return op(MArray::eval_at<NDim, Dim>(expr, i));
    }

    template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
    vector_type eval_vec(len_type i) const
    {
        return op.template vec<result_type>(
            vector_traits<input_result_type>::template convert<result_type>(
                MArray::eval_vec<NDim, Dim, Width, Aligned>(expr, i)));
    }
};

template <typename Expr>
using negate_expr = unary_expr<Expr, operators::negate>;

template <typename Expr>
using exp_expr = unary_expr<Expr, operators::exp>;

template <typename Expr>
using sqrt_expr = unary_expr<Expr, operators::sqrt>;

template <typename T, typename>
struct is_array_expression : std::false_type {};

template <typename T, typename... Dims>
struct is_array_expression<array_expr<T, Dims...>> : std::true_type {};

template <typename T, typename>
struct is_expression : std::false_type {};

template <typename T, typename... Dims>
struct is_expression<array_expr<T, Dims...>> : std::true_type {};

template <typename LHS, typename RHS, typename Op>
struct is_expression<binary_expr<LHS, RHS, Op>> : std::true_type {};

template <typename Expr, typename Op>
struct is_expression<unary_expr<Expr, Op>> : std::true_type {};

template <typename Expr, typename>
struct is_unary_expression : std::false_type {};

template <typename Expr>
struct is_unary_expression<Expr,
    detail::enable_if_t<is_expression<typename Expr::expr_type>::value>>
    : std::true_type {};

template <typename Expr, typename>
struct is_binary_expression : std::false_type {};

template <typename Expr>
struct is_binary_expression<Expr,
    detail::enable_if_t<is_expression<typename Expr::first_type>::value ||
                        is_expression<typename Expr::second_type>::value>>
    : std::true_type {};

template <typename Array, typename Dim>
struct array_expr_type_helper2;

template <typename T, typename... Dims, typename Dim>
struct array_expr_type_helper2<array_expr<T, Dims...>, Dim>
{
    typedef array_expr<T, Dims..., Dim> type;
};

template <typename T, unsigned NDim>
struct array_expr_helper;

template <typename T>
struct array_expr_helper<T, 0>
{
    typedef array_expr<T> type;
};

template <typename T, unsigned NDim>
struct array_expr_helper
{
    typedef typename array_expr_type_helper2<
        typename array_expr_helper<T, NDim-1>::type, slice_dim>::type type;
};

template <typename Array, typename... Dims1>
struct slice_array_expr_helper;

template <typename T, typename... Dims2, typename... Dims1>
struct slice_array_expr_helper<array_expr<T, Dims2...>, Dims1...>
{
    typedef array_expr<T, Dims1..., Dims2...> type;
};

template <typename Array, typename=void>
struct is_marray
{
    template <typename T, unsigned NDim, typename Derived, bool Owner>
    static std::true_type check(const marray_base<T, NDim, Derived, Owner>*);

    template <typename T, unsigned NDim, unsigned NIndexed, typename... Dims>
    static std::true_type check(const marray_slice<T, NDim, NIndexed, Dims...>*);

    static std::false_type check(...);

    static constexpr bool value = decltype(check((Array*)0))::value;
};

template <typename Expr, typename=void>
struct expression_type;

template <typename Expr>
struct expression_type<Expr, detail::enable_if_t<is_marray<Expr>::value>>
{
    template <typename T, unsigned NDim, unsigned NIndexed, typename... Dims>
    static typename slice_array_expr_helper<typename array_expr_helper<T, NDim-NIndexed>::type, Dims...>::type check(const marray_slice<T, NDim, NIndexed, Dims...>*);

    template <typename T, unsigned NDim, typename Derived>
    static typename array_expr_helper<T, NDim>::type check(const marray_base<T, NDim, Derived, false>*);

    template <typename T, unsigned NDim, typename Derived>
    static typename array_expr_helper<const T, NDim>::type check(const marray_base<T, NDim, Derived, true>*);

    template <typename T, unsigned NDim, typename Derived>
    static typename array_expr_helper<T, NDim>::type check(marray_base<T, NDim, Derived, true>*);

    static void check(...);

    typedef decltype(check((Expr*)0)) type;
};

template <typename Expr>
struct expression_type<Expr, detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value ||
                                                 is_scalar<detail::decay_t<Expr>>::value>>
{
    typedef detail::decay_t<Expr> type;
};

template <typename T, unsigned NDim, unsigned NIndexed, typename... Dims,
          size_t... I, size_t... J>
typename expression_type<marray_slice<T, NDim, NIndexed, Dims...>>::type
make_expression_helper(const marray_slice<T, NDim, NIndexed, Dims...>& x,
                       detail::integer_sequence<size_t, I...>,
                       detail::integer_sequence<size_t, J...>)
{
    return {x.data(), x.template dim<I>()...,
            slice_dim(x.template base_length<NIndexed+J>(),
                      x.template base_stride<NIndexed+J>())...};
}

template <typename T, unsigned NDim, typename Derived, bool Owner, size_t... I>
typename expression_type<const marray_base<T, NDim, Derived, Owner>>::type
make_expression_helper(const marray_base<T, NDim, Derived, Owner>& x,
                       detail::integer_sequence<size_t, I...>)
{
    return {x.data(), slice_dim(x.template length<I>(), x.template stride<I>())...};
}

template <typename T, unsigned NDim, typename Derived, bool Owner, size_t... I>
typename expression_type<marray_base<T, NDim, Derived, Owner>>::type
make_expression_helper(marray_base<T, NDim, Derived, Owner>& x,
                       detail::integer_sequence<size_t, I...>)
{
    return {x.data(), slice_dim(x.template length<I>(), x.template stride<I>())...};
}

template <typename T, unsigned NDim, unsigned NIndexed, typename... Dims>
typename expression_type<marray_slice<T, NDim, NIndexed, Dims...>>::type
make_expression(const marray_slice<T, NDim, NIndexed, Dims...>& x)
{
    return make_expression_helper(x, detail::static_range<size_t, sizeof...(Dims)>(),
                                  detail::static_range<size_t, NDim-NIndexed>());
}

template <typename T, unsigned NDim, typename Derived, bool Owner>
typename expression_type<const marray_base<T, NDim, Derived, Owner>>::type
make_expression(const marray_base<T, NDim, Derived, Owner>& x)
{
    return make_expression_helper(x, detail::static_range<size_t, NDim>());
}

template <typename T, unsigned NDim, typename Derived, bool Owner>
typename expression_type<marray_base<T, NDim, Derived, Owner>>::type
make_expression(marray_base<T, NDim, Derived, Owner>& x)
{
    return make_expression_helper(x, detail::static_range<size_t, NDim>());
}

template <typename Expr>
detail::enable_if_t<is_expression<detail::decay_t<Expr>>::value ||
                    is_scalar<detail::decay_t<Expr>>::value, Expr&&>
make_expression(Expr&& x)
{
    return std::forward<Expr>(x);
}

template <typename Expr>
struct is_expression_arg :
    std::integral_constant<bool, is_expression<detail::decay_t<Expr>>::value ||
                                 is_marray<detail::decay_t<Expr>>::value> {};

template <typename Expr>
struct is_expression_arg_or_scalar :
    std::integral_constant<bool, is_expression_arg<detail::decay_t<Expr>>::value ||
                                 is_scalar<detail::decay_t<Expr>>::value> {};

template <typename T, typename Expr, typename=void>
struct converted_scalar_type {};

template <typename T, typename Expr>
struct converted_scalar_type<T, Expr, detail::enable_if_t<is_scalar<T>::value &&
                                                          is_expression_arg<Expr>::value>> :
    std::common_type<T, typename expr_result_type<typename expression_type<Expr>::type>::type> {};

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    add_expr<typename expression_type<const LHS>::type,
                             typename expression_type<const RHS>::type>>
operator+(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    add_expr<typename expression_type<const LHS>::type,
                             typename converted_scalar_type<RHS, LHS>::type>>
operator+(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<RHS, LHS>::type T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    add_expr<typename converted_scalar_type<LHS, RHS>::type,
                             typename expression_type<const RHS>::type>>
operator+(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<LHS, RHS>::type T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    sub_expr<typename expression_type<const LHS>::type,
                             typename expression_type<const RHS>::type>>
operator-(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    sub_expr<typename expression_type<const LHS>::type,
                             typename converted_scalar_type<RHS, LHS>::type>>
operator-(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<RHS, LHS>::type T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    sub_expr<typename converted_scalar_type<LHS, RHS>::type,
                             typename expression_type<const RHS>::type>>
operator-(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<LHS, RHS>::type T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    mul_expr<typename expression_type<const LHS>::type,
                             typename expression_type<const RHS>::type>>
operator*(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    mul_expr<typename expression_type<const LHS>::type,
                             typename converted_scalar_type<RHS, LHS>::type>>
operator*(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<RHS, LHS>::type T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    mul_expr<typename converted_scalar_type<LHS, RHS>::type,
                             typename expression_type<const RHS>::type>>
operator*(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<LHS, RHS>::type T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    div_expr<typename expression_type<const LHS>::type,
                             typename expression_type<const RHS>::type>>
operator/(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    div_expr<typename expression_type<const LHS>::type,
                             typename converted_scalar_type<RHS, LHS>::type>>
operator/(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<RHS, LHS>::type T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    div_expr<typename converted_scalar_type<LHS, RHS>::type,
                             typename expression_type<const RHS>::type>>
operator/(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<LHS, RHS>::type T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    pow_expr<typename expression_type<const LHS>::type,
                             typename expression_type<const RHS>::type>>
pow(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    pow_expr<typename expression_type<const LHS>::type,
                             typename converted_scalar_type<RHS, LHS>::type>>
pow(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<RHS, LHS>::type T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
detail::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    pow_expr<typename converted_scalar_type<LHS, RHS>::type,
                             typename expression_type<const RHS>::type>>
pow(const LHS& lhs, const RHS& rhs)
{
    typedef typename converted_scalar_type<LHS, RHS>::type T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename Expr>
detail::enable_if_t<is_expression_arg<Expr>::value,
                    negate_expr<typename expression_type<const Expr>::type>>
operator-(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename Expr>
detail::enable_if_t<is_expression_arg<Expr>::value,
                    exp_expr<typename expression_type<const Expr>::type>>
exp(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename Expr>
detail::enable_if_t<is_expression_arg<Expr>::value,
                    sqrt_expr<typename expression_type<const Expr>::type>>
sqrt(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename Expr, typename=void> struct expr_dimension;

template <typename T, typename... Dims>
struct expr_dimension<array_expr<T, Dims...>>
    : std::integral_constant<unsigned, sizeof...(Dims)> {};

template <typename Expr>
struct expr_dimension<Expr, detail::enable_if_t<is_scalar<Expr>::value>>
    : std::integral_constant<unsigned, 0> {};

template <typename Expr>
struct expr_dimension<Expr, detail::enable_if_t<is_binary_expression<Expr>::value>>
    : detail::conditional_t<(expr_dimension<typename Expr::first_type>::value <
                             expr_dimension<typename Expr::second_type>::value),
                             expr_dimension<typename Expr::second_type>,
                             expr_dimension<typename Expr::first_type>> {};

template <typename Expr>
struct expr_dimension<Expr, detail::enable_if_t<is_unary_expression<Expr>::value>>
    : expr_dimension<typename Expr::expr_type> {};

template <typename Dim>
detail::enable_if_t<std::is_same<Dim,bcast_dim>::value,len_type>
get_array_length(const Dim&)
{
    static_assert(!std::is_same<Dim,bcast_dim>::value,
                  "Broadcast dimensions cannot be written to");
    return 0;
}

template <typename Dim>
detail::enable_if_t<std::is_same<Dim,slice_dim>::value,len_type>
get_array_length(const Dim& dim)
{
    return dim.len;
}

template <typename T, typename... Dims, size_t... I>
std::array<len_type, sizeof...(I)>
get_array_lengths_helper(const array_expr<T, Dims...>& array,
                         detail::integer_sequence<size_t, I...>)
{
    return {get_array_length(std::get<I>(array.dims))...};
}

template <typename T, typename... Dims>
std::array<len_type, sizeof...(Dims)>
get_array_lengths(const array_expr<T, Dims...>& array)
{
    return get_array_lengths_helper(array, detail::static_range<size_t, sizeof...(Dims)>());
}

inline bool check_expr_length(const bcast_dim&, len_type)
{
    return true;
}

inline bool check_expr_length(const slice_dim& dim, len_type len)
{
    return len == dim.len;
}

template <typename T, typename... Dims, size_t NDim, size_t... I>
bool check_expr_lengths_helper(const array_expr<T, Dims...>& array,
                               const std::array<len_type, NDim>& len,
                               detail::integer_sequence<size_t, I...>)
{
    std::array<bool, sizeof...(Dims)> values =
        {check_expr_length(std::get<I>(array.dims),
                           len[NDim-sizeof...(Dims)+I])...};

    bool ret = true;
    for (bool value : values) ret = ret && value;
    return ret;
}

template <typename T, typename... Dims, size_t NDim>
bool check_expr_lengths(const array_expr<T, Dims...>& array,
                        const std::array<len_type, NDim>& len)
{
    return check_expr_lengths_helper(array, len, detail::static_range<size_t, sizeof...(Dims)>());
}

template <typename Expr, size_t NDim>
detail::enable_if_t<is_scalar<Expr>::value,bool>
check_expr_lengths(const Expr&, const std::array<len_type, NDim>&)
{
    return true;
}

template <typename Expr, size_t NDim>
detail::enable_if_t<is_binary_expression<Expr>::value,bool>
check_expr_lengths(const Expr& expr, const std::array<len_type, NDim>& len)
{
    return check_expr_lengths(expr.first, len) &&
           check_expr_lengths(expr.second, len);
}

template <typename Expr, size_t NDim>
detail::enable_if_t<is_unary_expression<Expr>::value,bool>
check_expr_lengths(const Expr& expr, const std::array<len_type, NDim>& len)
{
    return check_expr_lengths(expr.expr, len);
}

/*
 * Return true if the dimension is vectorizable (stride-1).
 * Broadcast dimensions are trivially vectorizable.
 */
inline bool is_contiguous(const bcast_dim&)
{
    return true;
}

inline bool is_contiguous(const slice_dim& dim)
{
    return dim.stride == 1;
}

/*
 * Dim is one of the NDim dimensions of the array being assigned to. Since
 * the number of dimension in the subexpression (sizeof...(Dims)) can be
 * smaller, the initial implicit broadcast dimensions are trivially
 * vectorizable.
 */
template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim < NDim-sizeof...(Dims)), bool>
is_array_contiguous(const array_expr<T, Dims...>&)
{
    return true;
}

/*
 * For the remaining sizeof...(Dims) dimensions, subtract NDim-sizeof...(Dims)
 * to get the proper dimension number in the subexpression.
 */
template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim >= NDim-sizeof...(Dims)), bool>
is_array_contiguous(const array_expr<T, Dims...>& expr)
{
    return is_contiguous(std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
bool is_contiguous(const array_expr<T, Dims...>& expr)
{
    return is_array_contiguous<NDim, Dim, T, Dims...>(expr);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_scalar<Expr>::value, bool>
is_contiguous(const Expr&)
{
    return true;
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_binary_expression<Expr>::value, bool>
is_contiguous(const Expr& expr)
{
    return is_contiguous<NDim, Dim>(expr.first) &&
           is_contiguous<NDim, Dim>(expr.second);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_unary_expression<Expr>::value, bool>
is_contiguous(const Expr& expr)
{
    return is_contiguous<NDim, Dim>(expr.expr);
}

template <typename Expr, typename=void>
struct vector_width;

template <typename T, typename... Dims>
struct vector_width<array_expr<T, Dims...>>
{
    constexpr static unsigned value = vector_traits<typename std::remove_cv<T>::type>::vector_width;
};

template <typename Expr>
struct vector_width<Expr,detail::enable_if_t<is_scalar<Expr>::value>>
{
    constexpr static unsigned value = vector_traits<Expr>::vector_width;
};

template <typename Expr>
struct vector_width<Expr,detail::enable_if_t<is_unary_expression<Expr>::value>>
{
    constexpr static unsigned w1_ = vector_width<detail::decay_t<typename Expr::expr_type>>::value;
    constexpr static unsigned w2_ = vector_width<detail::decay_t<typename Expr::result_type>>::value;
    constexpr static unsigned value = (w1_ < w2_ ? w1_ : w2_);
};

template <typename Expr>
struct vector_width<Expr,detail::enable_if_t<is_binary_expression<Expr>::value>>
{
    constexpr static unsigned w1_ = vector_width<detail::decay_t<typename Expr::first_type>>::value;
    constexpr static unsigned w2_ = vector_width<detail::decay_t<typename Expr::second_type>>::value;
    constexpr static unsigned w3_ = vector_width<detail::decay_t<typename Expr::result_type>>::value;
    constexpr static unsigned value = (w3_ < (w1_ < w2_ ? w1_ : w2_) ? w3_ : (w1_ < w2_ ? w1_ : w2_));
};

/*
 * Increment (go to the next element) or decrement (return to the first element)
 * in a given dimension of the array subexpression.
 *
 * For broadcast dimensions this is a no-op.
 */
template <typename T, typename... Dims>
void increment(array_expr<T, Dims...>&, const bcast_dim&) {}

template <typename T, typename... Dims>
void increment(array_expr<T, Dims...>& expr, const slice_dim& dim)
{
    expr.data += dim.stride;
}

template <typename T, typename... Dims>
void decrement(array_expr<T, Dims...>&, const bcast_dim&) {}

template <typename T, typename... Dims>
void decrement(array_expr<T, Dims...>& expr, const slice_dim& dim)
{
    expr.data -= dim.len*dim.stride;
}

/*
 * Dim is one of the NDim dimensions of the array being assigned to. Since
 * the number of dimension in the subexpression (sizeof...(Dims)) can be
 * smaller, ignore increments and decrements for the first NDim-sizeof...(Dims)
 * dimensions.
 */
template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim < NDim-sizeof...(Dims))>
increment_array(array_expr<T, Dims...>&) {}

template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim < NDim-sizeof...(Dims))>
decrement_array(array_expr<T, Dims...>&) {}

/*
 * For the remaining sizeof...(Dims) dimensions, subtract NDim-sizeof...(Dims)
 * to get the proper dimension number in the subexpression.
 */
template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim >= NDim-sizeof...(Dims))>
increment_array(array_expr<T, Dims...>& expr)
{
    increment(expr, std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
detail::enable_if_t<(Dim >= NDim-sizeof...(Dims))>
decrement_array(array_expr<T, Dims...>& expr)
{
    decrement(expr, std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
void increment(array_expr<T, Dims...>& expr)
{
    increment_array<NDim, Dim, T, Dims...>(expr);
}

template <unsigned NDim, unsigned Dim, typename T, typename... Dims>
void decrement(array_expr<T, Dims...>& expr)
{
    decrement_array<NDim, Dim, T, Dims...>(expr);
}

/*
 * Lastly, for scalars do nothing since they are implicitly broadcast (this
 * only happens when assigning directly to a scalar as any other scalars are
 * absorbed into expression nodes).
 */
template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_scalar<Expr>::value>
increment(const Expr&) {}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_scalar<Expr>::value>
decrement(const Expr&) {}

/*
 * For binary and unary subexpressions, increment/decrement their children.
 */
template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_binary_expression<Expr>::value>
increment(Expr& expr)
{
    increment<NDim, Dim>(expr.first);
    increment<NDim, Dim>(expr.second);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_binary_expression<Expr>::value>
decrement(Expr& expr)
{
    decrement<NDim, Dim>(expr.first);
    decrement<NDim, Dim>(expr.second);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_unary_expression<Expr>::value>
increment(Expr& expr)
{
    increment<NDim, Dim>(expr.expr);
}

template <unsigned NDim, unsigned Dim, typename Expr>
detail::enable_if_t<is_unary_expression<Expr>::value>
decrement(Expr& expr)
{
    decrement<NDim, Dim>(expr.expr);
}

template <typename T, typename U>
struct assign_expr_value
{
    void operator()(T& lhs, const U& rhs) const
    {
        lhs = rhs;
    }
};

template <typename T, typename U>
struct assign_expr_value<T, std::complex<U>>
{
    void operator()(T& lhs, const std::complex<U>& rhs) const
    {
        lhs = rhs.real();
    }
};

template <typename T, typename U>
struct assign_expr_value<std::complex<T>, std::complex<U>>
{
    void operator()(std::complex<T>& lhs, const std::complex<U>& rhs) const
    {
        lhs.real(rhs.real());
        lhs.imag(rhs.imag());
    }
};

template <unsigned NDim, unsigned Dim=1>
struct assign_expr_loop;

template <unsigned NDim>
struct assign_expr_loop<NDim, NDim>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_value<detail::decay_t<typename expr_result_type<LHS>::type>,
                          detail::decay_t<typename expr_result_type<RHS>::type>> assign;

        for (len_type i = 0;i < len[NDim-1];i++)
        {
            assign(eval(lhs), eval(rhs));

            increment<NDim, NDim-1>(lhs);
            increment<NDim, NDim-1>(rhs);
        }

        decrement<NDim, NDim-1>(lhs);
        decrement<NDim, NDim-1>(rhs);
    }
};

template <unsigned NDim, unsigned Dim>
struct assign_expr_loop
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_loop<NDim, Dim+1> next_loop;

        for (len_type i = 0;i < len[Dim-1];i++)
        {
            next_loop(lhs, rhs, len);

            increment<NDim, Dim-1>(lhs);
            increment<NDim, Dim-1>(rhs);
        }

        decrement<NDim, Dim-1>(lhs);
        decrement<NDim, Dim-1>(rhs);
    }
};

template <unsigned NDim, unsigned Dim, unsigned Width, bool Aligned>
struct assign_expr_inner_loop_vec
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        typedef detail::decay_t<typename expr_result_type<LHS>::type> T;
        typedef detail::decay_t<typename expr_result_type<RHS>::type> U;

        assign_expr_value<T,U> assign;

        len_type misalignment = ((intptr_t)lhs.data %
                                 vector_traits<T>::alignment) / sizeof(T);

        len_type peel = Aligned ? 0 :
                        std::min(len[Dim], vector_traits<T>::vector_width - misalignment);

        // Number of elements in the final incomplete vector
        len_type remainder = (len[Dim]-peel) %
                             vector_traits<T>::vector_width;

        // Number of vectors or partial vectors (if Width is less than
        // the natural vector width for the output type)
        len_type nvector = (len[Dim]-peel-remainder) / Width;

        len_type i = 0;

        for (len_type j = 0;j < peel;j++, i++)
        {
            assign(eval_at<NDim, Dim>(lhs, i), eval_at<NDim, Dim>(rhs, i));
        }

        for (len_type j = 0;j < nvector;j++, i += Width)
        {
            lhs.template store_vec<NDim, Dim, Width, true>(i,
                vector_traits<U>::template convert<T>(
                    eval_vec<NDim, Dim, Width, Aligned>(rhs, i)));
        }

        for (len_type j = 0;j < remainder;j++, i++)
        {
            assign(eval_at<NDim, Dim>(lhs, i), eval_at<NDim, Dim>(rhs, i));
        }
    }
};

template <unsigned NDim, unsigned Dim, bool Aligned>
struct assign_expr_inner_loop_vec<NDim, Dim, 1, Aligned>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_value<detail::decay_t<typename expr_result_type<LHS>::type>,
                          detail::decay_t<typename expr_result_type<RHS>::type>> assign;

        for (len_type i = 0;i < len[Dim];i++)
        {
            assign(eval_at<NDim, Dim>(lhs, i), eval_at<NDim, Dim>(rhs, i));
        }
    }
};

template <unsigned NDim, unsigned Dim=1>
struct assign_expr_loop_vec_row_major;

template <unsigned NDim>
struct assign_expr_loop_vec_row_major<NDim, NDim>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        constexpr unsigned Width = (vector_width<LHS>::value < vector_width<RHS>::value ?
                                    vector_width<LHS>::value : vector_width<RHS>::value);
        constexpr bool Aligned = false;

        assign_expr_inner_loop_vec<NDim, NDim-1, Width, Aligned>()(lhs, rhs, len);
    }
};

template <unsigned NDim, unsigned Dim>
struct assign_expr_loop_vec_row_major
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_loop_vec_row_major<NDim, Dim+1> next_loop;

        for (len_type i = 0;i < len[Dim-1];i++)
        {
            next_loop(lhs, rhs, len);

            increment<NDim, Dim-1>(lhs);
            increment<NDim, Dim-1>(rhs);
        }

        decrement<NDim, Dim-1>(lhs);
        decrement<NDim, Dim-1>(rhs);
    }
};

template <unsigned NDim, unsigned Dim=NDim-1>
struct assign_expr_loop_vec_col_major;

template <unsigned NDim>
struct assign_expr_loop_vec_col_major<NDim, 0>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        constexpr unsigned Width = (vector_width<LHS>::value < vector_width<RHS>::value ?
                                    vector_width<LHS>::value : vector_width<RHS>::value);
        constexpr bool Aligned = false;

        assign_expr_inner_loop_vec<NDim, 0, Width, Aligned>()(lhs, rhs, len);
    }
};

template <unsigned NDim, unsigned Dim>
struct assign_expr_loop_vec_col_major
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_loop_vec_col_major<NDim, Dim-1> next_loop;

        for (len_type i = 0;i < len[Dim];i++)
        {
            next_loop(lhs, rhs, len);

            increment<NDim, Dim>(lhs);
            increment<NDim, Dim>(rhs);
        }

        decrement<NDim, Dim>(lhs);
        decrement<NDim, Dim>(rhs);
    }
};

template <typename Array, typename Expr>
detail::enable_if_t<(is_array_expression<detail::decay_t<Array>>::value ||
                     is_marray<detail::decay_t<Array>>::value) &&
                    is_expression_arg_or_scalar<detail::decay_t<Expr>>::value>
assign_expr(Array&& array_, Expr&& expr_)
{
    typedef typename expression_type<detail::decay_t<Array>>::type array_type;
    typedef typename expression_type<detail::decay_t<Expr>>::type expr_type;

    static_assert(expr_dimension<array_type>::value >=
                  expr_dimension<expr_type>::value,
                  "Dimensionality of the expression must not exceed that of the target");

    constexpr unsigned NDim = expr_dimension<array_type>::value;

    auto array = make_expression(std::forward<Array>(array_));
    auto expr = make_expression(std::forward<Expr>(expr_));

    auto len = get_array_lengths(array);
    MARRAY_ASSERT(check_expr_lengths(expr, len));

    if (is_contiguous<NDim, NDim-1>(array) &&
        is_contiguous<NDim, NDim-1>(expr))
    {
        assign_expr_loop_vec_row_major<NDim>()(array, expr, len);
    }
    else if (is_contiguous<NDim, 0>(array) &&
             is_contiguous<NDim, 0>(expr))
    {
        assign_expr_loop_vec_col_major<NDim>()(array, expr, len);
    }
    else
    {
        assign_expr_loop<NDim>()(array, expr, len);
    }
}

}

#endif
