#ifndef MARRAY_EXPRESSION_HPP
#define MARRAY_EXPRESSION_HPP

#include "marray.hpp"
#include "detail/vector.hpp"
#include "fwd/expression_fwd.hpp"

namespace MArray
{

template <typename Expr, typename=void> struct expression_type;
template <typename Expr, typename=void> struct expr_dimension;

template <typename Expr> using expression_type_t = typename expression_type<Expr>::type;

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

    template <int I>
    using dim_type = typename std::tuple_element<I, std::tuple<Dims...>>::type;

    array_expr(T* data, const Dims&... dims)
    : data(data), dims{dims...} {}

    result_type eval() const
    {
        return *data;
    }

    template <int NDim, int Dim>
    std::enable_if_t<(Dim < NDim-sizeof...(Dims)),result_type>
    eval_at(len_type) const
    {
        return *data;
    }

    template <int NDim, int Dim>
    std::enable_if_t<(Dim >= NDim-sizeof...(Dims)),result_type>
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

    template <int NDim, int Dim, int Width, bool Aligned>
    std::enable_if_t<(Dim < NDim-sizeof...(Dims)),vector_type>
    eval_vec(len_type) const
    {
        return vec_traits::load1(data);
    }

    template <int NDim, int Dim, int Width, bool Aligned>
    std::enable_if_t<(Dim >= NDim-sizeof...(Dims)),vector_type>
    eval_vec(len_type i) const
    {
        return eval_vec<Width,Aligned>(i, std::get<Dim-(NDim-sizeof...(Dims))>(dims));
    }

    template <int Width, bool Aligned>
    vector_type eval_vec(len_type i, const slice_dim&) const
    {
        return vec_traits::template load<Width,Aligned>(data+i);
    }

    template <int Width, bool Aligned>
    vector_type eval_vec(len_type, const bcast_dim&) const
    {
        return vec_traits::load1(data);
    }

    template <int NDim, int Dim, int Width, bool Aligned>
    void store_vec(len_type i, vector_type v) const
    {
        vec_traits::template store<Width,Aligned>(v, data+i);
    }
};

template <typename Expr>
struct expr_result_type<Expr, std::enable_if_t<is_scalar<std::decay_t<Expr>>::value>>
{
    typedef std::decay_t<Expr> type;
};

template <typename Expr>
struct expr_result_type<Expr, std::enable_if_t<is_expression<std::decay_t<Expr>>::value>>
{
    typedef typename std::decay_t<Expr>::result_type type;
};

template <typename Expr>
std::enable_if_t<is_expression<std::decay_t<Expr>>::value,
                 expr_result_type_t<std::decay_t<Expr>>>
eval(Expr&& expr)
{
    return expr.eval();
}

template <typename Expr>
std::enable_if_t<is_scalar<std::decay_t<Expr>>::value,Expr>
eval(Expr&& expr)
{
    return expr;
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_expression<std::decay_t<Expr>>::value,
                 expr_result_type_t<std::decay_t<Expr>>>
eval_at(Expr&& expr, len_type i)
{
    return expr.template eval_at<NDim, Dim>(i);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_scalar<std::decay_t<Expr>>::value,Expr>
eval_at(Expr&& expr, len_type)
{
    return expr;
}

template <int NDim, int Dim, int Width, bool Aligned, typename Expr>
std::enable_if_t<is_expression<std::decay_t<Expr>>::value,
                    typename std::decay_t<Expr>::vector_type>
eval_vec(Expr&& expr, len_type i)
{
    return expr.template eval_vec<NDim, Dim, Width, Aligned>(i);
}

template <int NDim, int Dim, int Width, bool Aligned, typename Expr>
std::enable_if_t<is_scalar<std::decay_t<Expr>>::value,
                    typename vector_traits<std::decay_t<Expr>>::vector_type>
eval_vec(Expr&& expr, len_type)
{
    return vector_traits<std::decay_t<Expr>>::set1(expr);
}

namespace operators
{

template <typename T>
using vector_type = typename vector_traits<T>::vector_type;

template <typename T, typename U, typename=void>
struct binary_result_type;

template <typename T, typename U>
struct binary_result_type<T, U,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::common_type_t<T, U> type;
};

template <typename T, typename U>
struct binary_result_type<T, std::complex<U>,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef T type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, U,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, std::complex<U>,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<typename std::common_type<T, U>::type> type;
};

template <typename T, typename U>
struct binary_result_type<std::complex<T>, U,
    std::enable_if_t<std::is_floating_point<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef std::complex<T> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    std::enable_if_t<std::is_integral<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef U type;
};

template <typename T, typename U>
struct binary_result_type<T, std::complex<U>,
    std::enable_if_t<std::is_integral<T>::value &&
                        std::is_floating_point<U>::value>>
{
    typedef std::complex<U> type;
};

template <typename T, typename U>
struct binary_result_type<T, U,
    std::enable_if_t<std::is_integral<T>::value &&
                        std::is_integral<U>::value>>
{
    typedef typename std::common_type<T,U>::type type;
};

template <typename T, typename U>
using binary_result_type_t = typename binary_result_type<T, U>::type;

struct plus
{
    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    std::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                     binary_result_type_t<T, U>>
    {
        typedef binary_result_type_t<T, U> V;
        return V(a)+V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    std::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
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
    std::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        binary_result_type_t<T, U>>
    {
        typedef binary_result_type_t<T, U> V;
        return V(a)-V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    std::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
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
    std::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        binary_result_type_t<T, U>>
    {
        typedef binary_result_type_t<T, U> V;
        return V(a)*V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    std::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
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
    std::enable_if_t<is_scalar<T>::value && is_scalar<U>::value,
                        binary_result_type_t<T, U>>
    {
        typedef binary_result_type_t<T, U> V;
        return V(a)/V(b);
    }

    template <typename T, typename U>
    auto operator()(const T& a, const U& b) const ->
    std::enable_if_t<!is_scalar<T>::value || !is_scalar<U>::value,
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

struct log
{
    template <typename T>
    auto operator()(const T& a) const -> decltype(std::log(a))
    {
        return std::log(a);
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a) const
    {
        return vector_traits<T>::log(a);
    }
};

struct abs
{
    template <typename T>
    auto operator()(const T& a) const -> decltype(std::abs(a))
    {
        return std::abs(a);
    }

    template <typename T>
    vector_type<T> vec(vector_type<T> a) const
    {
        return vector_traits<T>::abs(a);
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
    typedef std::decay_t<expr_result_type_t<LHS>> first_result_type;
    typedef std::decay_t<expr_result_type_t<RHS>> second_result_type;
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

    template <int NDim, int Dim>
    result_type eval_at(len_type i) const
    {
        return op(MArray::eval_at<NDim, Dim>(first, i),
                  MArray::eval_at<NDim, Dim>(second, i));
    }

    template <int NDim, int Dim, int Width, bool Aligned>
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
    typedef std::decay_t<expr_result_type_t<Expr>> input_result_type;
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

    template <int NDim, int Dim>
    result_type eval_at(len_type i) const
    {
        return op(MArray::eval_at<NDim, Dim>(expr, i));
    }

    template <int NDim, int Dim, int Width, bool Aligned>
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
using log_expr = unary_expr<Expr, operators::log>;

template <typename Expr>
using abs_expr = unary_expr<Expr, operators::abs>;

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
    std::enable_if_t<is_expression<typename Expr::expr_type>::value>>
    : std::true_type {};

template <typename Expr, typename>
struct is_binary_expression : std::false_type {};

template <typename Expr>
struct is_binary_expression<Expr,
    std::enable_if_t<is_expression<typename Expr::first_type>::value ||
                        is_expression<typename Expr::second_type>::value>>
    : std::true_type {};

template <typename Array, typename Dim>
struct array_expr_type_helper2;

template <typename T, typename... Dims, typename Dim>
struct array_expr_type_helper2<array_expr<T, Dims...>, Dim>
{
    typedef array_expr<T, Dims..., Dim> type;
};

template <typename T, int NDim>
struct array_expr_helper;

template <typename T>
struct array_expr_helper<T, 0>
{
    typedef array_expr<T> type;
};

template <typename T, int NDim>
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

template <typename Array, typename>
struct is_marray
{
    template <typename T, int NDim, typename Derived, bool Owner>
    static std::bool_constant<NDim != DYNAMIC> check(const marray_base<T, NDim, Derived, Owner>*);

    template <typename T, int NDim, int NIndexed, typename... Dims>
    static std::true_type check(const marray_slice<T, NDim, NIndexed, Dims...>*);

    static std::false_type check(...);

    static constexpr bool value = decltype(check((Array*)nullptr))::value;
};

template <typename Expr>
struct expression_type<Expr, std::enable_if_t<is_marray<Expr>::value>>
{
    template <typename T, int NDim, int NIndexed, typename... Dims>
    static typename slice_array_expr_helper<typename array_expr_helper<T, NDim-NIndexed>::type, Dims...>::type check(const marray_slice<T, NDim, NIndexed, Dims...>*);

    template <typename T, int NDim, typename Derived>
    static typename array_expr_helper<T, NDim>::type check(const marray_base<T, NDim, Derived, false>*);

    template <typename T, int NDim, typename Derived>
    static typename array_expr_helper<const T, NDim>::type check(const marray_base<T, NDim, Derived, true>*);

    template <typename T, int NDim, typename Derived>
    static typename array_expr_helper<T, NDim>::type check(marray_base<T, NDim, Derived, true>*);

    static void check(...);

    typedef decltype(check((Expr*)nullptr)) type;
};

template <typename Expr>
struct expression_type<Expr, std::enable_if_t<is_expression<std::decay_t<Expr>>::value ||
                                                 is_scalar<std::decay_t<Expr>>::value>>
{
    typedef std::decay_t<Expr> type;
};

template <typename T, int NDim, int NIndexed, typename... Dims,
          int... I, int... J>
expression_type_t<marray_slice<T, NDim, NIndexed, Dims...>>
make_expression_helper(const marray_slice<T, NDim, NIndexed, Dims...>& x,
                       std::integer_sequence<int, I...>,
                       std::integer_sequence<int, J...>)
{
    return {x.data(), x.template dim<I>()...,
            slice_dim(x.base(NIndexed+J),
                      x.length(NIndexed+J),
                      0,
                      x.stride(NIndexed+J))...};
}

template <typename T, int NDim, typename Derived, bool Owner, int... I>
expression_type_t<const marray_base<T, NDim, Derived, Owner>>
make_expression_helper(const marray_base<T, NDim, Derived, Owner>& x,
                       std::integer_sequence<int, I...>)
{
    return {x.data(), slice_dim(x.base(I), x.length(I), 0, x.stride(I))...};
}

template <typename T, int NDim, typename Derived, bool Owner, int... I>
expression_type_t<marray_base<T, NDim, Derived, Owner>>
make_expression_helper(marray_base<T, NDim, Derived, Owner>& x,
                       std::integer_sequence<int, I...>)
{
    return {x.data(), slice_dim(x.base(I), x.length(I), 0, x.stride(I))...};
}

template <typename T, int NDim, int NIndexed, typename... Dims>
expression_type_t<marray_slice<T, NDim, NIndexed, Dims...>>
make_expression(const marray_slice<T, NDim, NIndexed, Dims...>& x)
{
    return make_expression_helper(x, std::make_integer_sequence<int, sizeof...(Dims)>(),
                                     std::make_integer_sequence<int, NDim-NIndexed>());
}

template <typename T, int NDim, typename Derived, bool Owner>
expression_type_t<const marray_base<T, NDim, Derived, Owner>>
make_expression(const marray_base<T, NDim, Derived, Owner>& x)
{
    return make_expression_helper(x, std::make_integer_sequence<int, NDim>());
}

template <typename T, int NDim, typename Derived, bool Owner>
expression_type_t<marray_base<T, NDim, Derived, Owner>>
make_expression(marray_base<T, NDim, Derived, Owner>& x)
{
    return make_expression_helper(x, std::make_integer_sequence<int, NDim>());
}

template <typename Expr>
std::enable_if_t<is_expression<std::decay_t<Expr>>::value ||
                    is_scalar<std::decay_t<Expr>>::value, Expr&&>
make_expression(Expr&& x)
{
    return std::forward<Expr>(x);
}

template <typename Expr>
struct is_expression_arg :
    std::integral_constant<bool, is_expression<std::decay_t<Expr>>::value ||
                                 is_marray<std::decay_t<Expr>>::value> {};

template <typename Expr>
struct is_expression_arg_or_scalar :
    std::integral_constant<bool, is_expression_arg<std::decay_t<Expr>>::value ||
                                 is_scalar<std::decay_t<Expr>>::value> {};

template <typename T, typename Expr, typename=void>
struct converted_scalar_type {};

template <typename T, typename Expr>
struct converted_scalar_type<T, Expr, std::enable_if_t<is_scalar<T>::value &&
                                                       is_expression_arg<Expr>::value>> :
    std::common_type<T, expr_result_type_t<expression_type_t<Expr>>> {};

template <typename T, typename Expr>
using converted_scalar_type_t = typename converted_scalar_type<T, Expr>::type;

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    add_expr<expression_type_t<const LHS>,
                             expression_type_t<const RHS>>>
operator+(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    add_expr<expression_type_t<const LHS>,
                             converted_scalar_type_t<RHS, LHS>>>
operator+(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<RHS, LHS> T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    add_expr<converted_scalar_type_t<LHS, RHS>,
                             expression_type_t<const RHS>>>
operator+(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<LHS, RHS> T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    sub_expr<expression_type_t<const LHS>,
                             expression_type_t<const RHS>>>
operator-(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    sub_expr<expression_type_t<const LHS>,
                             converted_scalar_type_t<RHS, LHS>>>
operator-(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<RHS, LHS> T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    sub_expr<converted_scalar_type_t<LHS, RHS>,
                             expression_type_t<const RHS>>>
operator-(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<LHS, RHS> T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    mul_expr<expression_type_t<const LHS>,
                             expression_type_t<const RHS>>>
operator*(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    mul_expr<expression_type_t<const LHS>,
                             converted_scalar_type_t<RHS, LHS>>>
operator*(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<RHS, LHS> T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    mul_expr<converted_scalar_type_t<LHS, RHS>,
                             expression_type_t<const RHS>>>
operator*(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<LHS, RHS> T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    div_expr<expression_type_t<const LHS>,
                             expression_type_t<const RHS>>>
operator/(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    div_expr<expression_type_t<const LHS>,
                             converted_scalar_type_t<RHS, LHS>>>
operator/(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<RHS, LHS> T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    div_expr<converted_scalar_type_t<LHS, RHS>,
                             expression_type_t<const RHS>>>
operator/(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<LHS, RHS> T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    pow_expr<expression_type_t<const LHS>,
                             expression_type_t<const RHS>>>
pow(const LHS& lhs, const RHS& rhs)
{
    return {make_expression(lhs), make_expression(rhs)};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_expression_arg<LHS>::value &&
                    is_scalar<RHS>::value,
                    pow_expr<expression_type_t<const LHS>,
                             converted_scalar_type_t<RHS, LHS>>>
pow(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<RHS, LHS> T;
    return {make_expression(lhs), (T)rhs};
}

template <typename LHS, typename RHS>
std::enable_if_t<is_scalar<LHS>::value &&
                    is_expression_arg<RHS>::value,
                    pow_expr<converted_scalar_type_t<LHS, RHS>,
                             expression_type_t<const RHS>>>
pow(const LHS& lhs, const RHS& rhs)
{
    typedef converted_scalar_type_t<LHS, RHS> T;
    return {(T)lhs, make_expression(rhs)};
}

template <typename Expr>
std::enable_if_t<is_expression_arg<Expr>::value,
                    negate_expr<expression_type_t<const Expr>>>
operator-(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename Expr>
std::enable_if_t<is_expression_arg<Expr>::value,
                    exp_expr<expression_type_t<const Expr>>>
exp(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename Expr>
std::enable_if_t<is_expression_arg<Expr>::value,
                    sqrt_expr<expression_type_t<const Expr>>>
sqrt(const Expr& expr)
{
    return {make_expression(expr)};
}

template <typename T, typename... Dims>
struct expr_dimension<array_expr<T, Dims...>>
    : std::integral_constant<int, sizeof...(Dims)> {};

template <typename Expr>
struct expr_dimension<Expr, std::enable_if_t<is_scalar<Expr>::value>>
    : std::integral_constant<int, 0> {};

template <typename Expr>
struct expr_dimension<Expr, std::enable_if_t<is_binary_expression<Expr>::value>>
    : std::conditional_t<(expr_dimension<typename Expr::first_type>::value <
                             expr_dimension<typename Expr::second_type>::value),
                             expr_dimension<typename Expr::second_type>,
                             expr_dimension<typename Expr::first_type>> {};

template <typename Expr>
struct expr_dimension<Expr, std::enable_if_t<is_unary_expression<Expr>::value>>
    : expr_dimension<typename Expr::expr_type> {};

template <typename Dim>
std::enable_if_t<std::is_same<Dim,bcast_dim>::value,len_type>
get_array_length(const Dim&)
{
    static_assert(!std::is_same<Dim,bcast_dim>::value,
                  "Broadcast dimensions cannot be written to");
    return 0;
}

template <typename Dim>
std::enable_if_t<std::is_same<Dim,slice_dim>::value,len_type>
get_array_length(const Dim& dim)
{
    return dim.len;
}

template <typename T, typename... Dims, int... I>
std::array<len_type, sizeof...(I)>
get_array_lengths_helper(const array_expr<T, Dims...>& array,
                         std::integer_sequence<int, I...>)
{
    return {get_array_length(std::get<I>(array.dims))...};
}

template <typename T, typename... Dims>
std::array<len_type, sizeof...(Dims)>
get_array_lengths(const array_expr<T, Dims...>& array)
{
    return get_array_lengths_helper(array, std::make_integer_sequence<int, sizeof...(Dims)>());
}

inline len_type get_expr_length(const bcast_dim&)
{
    return -1;
}

inline len_type get_expr_length(const slice_dim& dim)
{
    return dim.len;
}

template <int NDim, typename T, typename... Dims, int... I, int... J>
std::array<len_type, NDim>
get_expr_lengths_helper(const array_expr<T, Dims...>& array,
                        std::integer_sequence<int, I...>,
                        std::integer_sequence<int, J...>)
{
    return std::array<len_type, NDim>
        {(I-I-1)..., get_expr_length(std::get<J>(array.dims))...};
}

template <int NDim, typename T, typename... Dims>
std::array<len_type, NDim>
get_expr_lengths(const array_expr<T, Dims...>& array)
{
    return get_expr_lengths_helper<NDim>(array,
        std::make_integer_sequence<int, NDim-sizeof...(Dims)>(),
        std::make_integer_sequence<int, sizeof...(Dims)>());
}

template <int NDim, typename Expr>
std::enable_if_t<is_scalar<Expr>::value,std::array<len_type, NDim>>
get_expr_lengths(const Expr&)
{
    return get_expr_lengths_helper<NDim>(array_expr<Expr>{nullptr},
        std::make_integer_sequence<int, NDim>(),
        std::make_integer_sequence<int, 0>());
}

template <int NDim, typename Expr>
std::enable_if_t<is_binary_expression<Expr>::value,std::array<len_type, NDim>>
get_expr_lengths(const Expr& expr)
{
    auto len1 = get_expr_lengths<NDim>(expr.first);
    auto len2 = get_expr_lengths<NDim>(expr.second);
    for (auto i : range(NDim))
    {
        MARRAY_ASSERT(len1[i] == len2[i] || len1[i] == -1 || len2[i] == -1);
        len1[i] = std::max(len1[i],len2[i]);
    }
    return len1;
}

template <int NDim, typename Expr>
std::enable_if_t<is_unary_expression<Expr>::value,std::array<len_type, NDim>>
get_expr_lengths(const Expr& expr)
{
    return get_expr_lengths<NDim>(expr.expr);
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
template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim < NDim-sizeof...(Dims)), bool>
is_array_contiguous(const array_expr<T, Dims...>&)
{
    return true;
}

/*
 * For the remaining sizeof...(Dims) dimensions, subtract NDim-sizeof...(Dims)
 * to get the proper dimension number in the subexpression.
 */
template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim >= NDim-sizeof...(Dims)), bool>
is_array_contiguous(const array_expr<T, Dims...>& expr)
{
    return is_contiguous(std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <int NDim, int Dim, typename T, typename... Dims>
bool is_contiguous(const array_expr<T, Dims...>& expr)
{
    return is_array_contiguous<NDim, Dim, T, Dims...>(expr);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_scalar<Expr>::value, bool>
is_contiguous(const Expr&)
{
    return true;
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_binary_expression<Expr>::value, bool>
is_contiguous(const Expr& expr)
{
    return is_contiguous<NDim, Dim>(expr.first) &&
           is_contiguous<NDim, Dim>(expr.second);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_unary_expression<Expr>::value, bool>
is_contiguous(const Expr& expr)
{
    return is_contiguous<NDim, Dim>(expr.expr);
}

template <typename Expr, typename=void>
struct vector_width;

template <typename T, typename... Dims>
struct vector_width<array_expr<T, Dims...>>
{
    static constexpr int value = vector_traits<typename std::remove_cv<T>::type>::vector_width;
};

template <typename Expr>
struct vector_width<Expr,std::enable_if_t<is_scalar<Expr>::value>>
{
    static constexpr int value = vector_traits<Expr>::vector_width;
};

template <typename Expr>
struct vector_width<Expr,std::enable_if_t<is_unary_expression<Expr>::value>>
{
    static constexpr int w1_ = vector_width<std::decay_t<typename Expr::expr_type>>::value;
    static constexpr int w2_ = vector_width<std::decay_t<typename Expr::result_type>>::value;
    static constexpr int value = (w1_ < w2_ ? w1_ : w2_);
};

template <typename Expr>
struct vector_width<Expr,std::enable_if_t<is_binary_expression<Expr>::value>>
{
    static constexpr int w1_ = vector_width<std::decay_t<typename Expr::first_type>>::value;
    static constexpr int w2_ = vector_width<std::decay_t<typename Expr::second_type>>::value;
    static constexpr int w3_ = vector_width<std::decay_t<typename Expr::result_type>>::value;
    static constexpr int value = (w3_ < (w1_ < w2_ ? w1_ : w2_) ? w3_ : (w1_ < w2_ ? w1_ : w2_));
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
template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim < NDim-sizeof...(Dims))>
increment_array(array_expr<T, Dims...>&) {}

template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim < NDim-sizeof...(Dims))>
decrement_array(array_expr<T, Dims...>&) {}

/*
 * For the remaining sizeof...(Dims) dimensions, subtract NDim-sizeof...(Dims)
 * to get the proper dimension number in the subexpression.
 */
template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim >= NDim-sizeof...(Dims))>
increment_array(array_expr<T, Dims...>& expr)
{
    increment(expr, std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <int NDim, int Dim, typename T, typename... Dims>
std::enable_if_t<(Dim >= NDim-sizeof...(Dims))>
decrement_array(array_expr<T, Dims...>& expr)
{
    decrement(expr, std::get<Dim-(NDim-sizeof...(Dims))>(expr.dims));
}

template <int NDim, int Dim, typename T, typename... Dims>
void increment(array_expr<T, Dims...>& expr)
{
    increment_array<NDim, Dim, T, Dims...>(expr);
}

template <int NDim, int Dim, typename T, typename... Dims>
void decrement(array_expr<T, Dims...>& expr)
{
    decrement_array<NDim, Dim, T, Dims...>(expr);
}

/*
 * Lastly, for scalars do nothing since they are implicitly broadcast (this
 * only happens when assigning directly to a scalar as any other scalars are
 * absorbed into expression nodes).
 */
template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_scalar<Expr>::value>
increment(const Expr&) {}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_scalar<Expr>::value>
decrement(const Expr&) {}

/*
 * For binary and unary subexpressions, increment/decrement their children.
 */
template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_binary_expression<Expr>::value>
increment(Expr& expr)
{
    increment<NDim, Dim>(expr.first);
    increment<NDim, Dim>(expr.second);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_binary_expression<Expr>::value>
decrement(Expr& expr)
{
    decrement<NDim, Dim>(expr.first);
    decrement<NDim, Dim>(expr.second);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_unary_expression<Expr>::value>
increment(Expr& expr)
{
    increment<NDim, Dim>(expr.expr);
}

template <int NDim, int Dim, typename Expr>
std::enable_if_t<is_unary_expression<Expr>::value>
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

template <int NDim, int Dim=1>
struct assign_expr_loop;

template <int NDim>
struct assign_expr_loop<NDim, NDim>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_value<std::decay_t<expr_result_type_t<LHS>>,
                          std::decay_t<expr_result_type_t<RHS>>> assign;

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

template <int NDim, int Dim>
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

template <int NDim, int Dim, int Width, bool Aligned>
struct assign_expr_inner_loop_vec
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        typedef std::decay_t<expr_result_type_t<LHS>> T;
        typedef std::decay_t<expr_result_type_t<RHS>> U;

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

template <int NDim, int Dim, bool Aligned>
struct assign_expr_inner_loop_vec<NDim, Dim, 1, Aligned>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        assign_expr_value<std::decay_t<expr_result_type_t<LHS>>,
                          std::decay_t<expr_result_type_t<RHS>>> assign;

        for (len_type i = 0;i < len[Dim];i++)
        {
            assign(eval_at<NDim, Dim>(lhs, i), eval_at<NDim, Dim>(rhs, i));
        }
    }
};

template <int NDim, int Dim=1>
struct assign_expr_loop_vec_row_major;

template <int NDim>
struct assign_expr_loop_vec_row_major<NDim, NDim>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        constexpr int Width = (vector_width<LHS>::value < vector_width<RHS>::value ?
                               vector_width<LHS>::value : vector_width<RHS>::value);
        constexpr bool Aligned = false;

        assign_expr_inner_loop_vec<NDim, NDim-1, Width, Aligned>()(lhs, rhs, len);
    }
};

template <int NDim, int Dim>
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

template <int NDim, int Dim=NDim-1>
struct assign_expr_loop_vec_col_major;

template <int NDim>
struct assign_expr_loop_vec_col_major<NDim, 0>
{
    template <typename LHS, typename RHS>
    void operator()(LHS& lhs, RHS& rhs, const std::array<len_type, NDim>& len) const
    {
        constexpr int Width = (vector_width<LHS>::value < vector_width<RHS>::value ?
                               vector_width<LHS>::value : vector_width<RHS>::value);
        constexpr bool Aligned = false;

        assign_expr_inner_loop_vec<NDim, 0, Width, Aligned>()(lhs, rhs, len);
    }
};

template <int NDim, int Dim>
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
std::enable_if_t<(is_array_expression<std::decay_t<Array>>::value ||
                     is_marray<std::decay_t<Array>>::value) &&
                    is_expression_arg_or_scalar<std::decay_t<Expr>>::value>
assign_expr(Array&& array_, Expr&& expr_)
{
    typedef expression_type_t<std::decay_t<Array>> array_type;
    typedef expression_type_t<std::decay_t<Expr>> expr_type;

    static_assert(expr_dimension<array_type>::value >=
                  expr_dimension<expr_type>::value,
                  "Dimensionality of the expression must not exceed that of the target");

    constexpr int NDim = expr_dimension<array_type>::value;

    auto array = make_expression(std::forward<Array>(array_));
    auto expr = make_expression(std::forward<Expr>(expr_));

    auto len = get_array_lengths(array);
    auto check = get_expr_lengths<NDim>(expr);
    for (auto i : range(NDim))
        MARRAY_ASSERT(len[i] == check[i] || check[i] == -1);

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

template <typename Type, int NDim, typename Allocator>
template <typename Expression,typename>
marray<Type,NDim,Allocator>::marray(const Expression& other)
{
    typedef expression_type_t<std::decay_t<Expression>> expr_type;

    static_assert(NDim == expr_dimension<expr_type>::value,
                  "Dimensionality of the expression must equal that of the target");

    reset(get_expr_lengths<NDim>(other), uninitialized);
    assign_expr(*this, other);
}

}

#endif //MARRAY_EXPRESSION_HPP
