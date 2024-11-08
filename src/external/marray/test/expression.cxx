#include "catch_amalgamated.hpp"
#include "marray.hpp"
#include "expression.hpp"

#include <typeinfo>
#include <complex>

using namespace std;
using namespace MArray;
using Catch::Matchers::WithinULP;

template <typename... T> struct types;

using all_types = std::tuple<float,double,std::complex<float>,std::complex<double>,int8_t,int16_t,int32_t,int64_t,uint8_t,uint16_t,uint32_t,uint64_t>;


template <typename... T>
struct concat;

template <>
struct concat<> { using type = std::tuple<>; };

template <typename... T>
struct concat<std::tuple<T...>> { using type = std::tuple<T...>; };

template <typename... T, typename... U, typename... V>
struct concat<std::tuple<T...>, std::tuple<U...>, V...> { using type = typename concat<std::tuple<T..., U...>, V...>::type; };

template <typename T, typename U>
struct pairs;

template <typename... T, typename U>
struct pairs<std::tuple<T...>, U> { using type = std::tuple<std::pair<T,U>...>; };

template <typename T, typename U>
struct pair_types;

template <typename... T, typename... U>
struct pair_types<std::tuple<T...>, std::tuple<U...>>
{
    using type = typename concat<typename pairs<std::tuple<T...>, U>::type...>::type;
};

using all_pair_types = typename pair_types<all_types, all_types>::type;

#define VECTOR_INITIALIZER { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}

/*
 * These are handled outside of the expression framework now.
 *

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_assign", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(a[i] == T(i%32));
        CHECK(b[i] == U(i%32));
    }
}

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_set", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;

    U data2[128] = {0};

    marray_view<U,1> b({128}, data2);

    b = T(3);

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(b[i] == U(3));
    }
}

*/

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_add", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;
    typedef decltype(std::declval<operators::plus>()(std::declval<T>(),
                                                     std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a+a;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(a[i] == T(i%32));
        CHECK(b[i] == U(2*(i%32)));
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c+d;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(c[i] == T(i%32));
        CHECK(d[i] == U(i%32));
        CHECK(e[i] == V(2*(i%32)));
    }
}

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_sub", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;
    typedef decltype(std::declval<operators::minus>()(std::declval<T>(),
                                                      std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a-a;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(a[i] == T(i%32));
        CHECK(b[i] == U(0));
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c-d;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(c[i] == T(i%32));
        CHECK(d[i] == U(i%32));
        CHECK(e[i] == V(0));
    }
}

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_mul", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;
    typedef decltype(std::declval<operators::multiplies>()(std::declval<T>(),
                                                           std::declval<U>())) V;

    // avoid overflow in floating -> integral since it is undefined and
    // in integral*integral because signed overflow is undefined
    T data1[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a*a;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(a[i] == T(i%8));
        CHECK(b[i] == U((i%8)*(i%8)));
    }

    T data3[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};
    U data4[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c*d;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(c[i] == T(i%8));
        CHECK(d[i] == U(i%8));
        CHECK(e[i] == V((i%8)*(i%8)));
    }
}

TEMPLATE_LIST_TEST_CASE("expression_vector::vector_div", "", all_pair_types)
{
    typedef typename TestType::first_type T;
    typedef typename TestType::second_type U;
    typedef decltype(std::declval<operators::divides>()(std::declval<T>(),
                                                        std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);
    a[ 0] = T(1);
    a[32] = T(1);
    a[64] = T(1);
    a[96] = T(1);

    b = a/a;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(a[i] == T(std::max(1,i%32)));
        CHECK(b[i] == U(1));
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());
    c[ 0] = T(1);
    c[32] = T(1);
    c[64] = T(1);
    c[96] = T(1);
    d[ 0] = U(1);
    d[32] = U(1);
    d[64] = U(1);
    d[96] = U(1);

    e = c/d;

    for (int i = 0;i < 128;i++)
    {
        INFO("i = " << i);
        CHECK(c[i] == T(std::max(1,i%32)));
        CHECK(d[i] == U(std::max(1,i%32)));
        CHECK(e[i] == V(1));
    }
}

TEST_CASE("expression::assign")
{
    using namespace slice;

    double data1[12] = { 1, 2, 3, 4, 5, 6,
                         7, 8, 9,10,11,12};
    double data2[12] = {12,11,10, 9, 8, 7,
                         6, 5, 4, 3, 2, 1};
    double data3[6] = {-1,-1,-1,-1,-1,-1};

    marray<double,3> v1{2, 3, 2};
    marray_view<double,3> v2({2, 3, 2}, data1);
    marray<double,4> v3({4, 3, 2, 2}, 1.0);
    marray<double,3> v4({2, 3, 2}, 4.0);
    marray_view<double,3> v5({2, 3, 2}, data2);
    marray_view<double,2> v6({3, 2}, data3);
    marray<double,2> v7({3, 2}, 5.0);

    v1 = 2.0;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{2, 2, 2, 2, 2, 2,
                                                            2, 2, 2, 2, 2, 2});

    v1 = 3;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{3, 3, 3, 3, 3, 3,
                                                            3, 3, 3, 3, 3, 3});

    v1 = v2;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 1, 2, 3, 4, 5, 6,
                                                             7, 8, 9,10,11,12});

    v1 = v3[range(2)][all][1][all];
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 1, 1, 1, 1, 1, 1,
                                                             1, 1, 1, 1, 1, 1});

    v1 = v4;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 4, 4, 4, 4, 4, 4,
                                                             4, 4, 4, 4, 4, 4});

    v2 = 2.0;
    CHECK(*(array<double,12>*)v2.data() == array<double,12>{2, 2, 2, 2, 2, 2,
                                                            2, 2, 2, 2, 2, 2});

    v2 = 3;
    CHECK(*(array<double,12>*)v2.data() == array<double,12>{3, 3, 3, 3, 3, 3,
                                                            3, 3, 3, 3, 3, 3});

    v2 = v5;
    CHECK(*(array<double,12>*)v2.data() == array<double,12>{12,11,10, 9, 8, 7,
                                                             6, 5, 4, 3, 2, 1});

    v2 = v3[range(2)][all][1][all];
    CHECK(*(array<double,12>*)v2.data() == array<double,12>{ 1, 1, 1, 1, 1, 1,
                                                             1, 1, 1, 1, 1, 1});

    v2 = v4;
    CHECK(*(array<double,12>*)v2.data() == array<double,12>{ 4, 4, 4, 4, 4, 4,
                                                             4, 4, 4, 4, 4, 4});

    v1[1][all][all] = 2.0;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{4, 4, 4, 4, 4, 4,
                                                            2, 2, 2, 2, 2, 2});

    v1[1][all][all] = 3;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{4, 4, 4, 4, 4, 4,
                                                            3, 3, 3, 3, 3, 3});

    v1[1][all][all] = v6;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 4, 4, 4, 4, 4, 4,
                                                            -1,-1,-1,-1,-1,-1});

    v1[1][all][all] = v3[1][all][1][all];
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 4, 4, 4, 4, 4, 4,
                                                             1, 1, 1, 1, 1, 1});

    v1[1][all][all] = v7;
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 4, 4, 4, 4, 4, 4,
                                                             5, 5, 5, 5, 5, 5});
}

TEST_CASE("expression::bcast")
{
    using namespace slice;

    double data[3] = {1, 2, 3};

    marray<double,3> v1{3, 2, 3};
    marray_view<double,1> v2({3}, data);

    v1 = v2;
    CHECK(*(array<double,18>*)v1.data() == array<double,18>{1, 2, 3, 1, 2, 3,
                                                            1, 2, 3, 1, 2, 3,
                                                            1, 2, 3, 1, 2, 3});

    v1 = 0;
    CHECK(*(array<double,18>*)v1.data() == array<double,18>{0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0,
                                                            0, 0, 0, 0, 0, 0});

    v1 = v2[bcast][bcast];
    CHECK(*(array<double,18>*)v1.data() == array<double,18>{1, 2, 3, 1, 2, 3,
                                                            1, 2, 3, 1, 2, 3,
                                                            1, 2, 3, 1, 2, 3});

    v1 = v2[all][bcast][bcast];
    CHECK(*(array<double,18>*)v1.data() == array<double,18>{1, 1, 1, 1, 1, 1,
                                                            2, 2, 2, 2, 2, 2,
                                                            3, 3, 3, 3, 3, 3});
}

TEST_CASE("expression::add")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 + v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{4, 4, 4});

    v1 = v2 + 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{2, 3, 4});

    v1 = 2.0 + v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{5, 4, 3});

    v1 += v2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{6, 6, 6});

    v1 += 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{7, 7, 7});
}

TEST_CASE("expression::sub")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 - v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{-2, 0, 2});

    v1 = v2 - 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{0, 1, 2});

    v1 = 2.0 - v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{-1, 0, 1});

    v1 -= v2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{-2, -2, -2});

    v1 -= 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{-3, -3, -3});
}

TEST_CASE("expression::mul")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 * v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{3, 4, 3});

    v1 = v2 * 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1, 2, 3});

    v1 = 2.0 * v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{6, 4, 2});

    v1 *= v2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{6, 8, 6});

    v1 *= 2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{12, 16, 12});
}

TEST_CASE("expression::div")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 / v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1.0/3, 1, 3});

    v1 = v2 / 1;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1, 2, 3});

    v1 = 2.0 / v3;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{2.0/3, 1, 2});

    v1 /= v2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{2.0/3, 0.5, 2.0/3});

    v1 /= 2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1.0/3, 0.25, 1.0/3});
}

TEST_CASE("expression::pow")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = pow(v2, v3);
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1, 4, 3});

    v1 = pow(v2, 2);
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{1, 4, 9});

    v1 = pow(2.0, v3);
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{8, 4, 2});
}

TEST_CASE("expression::negate")
{
    double data1[3] = {1, 2, 3};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = -v2;
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{-1, -2, -3});
}

TEST_CASE("expression::exp")
{
    double data1[3] = {1, 2, 3};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = exp(v2);
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{exp(1), exp(2), exp(3)});
}

TEST_CASE("expression::sqrt")
{
    double data1[3] = {4, 9, 16};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = sqrt(v2);
    CHECK(*(array<double,3>*)v1.data() == array<double,3>{2, 3, 4});
}

TEST_CASE("expression::compound")
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};
    double data3[3] = {4, 7, 2};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);
    marray_view<double,1> v4({3}, data3);

    v1 = (pow(v2, 2) * v3 + 1)/4 + sqrt(v4);
    CHECK_THAT(v1[0], WithinULP(3.0, 4));
    CHECK_THAT(v1[1], WithinULP(9.0/4 + sqrt(7), 4));
    CHECK_THAT(v1[2], WithinULP(5.0/2 + sqrt(2), 4));
}

TEST_CASE("expression::mixed_rank")
{
    double data1[12] = {1, 2, 3, 4, 5, 6,
                        7, 8, 9,10,11,12};
    double data2[6] = {1, 2, 3, 4, 5, 6};
    double data3[3] = {3, 2, 1};

    marray<double,3> v1{2, 2, 3};
    marray_view<double,3> v2({2, 2, 3}, data1);
    marray_view<double,2> v3({2, 3}, data2);
    marray_view<double,1> v4({3}, data3);

    v1 = v2 * 2 + v3 / v4;
    CHECK_THAT(v1.data()[ 0], WithinULP( 2 + 1.0/3, 4));
    CHECK_THAT(v1.data()[ 1], WithinULP( 4 + 2.0/2, 4));
    CHECK_THAT(v1.data()[ 2], WithinULP( 6 + 3.0/1, 4));
    CHECK_THAT(v1.data()[ 3], WithinULP( 8 + 4.0/3, 4));
    CHECK_THAT(v1.data()[ 4], WithinULP(10 + 5.0/2, 4));
    CHECK_THAT(v1.data()[ 5], WithinULP(12 + 6.0/1, 4));
    CHECK_THAT(v1.data()[ 6], WithinULP(14 + 1.0/3, 4));
    CHECK_THAT(v1.data()[ 7], WithinULP(16 + 2.0/2, 4));
    CHECK_THAT(v1.data()[ 8], WithinULP(18 + 3.0/1, 4));
    CHECK_THAT(v1.data()[ 9], WithinULP(20 + 4.0/3, 4));
    CHECK_THAT(v1.data()[10], WithinULP(22 + 5.0/2, 4));
    CHECK_THAT(v1.data()[11], WithinULP(24 + 6.0/1, 4));
}
