#include "indexed/indexed_marray.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

#define CHECK_INDEXED_VARRAY_RESET(v) \
    CHECK(v.dimension() == 0u); \
    CHECK(v.dense_dimension() == 0u); \
    CHECK(v.indexed_dimension() == 0u); \
    CHECK(v.num_indices() == 1u); \
    CHECK(v.data().size() == 0u);

#define CHECK_INDEXED_VARRAY(v,value,...) \
    CHECK(v.dimension() == 4u); \
    CHECK(v.dense_dimension() == 2u); \
    CHECK(v.indexed_dimension() == 2u); \
    CHECK(v.lengths() == len_vector{4, 2, 5, 4}); \
    CHECK(v.dense_lengths() == len_vector{4, 2}); \
    CHECK(v.indexed_lengths() == len_vector{5, 4}); \
    CHECK(v.dense_strides() == stride_vector __VA_ARGS__); \
    CHECK(v.num_indices() == 3u); \
    CHECK(v.indices() == matrix<len_type>{{0, 0}, {2, 1}, {4, 3}}); \
    CHECK(v.data(0)[0] == value);

TEST_CASE("indexed_varray::constructor")
{
    indexed_marray<double> v1;
    CHECK_INDEXED_VARRAY_RESET(v1)

    indexed_marray<double> v2({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v2, 0.0, {2, 1})

    indexed_marray<double> v3(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v3, 0.0, {2, 1})

    indexed_marray<double> v21({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v21, 1.0, {2, 1})

    indexed_marray<double> v31(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v31, 1.0, {2, 1})

    indexed_marray<double> v4({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v4, 1.0, {1, 4})

    indexed_marray<double> v41({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v41, 0.0, {1, 4})

    indexed_marray<double> v5(v21.view());
    CHECK_INDEXED_VARRAY(v5, 1.0, {2, 1})

    indexed_marray<double> v52(v21.cview());
    CHECK_INDEXED_VARRAY(v52, 1.0, {2, 1})

    indexed_marray<double> v51(v21);
    CHECK_INDEXED_VARRAY(v51, 1.0, {2, 1})

    indexed_marray<double> v6(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}));
    CHECK_INDEXED_VARRAY(v6, 0.0, {2, 1})

    indexed_marray<double> v61(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR));
    CHECK_INDEXED_VARRAY(v61, 0.0, {1, 4})

    indexed_marray<double> v62(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}), COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v62, 0.0, {1, 4})
}

TEST_CASE("indexed_varray::reset")
{
    indexed_marray<double> v2({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);

    indexed_marray<double> v1;
    CHECK_INDEXED_VARRAY_RESET(v1)

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 1.0, {1, 4})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset(v2.view());
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(v2.cview());
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(v2);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}));
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR));
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}), COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset();
    CHECK_INDEXED_VARRAY_RESET(v1)
}

TEST_CASE("indexed_varray::view")
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);

    auto v2 = v1.cview();
    CHECK_INDEXED_VARRAY(v2, 1.0, {2, 1});

    auto v3 = v1.view();
    CHECK_INDEXED_VARRAY(v3, 1.0, {2, 1});

    auto v4 = std::as_const(v1).view();
    CHECK_INDEXED_VARRAY(v4, 1.0, {2, 1});
}

TEST_CASE("indexed_varray::access")
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});

    auto v2 = v1[0];
    CHECK(v2.lengths() == len_vector{4, 2});
    CHECK(v2.strides() == stride_vector{2, 1});
    CHECK(v2.data() == v1.data(0));

    auto v3 = std::as_const(v1)[2];
    CHECK(v3.lengths() == len_vector{4, 2});
    CHECK(v3.strides() == stride_vector{2, 1});
    CHECK(v3.data() == v1.data(2));
}

TEST_CASE("indexed_varray::index_iteration")
{
    int indices[3][2] = {{0, 0}, {2, 1}, {4, 3}};
    array<int,3> visited;

    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    const indexed_marray<double> v2(v1);

    visited = {};
    v1.for_each_index(
    [&](const marray_view<double>& v, const index_vector& idx)
    {
        CHECK(idx.size() == 2u);
        len_type i = idx[0];
        len_type j = idx[1];
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                CHECK(v.data() == v1.data(m));
                found = true;
                visited[m]++;
            }
        }
        CHECK(found);
        CHECK(v.lengths() == len_vector{4, 2});
        CHECK(v.strides() == stride_vector{2, 1});
    });

    for (len_type i = 0;i < 3;i++)
    {
        CHECK(1 == visited[i]);
    }

    visited = {};
    v2.for_each_index(
    [&](const marray_view<const double>& v, const index_vector& idx)
    {
        CHECK(idx.size() == 2u);
        len_type i = idx[0];
        len_type j = idx[1];
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                CHECK(v.data() == v2.data(m));
                found = true;
                visited[m]++;
            }
        }
        CHECK(found);
        CHECK(v.lengths() == len_vector{4, 2});
        CHECK(v.strides() == stride_vector{2, 1});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    visited = {};
    v1.for_each_index<2,2>(
    [&](const marray_view<double,2>& v, len_type i, len_type j)
    {
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                CHECK(v.data() == v1.data(m));
                found = true;
                visited[m]++;
            }
        }
        CHECK(found);
        CHECK(v.lengths() == array<len_type,2>{4, 2});
        CHECK(v.strides() == array<stride_type,2>{2, 1});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    visited = {};
    v2.for_each_index<2,2>(
    [&](const marray_view<const double,2>& v, len_type i, len_type j)
    {
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                CHECK(v.data() == v2.data(m));
                found = true;
                visited[m]++;
            }
        }
        CHECK(found);
        CHECK(v.lengths() == array<len_type,2>{4, 2});
        CHECK(v.strides() == array<stride_type,2>{2, 1});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);
}

TEST_CASE("indexed_varray::element_iteration")
{
    int indices[3][2] = {{0, 0}, {2, 1}, {4, 3}};
    array<array<array<int,3>,2>,4> visited;

    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    const indexed_marray<double> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, const len_vector& idx)
    {
        CHECK(idx.size() == 4u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                CHECK(&v == &v1[m](i, j));
                found = true;
                visited[i][j][m]++;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 2;j++)
    for (len_type k = 0;k < 3;k++)
        CHECK(visited[i][j][k]);

    visited = {};
    v2.for_each_element(
    [&](const double& v, const len_vector& idx)
    {
        CHECK(idx.size() == 4u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                CHECK(&v == &v2[m](i, j));
                found = true;
                visited[i][j][m]++;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 2;j++)
    for (len_type k = 0;k < 3;k++)
        CHECK(visited[i][j][k]);

    visited = {};
    v1.for_each_element<2,2>(
    [&](double& v, len_type i, len_type j, len_type k, len_type l)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                CHECK(&v == &v1[m](i, j));
                found = true;
                visited[i][j][m]++;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 2;j++)
    for (len_type k = 0;k < 3;k++)
        CHECK(visited[i][j][k]);

    visited = {};
    v2.for_each_element<2,2>(
    [&](const double& v, len_type i, len_type j, len_type k, len_type l)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                CHECK(&v == &v2[m](i, j));
                found = true;
                visited[i][j][m]++;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 2;j++)
    for (len_type k = 0;k < 3;k++)
        CHECK(visited[i][j][k]);
}

TEST_CASE("indexed_varray::swap")
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    indexed_marray<double> v2({4, 5, 7}, {{0, 4}, {2, 2}, {4, 1}, {1, 1}});

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    CHECK(v2.dimension() == 4u);
    CHECK(v2.dense_dimension() == 2u);
    CHECK(v2.indexed_dimension() == 2u);
    CHECK(v2.lengths() == len_vector{4, 2, 5, 4});
    CHECK(v2.dense_lengths() == len_vector{4, 2});
    CHECK(v2.indexed_lengths() == len_vector{5, 4});
    CHECK(v2.dense_strides() == stride_vector{2, 1});
    CHECK(v2.num_indices() == 3u);
    CHECK(v2.indices() == matrix<len_type>{{0, 0}, {2, 1}, {4, 3}});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.dense_dimension() == 1u);
    CHECK(v1.indexed_dimension() == 2u);
    CHECK(v1.lengths() == len_vector{4, 5, 7});
    CHECK(v1.dense_lengths() == len_vector{4});
    CHECK(v1.indexed_lengths() == len_vector{5, 7});
    CHECK(v1.dense_strides() == stride_vector{1});
    CHECK(v1.num_indices() == 4u);
    CHECK(v1.indices() == matrix<len_type>{{0, 4}, {2, 2}, {4, 1}, {1, 1}});

    swap(v2, v1);

    CHECK(v1.dimension() == 4u);
    CHECK(v1.dense_dimension() == 2u);
    CHECK(v1.indexed_dimension() == 2u);
    CHECK(v1.lengths() == len_vector{4, 2, 5, 4});
    CHECK(v1.dense_lengths() == len_vector{4, 2});
    CHECK(v1.indexed_lengths() == len_vector{5, 4});
    CHECK(v1.dense_strides() == stride_vector{2, 1});
    CHECK(v1.num_indices() == 3u);
    CHECK(v1.indices() == matrix<len_type>{{0, 0}, {2, 1}, {4, 3}});
    CHECK(v2.dimension() == 3u);
    CHECK(v2.dense_dimension() == 1u);
    CHECK(v2.indexed_dimension() == 2u);
    CHECK(v2.lengths() == len_vector{4, 5, 7});
    CHECK(v2.dense_lengths() == len_vector{4});
    CHECK(v2.indexed_lengths() == len_vector{5, 7});
    CHECK(v2.dense_strides() == stride_vector{1});
    CHECK(v2.num_indices() == 4u);
    CHECK(v2.indices() == matrix<len_type>{{0, 4}, {2, 2}, {4, 1}, {1, 1}});
}
