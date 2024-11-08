#include "indexed_dpd/indexed_dpd_marray.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

static dpd_layout layouts[6] =
{
    PREFIX_ROW_MAJOR,
    PREFIX_COLUMN_MAJOR,
    BLOCKED_ROW_MAJOR,
    BLOCKED_COLUMN_MAJOR,
    BALANCED_ROW_MAJOR,
    BALANCED_COLUMN_MAJOR,
};

static dim_vector perms[6] =
    {{3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}};

static irrep_vector irreps[8] =
    {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {1,1,1,0},
     {0,0,0,1}, {1,1,0,1}, {1,0,1,1}, {0,1,1,1}};
static len_vector lengths[8] =
    {{1,2,1,3}, {3,2,1,3}, {3,2,2,3}, {1,2,2,3},
     {3,2,1,4}, {1,2,1,4}, {1,2,2,4}, {3,2,2,4}};

static stride_vector strides[6][8] =
{
    {{42,11, 3, 1}, {42,11, 3, 1}, {42,10, 3, 1}, {42,10, 3, 1},
     {42,10, 4, 1}, {42,10, 4, 1}, {42,11, 4, 1}, {42,11, 4, 1}},
    {{ 1, 1, 8,24}, { 1, 3, 8,24}, { 1, 3, 8,24}, { 1, 1, 8,24},
     { 1, 3, 8,24}, { 1, 1, 8,24}, { 1, 1, 8,24}, { 1, 3, 8,24}},
    {{ 6, 3, 3, 1}, { 6, 3, 3, 1}, {12, 6, 3, 1}, {12, 6, 3, 1},
     { 8, 4, 4, 1}, { 8, 4, 4, 1}, {16, 8, 4, 1}, {16, 8, 4, 1}},
    {{ 1, 1, 2, 2}, { 1, 3, 6, 6}, { 1, 3, 6,12}, { 1, 1, 2, 4},
     { 1, 3, 6, 6}, { 1, 1, 2, 2}, { 1, 1, 2, 4}, { 1, 3, 6,12}},
    {{22,11, 3, 1}, {22,11, 3, 1}, {20,10, 3, 1}, {20,10, 3, 1},
     {20,10, 4, 1}, {20,10, 4, 1}, {22,11, 4, 1}, {22,11, 4, 1}},
    {{ 1, 1, 8, 8}, { 1, 3, 8, 8}, { 1, 3, 8,16}, { 1, 1, 8,16},
     { 1, 3, 8, 8}, { 1, 1, 8, 8}, { 1, 1, 8,16}, { 1, 3, 8,16}}
};

static stride_type offsets[6][8] =
{
     {126, 20,  4,152,  0,148,129, 23},
     {  0,  2,  8, 14, 72, 78, 80, 82},
     {162,144, 96,132,  0, 24, 80, 32},
     {  0, 42,108, 22,144, 34,  6, 60},
     { 80+66, 80   ,     0+4,  0+60+4,
        0   ,  0+60, 80+66+3, 80   +3},
     {  0   ,  0   +2, 88   , 88   +6,
       88+48, 88+48+6,  0+24,  0+24+2}
};

#define CHECK_INDEXED_DPD_VARRAY_RESET(v) \
    CHECK(v.dimension() == 0u); \
    CHECK(v.dense_dimension() == 0u); \
    CHECK(v.indexed_dimension() == 0u); \
    CHECK(v.num_indices() == 1u); \
    CHECK(v.permutation() == dim_vector{}); \
    CHECK(v.lengths() == matrix<len_type>{}); \
    CHECK(v.data().size() == 0u);

#define CHECK_INDEXED_DPD_VARRAY(v,j,value) \
    INFO("j = " << j); \
    CHECK(v.dimension() == 6u); \
    CHECK(v.dense_dimension() == 4u); \
    CHECK(v.indexed_dimension() == 2u); \
    CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}}); \
    CHECK(v.dense_lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}}); \
    CHECK(v.indexed_lengths() == len_vector{2, 5}); \
    CHECK(v.num_indices() == 3u); \
    CHECK(v.indexed_irreps() == irrep_vector{1, 1}); \
    CHECK(v.indices() == matrix<len_type>{{0, 0}, {1, 3}, {0, 3}}); \
    CHECK(v.data(0)[0] == value); \
    CHECK(v.irrep() == 1u); \
    CHECK(v.num_irreps() == 2u); \
    CHECK(v.permutation() == perms[j]); \
    \
    for (int m = 0;m < 3u;m++) \
    { \
        INFO("m = " << m); \
        { \
            auto vs = v[m](1,0,0,0); \
            CHECK(vs.data() == v.data(m) + offsets[j][0]); \
            for (int k = 0;k < 4;k++) \
            { \
                INFO("k = " << k); \
                CHECK(vs.length(k) == lengths[0][k]); \
                CHECK(vs.stride(k) == strides[j][0][k]); \
            } \
        } \
        \
        { \
            auto vs = v[m]({0,1,0,0}); \
            CHECK(vs.data() == v.data(m) + offsets[j][1]); \
            CHECK(vs.lengths() == lengths[1]); \
            CHECK(vs.strides() == strides[j][1]); \
        } \
        \
        for (int i = 2;i < 8;i++) \
        { \
            INFO("i = " << i); \
            auto vs = v[m](irreps[i]); \
            CHECK(vs.data() == v.data(m) + offsets[j][i]); \
            CHECK(vs.lengths() == lengths[i]); \
            CHECK(vs.strides() == strides[j][i]); \
        } \
    }

TEST_CASE("indexed_dpd_varray::constructor")
{
    indexed_dpd_marray<double> v1;
    CHECK_INDEXED_DPD_VARRAY_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        indexed_dpd_marray<double> v2(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                      {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v2, j, 0.0)

        indexed_dpd_marray<double> v21(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                       {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v21, j, 1.0)

        indexed_dpd_marray<double> v5(v21.view(), layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v5, j, 1.0)

        indexed_dpd_marray<double> v52(v21.cview(), layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v52, j, 1.0)

        indexed_dpd_marray<double> v51(v21);
        CHECK_INDEXED_DPD_VARRAY(v51, j, 1.0)

        indexed_dpd_marray<double> v53(v21, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v53, j, 1.0)
    }
}

TEST_CASE("indexed_dpd_varray::reset")
{
    indexed_dpd_marray<double> v1;
    CHECK_INDEXED_DPD_VARRAY_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        indexed_dpd_marray<double> v2(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                      {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[j]);

        v1.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                      {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 0.0)

        v1.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                       {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 1.0)

        v1.reset(v2.view(), layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 1.0)

        v1.reset(v2.cview(), layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 1.0)

        v1.reset(v2);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 1.0)

        v1.reset(v2, layouts[j]);
        CHECK_INDEXED_DPD_VARRAY(v1, j, 1.0)
    }

    v1.reset();
    CHECK_INDEXED_DPD_VARRAY_RESET(v1)
}

TEST_CASE("indexed_dpd_varray::view")
{
    indexed_dpd_marray<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);

    auto v2 = v1.cview();
    CHECK_INDEXED_DPD_VARRAY(v2, 0, 1.0)

    auto v3 = v1.view();
    CHECK_INDEXED_DPD_VARRAY(v3, 0, 1.0)

    auto v4 = std::as_const(v1).view();
    CHECK_INDEXED_DPD_VARRAY(v4, 0, 1.0)
}

TEST_CASE("indexed_dpd_varray::access")
{
    indexed_dpd_marray<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);

    auto v2 = v1[0];
    CHECK(v2.data() == v1.data(0));
    CHECK(v2.irrep() == 1u);
    CHECK(v2.num_irreps() == 2u);
    CHECK(v2.permutation() == perms[0]);
    CHECK(v2.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});

    auto v3 = std::as_const(v1)[2];
    CHECK(v3.data() == v1.data(2));
    CHECK(v3.irrep() == 1u);
    CHECK(v3.num_irreps() == 2u);
    CHECK(v3.permutation() == perms[0]);
    CHECK(v3.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});

    indexed_dpd_marray<double> v4(0, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {0, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);

    auto v5 = v4[1];
    CHECK(v5.data() == v4.data(1));
    CHECK(v5.irrep() == 1u);
    CHECK(v5.num_irreps() == 2u);
    CHECK(v5.permutation() == perms[0]);
    CHECK(v5.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
}

TEST_CASE("indexed_dpd_varray::index_iteration")
{
    int indices[3][2] = {{0, 0}, {1, 3}, {0, 3}};
    array<int,3> visited;

    indexed_dpd_marray<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);
    const indexed_dpd_marray<double> v2(v1);

    visited = {};
    v1.for_each_index(
    [&](const dpd_marray_view<double>& v, const index_vector& idx)
    {
        CHECK(2u == idx.size());
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
        CHECK(v.irrep() == 1u);
        CHECK(v.num_irreps() == 2u);
        CHECK(v.permutation() == perms[0]);
        CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    visited = {};
    v2.for_each_index(
    [&](const dpd_marray_view<const double>& v, const index_vector& idx)
    {
        CHECK(2u == idx.size());
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
        CHECK(v.irrep() == 1u);
        CHECK(v.num_irreps() == 2u);
        CHECK(v.permutation() == perms[0]);
        CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    visited = {};
    v1.for_each_index<4,2>(
    [&](const auto& v, len_type i, len_type j)
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
        CHECK(v.irrep() == 1u);
        CHECK(v.num_irreps() == 2u);
        CHECK(v.permutation() == perms[0]);
        CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    visited = {};
    v2.for_each_index<4,2>(
    [&](const auto& v, len_type i, len_type j)
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
        CHECK(v.irrep() == 1u);
        CHECK(v.num_irreps() == 2u);
        CHECK(v.permutation() == perms[0]);
        CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);

    indexed_dpd_marray<double> v3(0, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {0, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);

    visited = {};
    v3.for_each_index(
    [&](const dpd_marray_view<double>& v, const index_vector& idx)
    {
        CHECK(2u == idx.size());
        len_type i = idx[0];
        len_type j = idx[1];
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                CHECK(v.data() == v3.data(m));
                found = true;
                visited[m]++;
            }
        }
        CHECK(found);
        CHECK(v.irrep() == 1u);
        CHECK(v.num_irreps() == 2u);
        CHECK(v.permutation() == perms[0]);
        CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    });

    for (len_type i = 0;i < 3;i++)
        CHECK(visited[i]);
}

TEST_CASE("indexed_dpd_varray::element_iteration")
{
    array<len_vector,3> indices = {{{0, 0}, {1, 3}, {0, 3}}};
    array<array<int,3>,31> visited;
    array<len_vector,5> len = {{{2, 3}, {1, 2}, {3, 1}, {2, 2}, {4, 5}}};

    indexed_dpd_marray<double> v1(0, 2, len, vector<int>{1, 1}, indices, 1.0, layouts[0]);
    const indexed_dpd_marray<double> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, const irrep_vector& irreps, const len_vector& idx)
    {
        CHECK(5u == irreps.size());
        CHECK(5u == idx.size());
        int a = irreps[0];
        int b = irreps[1];
        int c = irreps[2];
        int d = irreps[3];
        int e = irreps[4];
        CHECK(a < 2u);
        CHECK(b < 2u);
        CHECK(c < 2u);
        CHECK(1u == d);
        CHECK(1u == e);
        CHECK((a^b^c^d^e) == 0u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        len_type m = idx[4];
        CHECK(i >= 0);
        CHECK(i < len[0][a]);
        CHECK(j >= 0);
        CHECK(j < len[1][b]);
        CHECK(k >= 0);
        CHECK(k < len[2][c]);
        bool found = false;
        for (int n = 0;n < 3;n++)
        {
            if (l == indices[n][0] && m == indices[n][1])
            {
                auto v3 = v1[n](a, b, c);
                CHECK(&v3(i, j, k) == &v);
                visited[&v - v1.data(n)][n]++;
                found = true;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 31;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v2.for_each_element(
    [&](const double& v, const irrep_vector& irreps, const len_vector& idx)
    {
        CHECK(5u == irreps.size());
        CHECK(5u == idx.size());
        int a = irreps[0];
        int b = irreps[1];
        int c = irreps[2];
        int d = irreps[3];
        int e = irreps[4];
        CHECK(a < 2u);
        CHECK(b < 2u);
        CHECK(c < 2u);
        CHECK(1u == d);
        CHECK(1u == e);
        CHECK((a^b^c^d^e) == 0u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        len_type m = idx[4];
        CHECK(i >= 0);
        CHECK(i < len[0][a]);
        CHECK(j >= 0);
        CHECK(j < len[1][b]);
        CHECK(k >= 0);
        CHECK(k < len[2][c]);
        bool found = false;
        for (int n = 0;n < 3;n++)
        {
            if (l == indices[n][0] && m == indices[n][1])
            {
                CHECK(&v2[n](a, b, c)(i, j, k) == &v);
                visited[&v - v2.data(n)][n]++;
                found = true;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 31;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v1.for_each_element<3,2>(
    [&](double& v, int a, int b, int c, int d, int e,
        len_type i, len_type j, len_type k, len_type l, len_type m)
    {
        CHECK(a < 2u);
        CHECK(b < 2u);
        CHECK(c < 2u);
        CHECK(1u == d);
        CHECK(1u == e);
        CHECK((a^b^c^d^e) == 0u);
        CHECK(i >= 0);
        CHECK(i < len[0][a]);
        CHECK(j >= 0);
        CHECK(j < len[1][b]);
        CHECK(k >= 0);
        CHECK(k < len[2][c]);
        bool found = false;
        for (int n = 0;n < 3;n++)
        {
            if (l == indices[n][0] && m == indices[n][1])
            {
                CHECK(&v1[n](a, b, c)(i, j, k) == &v);
                visited[&v - v1.data(n)][n]++;
                found = true;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 31;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v2.for_each_element<3,2>(
    [&](const double& v, int a, int b, int c, int d, int e,
        len_type i, len_type j, len_type k, len_type l, len_type m)
    {
        CHECK(a < 2u);
        CHECK(b < 2u);
        CHECK(c < 2u);
        CHECK(1u == d);
        CHECK(1u == e);
        CHECK((a^b^c^d^e) == 0u);
        CHECK(i >= 0);
        CHECK(i < len[0][a]);
        CHECK(j >= 0);
        CHECK(j < len[1][b]);
        CHECK(k >= 0);
        CHECK(k < len[2][c]);
        bool found = false;
        for (int n = 0;n < 3;n++)
        {
            if (l == indices[n][0] && m == indices[n][1])
            {
                CHECK(&v2[n](a, b, c)(i, j, k) == &v);
                visited[&v - v2.data(n)][n]++;
                found = true;
            }
        }
        CHECK(found);
    });

    for (len_type i = 0;i < 31;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);
}

TEST_CASE("indexed_dpd_varray::swap")
{
    indexed_dpd_marray<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}},
                                  {1, 1}, {{0, 0}, {1, 3}, {0, 3}}, 1.0, layouts[0]);
    indexed_dpd_marray<double> v2(0, 2, {{2, 3}, {1, 2}, {3, 1}, {2, 2}, {4, 5}},
                                  {1, 0}, {{0, 0}, {1, 0}, {1, 2}}, 1.0, layouts[0]);

    auto data1 = v1.data();
    auto data2 = v2.data();

    CHECK(v1.dimension() == 6u);
    CHECK(v1.dense_dimension() == 4u);
    CHECK(v1.indexed_dimension() == 2u);
    CHECK(v1.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}});
    CHECK(v1.dense_lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    CHECK(v1.indexed_lengths() == len_vector{2, 5});
    CHECK(v1.num_indices() == 3);
    CHECK(v1.indexed_irreps() == irrep_vector{1, 1});
    CHECK(v1.indices() == matrix<len_type>{{0, 0}, {1, 3}, {0, 3}});
    CHECK(v1.irrep() == 1u);
    CHECK(v1.num_irreps() == 2u);

    CHECK(v2.dimension() == 5u);
    CHECK(v2.dense_dimension() == 3u);
    CHECK(v2.indexed_dimension() == 2u);
    CHECK(v2.lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}, {2, 2}, {4, 5}});
    CHECK(v2.dense_lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}});
    CHECK(v2.indexed_lengths() == len_vector{2, 4});
    CHECK(v2.num_indices() == 3);
    CHECK(v2.indexed_irreps() == irrep_vector{1, 0});
    CHECK(v2.indices() == matrix<len_type>{{0, 0}, {1, 0}, {1, 2}});
    CHECK(v2.irrep() == 0u);
    CHECK(v2.num_irreps() == 2u);

    v1.swap(v2);

    CHECK(v2.dimension() == 6u);
    CHECK(v2.dense_dimension() == 4u);
    CHECK(v2.indexed_dimension() == 2u);
    CHECK(v2.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}});
    CHECK(v2.dense_lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    CHECK(v2.indexed_lengths() == len_vector{2, 5});
    CHECK(v2.num_indices() == 3);
    CHECK(v2.indexed_irreps() == irrep_vector{1, 1});
    CHECK(v2.indices() == matrix<len_type>{{0, 0}, {1, 3}, {0, 3}});
    CHECK(v2.irrep() == 1u);
    CHECK(v2.num_irreps() == 2u);

    CHECK(v1.dimension() == 5u);
    CHECK(v1.dense_dimension() == 3u);
    CHECK(v1.indexed_dimension() == 2u);
    CHECK(v1.lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}, {2, 2}, {4, 5}});
    CHECK(v1.dense_lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}});
    CHECK(v1.indexed_lengths() == len_vector{2, 4});
    CHECK(v1.num_indices() == 3);
    CHECK(v1.indexed_irreps() == irrep_vector{1, 0});
    CHECK(v1.indices() == matrix<len_type>{{0, 0}, {1, 0}, {1, 2}});
    CHECK(v1.irrep() == 0u);
    CHECK(v1.num_irreps() == 2u);

    swap(v2, v1);

    CHECK(v1.dimension() == 6u);
    CHECK(v1.dense_dimension() == 4u);
    CHECK(v1.indexed_dimension() == 2u);
    CHECK(v1.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}, {2, 2}, {4, 5}});
    CHECK(v1.dense_lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}});
    CHECK(v1.indexed_lengths() == len_vector{2, 5});
    CHECK(v1.num_indices() == 3);
    CHECK(v1.indexed_irreps() == irrep_vector{1, 1});
    CHECK(v1.indices() == matrix<len_type>{{0, 0}, {1, 3}, {0, 3}});
    CHECK(v1.irrep() == 1u);
    CHECK(v1.num_irreps() == 2u);

    CHECK(v2.dimension() == 5u);
    CHECK(v2.dense_dimension() == 3u);
    CHECK(v2.indexed_dimension() == 2u);
    CHECK(v2.lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}, {2, 2}, {4, 5}});
    CHECK(v2.dense_lengths() == matrix<len_type>{{2, 3}, {1, 2}, {3, 1}});
    CHECK(v2.indexed_lengths() == len_vector{2, 4});
    CHECK(v2.num_indices() == 3);
    CHECK(v2.indexed_irreps() == irrep_vector{1, 0});
    CHECK(v2.indices() == matrix<len_type>{{0, 0}, {1, 0}, {1, 2}});
    CHECK(v2.irrep() == 0u);
    CHECK(v2.num_irreps() == 2u);
}
