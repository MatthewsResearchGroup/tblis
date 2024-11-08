#include "dpd/dpd_marray.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

static std::array<dpd_layout,6> layouts =
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

#define CHECK_DPD_MARRAY_RESET(v) \
    CHECK(v.data() == nullptr); \
    CHECK(v.irrep() == 0u); \
    CHECK(v.num_irreps() == 0u); \
    CHECK(v.permutation() == dim_vector{}); \
    CHECK(v.lengths() == matrix<len_type>{});

#define CHECK_DPD_MARRAY(v,j) \
    INFO("j = " << j); \
    CHECK(v.irrep() == 1u); \
    CHECK(v.num_irreps() == 2u); \
    CHECK(v.permutation() == perms[j]); \
    CHECK(v.lengths() == matrix<len_type>{{3, 1}, {2, 2}, {1, 2}, {3, 4}}); \
    \
    { \
        auto vs = v(1,0,0,0); \
        CHECK(vs.data() == v.data() + offsets[j][0]); \
        for (int k = 0;k < 4;k++) \
        { \
            CHECK(vs.length(k) == lengths[0][k]); \
            CHECK(vs.stride(k) == strides[j][0][k]); \
        } \
    } \
    \
    { \
        auto vs = v({0,1,0,0}); \
        CHECK(vs.data() == v.data() + offsets[j][1]); \
        CHECK(vs.lengths() == lengths[1]); \
        CHECK(vs.strides() == strides[j][1]); \
    } \
    \
    for (int i = 2;i < 8;i++) \
    { \
        INFO("i = " << i); \
        auto vs = v(irreps[i]); \
        CHECK(vs.data() == v.data() + offsets[j][i]); \
        CHECK(vs.lengths() == lengths[i]); \
        CHECK(vs.strides() == strides[j][i]); \
    }

TEST_CASE("dpd_varray::constructor")
{
    double data[168];
    for (len_type i = 0;i < 168;i++) data[i] = i;

    dpd_marray<double> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);
    dpd_marray_view<double> v00(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);

    dpd_marray<double> v1;
    CHECK_DPD_MARRAY_RESET(v1)

    dpd_marray<double> v2(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);
    CHECK_DPD_MARRAY(v2, 0)

    for (int j = 1;j < 6;j++)
    {
        dpd_marray<double> v3(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[j]);
        CHECK_DPD_MARRAY(v3, j)
    }

    dpd_marray<double> v5(v2);
    CHECK_DPD_MARRAY(v5, 0)

    dpd_marray<double> v51(v0);
    CHECK_DPD_MARRAY(v51, 0)
    for (len_type i = 0;i < 168;i++) CHECK(0.0 == v51.data()[i]);

    dpd_marray<double> v52(v00);
    CHECK_DPD_MARRAY(v51, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v52.data()[i]);

    dpd_marray<double> v6(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY(v6, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v52.data()[i]);
}

TEST_CASE("dpd_varray::reset")
{
    double data[168];
    for (len_type i = 0;i < 168;i++) data[i] = i;

    dpd_marray<double> v1;
    dpd_marray_view<double> v3(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray_view<const double> v4(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray<double> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);

    CHECK_DPD_MARRAY_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        v1.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, uninitialized, layouts[j]);
        CHECK_DPD_MARRAY(v1, j)
    }

    v1.reset(v3);
    CHECK_DPD_MARRAY(v1, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v1.data()[i]);

    v1.reset(v4);
    CHECK_DPD_MARRAY(v1, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v1.data()[i]);

    v1.reset(v0);
    CHECK_DPD_MARRAY(v1, 0)
    for (len_type i = 0;i < 168;i++) CHECK(0.0 == v1.data()[i]);

    v1.reset(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY(v1, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v1.data()[i]);

    v1.reset(dpd_marray_view<const double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY(v1, 0)
    for (len_type i = 0;i < 168;i++) CHECK(i == v1.data()[i]);

    v1.reset();
    CHECK_DPD_MARRAY_RESET(v1)
}

TEST_CASE("dpd_varray::permute")
{
    int perm_irreps[8] = {1, 0, 2, 3, 4, 5, 7, 6};

    dim_vector perms2[6] =
        {{2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}};

    for (int j = 0;j < 6;j++)
    {
        INFO("j = " << j);

        dpd_marray<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[j]);

        auto v2 = v1.permuted({1, 0, 2, 3});
        CHECK(v2.data() == v1.data());
        CHECK(v2.irrep() == 1u);
        CHECK(v2.num_irreps() == 2u);
        CHECK(v2.permutation() == perms2[j]);
        CHECK(v2.lengths() == matrix<len_type>{{2, 2}, {3, 1}, {1, 2}, {3, 4}});

        for (int i = 0;i < 8;i++)
        {
            INFO("i = " << i);
            len_vector len(4);
            stride_vector stride(4);
            for (int k = 0;k < 4;k++)
            {
                len[k] = lengths[i][perms2[1][k]];
                stride[k] = strides[j][i][perms2[1][k]];
            }
            auto vs = v2(irreps[perm_irreps[i]]);
            CHECK(vs.data() == v1.data() + offsets[j][i]);
            CHECK(vs.lengths() == len);
            CHECK(vs.strides() == stride);
        }
    }
}

TEST_CASE("dpd_varray::block_iteration")
{
    array<array<int,2>,2> visited;

    for (int l = 0;l < 6;l++)
    {
        INFO("l = " << l);

        dpd_marray<double> v1(0, 2, {{2, 3}, {1, 2}, {3, 1}}, layouts[l]);

        visited = {};
        v1.for_each_block(
        [&](marray_view<double>&& v3, const irrep_vector& irreps)
        {
            CHECK(3u == irreps.size());
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            CHECK(i < 2u);
            CHECK(j < 2u);
            CHECK(k < 2u);
            CHECK((i^j^k) == 0u);
            auto v4 = v1({i, j, k});
            CHECK(v4.data() == v3.data());
            CHECK(v4.lengths() == v3.lengths());
            CHECK(v4.strides() == v3.strides());
            visited[i][j]++;
        });

        for (len_type i = 0;i < 2;i++)
        for (len_type j = 0;j < 2;j++)
            CHECK(visited[i][j]);

        visited = {};
        v1.for_each_block<3>(
        [&](marray_view<double,3>&& v3, int i, int j, int k)
        {
            CHECK(i < 2u);
            CHECK(j < 2u);
            CHECK(k < 2u);
            CHECK((i^j^k) == 0u);
            auto v4 = v1(i, j, k);
            CHECK(v4.data() == v3.data());
            CHECK(v4.lengths() == v3.lengths());
            CHECK(v4.strides() == v3.strides());
            visited[i][j]++;
        });

        for (len_type i = 0;i < 2;i++)
        for (len_type j = 0;j < 2;j++)
            CHECK(visited[i][j]);
    }
}

TEST_CASE("dpd_varray::element_iteration")
{
    array<int,31> visited;
    array<len_vector,3> len = {{{2, 3}, {1, 2}, {3, 1}}};

    for (int l = 0;l < 6;l++)
    {
        INFO("l = " << l);

        dpd_marray<double> v1(0, 2, len, layouts[l]);

        visited = {};
        v1.for_each_element(
        [&](double& v, const irrep_vector& irreps, const len_vector& pos)
        {
            CHECK(3u == irreps.size());
            CHECK(3u == pos.size());
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            len_type a = pos[0];
            len_type b = pos[1];
            len_type c = pos[2];
            CHECK(i < 2u);
            CHECK(j < 2u);
            CHECK(k < 2u);
            CHECK(a >= 0);
            CHECK(a < len[0][i]);
            CHECK(b >= 0);
            CHECK(b < len[1][j]);
            CHECK(c >= 0);
            CHECK(c < len[2][k]);
            CHECK((i^j^k) == 0u);
            auto v3 = v1(i, j, k);
            CHECK(&v3(a, b, c) == &v);
            visited[&v - v1.data()]++;
        });

        for (int i = 0;i < 31;i++)
            CHECK(visited[i]);

        visited = {};
        v1.for_each_element<3>(
        [&](double& v, int i, int j, int k, len_type a, len_type b, len_type c)
        {
            CHECK(i < 2u);
            CHECK(j < 2u);
            CHECK(k < 2u);
            CHECK(a >= 0);
            CHECK(a < len[0][i]);
            CHECK(b >= 0);
            CHECK(b < len[1][j]);
            CHECK(c >= 0);
            CHECK(c < len[2][k]);
            CHECK((i^j^k) == 0u);
            auto v3 = v1(i, j, k);
            CHECK(&v3(a, b, c) == &v);
            visited[&v - v1.data()]++;
        });

        for (int i = 0;i < 31;i++)
            CHECK(visited[i]);
    }
}

TEST_CASE("dpd_varray::swap")
{
    dpd_marray<double> v1(1, 2, {{2, 3}, {2, 1}, {5, 3}}, PREFIX_ROW_MAJOR);
    dpd_marray<double> v2(0, 2, {{1, 1}, {6, 3}, {2, 4}}, PREFIX_COLUMN_MAJOR);

    double* data1 = v1.data();
    double* data2 = v2.data();

    v1.swap(v2);

    CHECK(v1.data() == data2);
    CHECK(v1.irrep() == 0u);
    CHECK(v1.num_irreps() == 2u);
    CHECK(v1.permutation() == dim_vector{0, 1, 2});
    CHECK(v1.lengths() == matrix<len_type>{{1, 1}, {6, 3}, {2, 4}});

    CHECK(v2.data() == data1);
    CHECK(v2.irrep() == 1u);
    CHECK(v2.num_irreps() == 2u);
    CHECK(v2.permutation() == dim_vector{2, 1, 0});
    CHECK(v2.lengths() == matrix<len_type>{{2, 3}, {2, 1}, {5, 3}});

    swap(v2, v1);

    CHECK(v1.data() == data1);
    CHECK(v1.irrep() == 1u);
    CHECK(v1.num_irreps() == 2u);
    CHECK(v1.permutation() == dim_vector{2, 1, 0});
    CHECK(v1.lengths() == matrix<len_type>{{2, 3}, {2, 1}, {5, 3}});

    CHECK(v2.data() == data2);
    CHECK(v2.irrep() == 0u);
    CHECK(v2.num_irreps() == 2u);
    CHECK(v2.permutation() == dim_vector{0, 1, 2});
    CHECK(v2.lengths() == matrix<len_type>{{1, 1}, {6, 3}, {2, 4}});
}

TEST_CASE("dpd_varray::slice")
{
    for (auto k : range(layouts.size()))
    {
        INFO("k = " << k);
        dpd_marray<double> v1(1, 2, {{2, 3}, {2, 1}, {5, 3}}, layouts[k]);

        auto v2 = v1(dpd_range(1, {2}), slice::all, dpd_index{1, 1});
        CHECK(v2.dimension() == 2);
        CHECK(v2.irrep() == 0);
        CHECK(v2.num_irreps() == 2);
        CHECK(v2.lengths() == matrix<len_type>{{0, 2}, {2, 1}});

        v2.for_each_element<2>([&](double& v, int irrepi, int irrepj, len_type i, len_type j)
        {
            INFO("irrepi = " << irrepi);
            INFO("irrepj = " << irrepj);
            INFO("i = " << i);
            INFO("j = " << j);
            CHECK(&v-v1.data() == &v1(irrepi,irrepj,1)(i,j,1)-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i > 1) return;
            if (irrepk != 1 || k != 1) return;
            INFO("irrepi = " << irrepi);
            INFO("irrepj = " << irrepj);
            INFO("irrepk = " << irrepk);
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("k = " << k);
            CHECK(&v-v1.data() == &v2(irrepi,irrepj)(i,j)-v1.data());
        });

        auto v3 = v2(dpd_index{1, 1}, dpd_range(0, {1, 2})(1, {1}));
        CHECK(v3.dimension() == 1);
        CHECK(v3.irrep() == 1);
        CHECK(v3.num_irreps() == 2);
        CHECK(v3.lengths() == matrix<len_type>{{1, 1}});

        v3.for_each_element<1>([&](double& v, int irrepi, len_type i)
        {
            INFO("irrepi = " << irrepi);
            INFO("i = " << i);
            CHECK(&v-v1.data() == &v1(1,irrepi,1)(1,i,1)-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i != 1) return;
            if (irrepj == 0 && i != 1) return;
            if (irrepk != 1 || k != 1) return;
            INFO("irrepi = " << irrepi);
            INFO("irrepj = " << irrepj);
            INFO("irrepk = " << irrepk);
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("k = " << k);
            CHECK(&v3(irrepj)(irrepj == 0 ? j-1 : j)-v1.data() == &v-v1.data());
        });

        auto v4 = v1({dpd_range(1, {2}), dpd_range(0, {1, 2})(1, {1}), dpd_range(0, {2})(1, {2, 3})});
        CHECK(v4.dimension() == 3);
        CHECK(v4.irrep() == 1);
        CHECK(v4.num_irreps() == 2);
        CHECK(v4.lengths() == matrix<len_type>{{0, 2}, {1, 1}, {2, 1}});

        v4.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            INFO("irrepi = " << irrepi);
            INFO("irrepj = " << irrepj);
            INFO("irrepk = " << irrepk);
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("k = " << k);
            CHECK(&v-v1.data() == &v1(irrepi,irrepj,irrepk)(i, irrepj == 0 ? j+1 : j, irrepk == 1 ? k+2 : k)-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i > 1) return;
            if (irrepj == 0 && j != 1) return;
            if (k > (irrepk == 0 ? 1 : 2) || k < (irrepk == 0 ? 0 : 2)) return;
            INFO("irrepi = " << irrepi);
            INFO("irrepj = " << irrepj);
            INFO("irrepk = " << irrepk);
            INFO("i = " << i);
            INFO("j = " << j);
            INFO("k = " << k);
            CHECK(&v-v1.data() == &v4(irrepi,irrepj,irrepk)(i, irrepj == 0 ? j-1 : j, irrepk == 1 ? k-2 : k)-v1.data());
        });
    }
}
