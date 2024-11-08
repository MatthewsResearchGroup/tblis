#include "marray.hpp"

#include "rotate.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

TEST_CASE("marray::constructor")
{
    double data[40];

    marray_view<double,3> v0({4, 2, 5}, data);
    marray_view<const double,3> v01({4, 2, 5}, data);
    marray_view<int,3> v02({4, 2, 5}, (int*)data);
    marray<int,3> v03{4, 2, 5};
    marray_view<double,4> v04({3, 4, 2, 5}, data);

    marray<double,2> v1;
    CHECK(v1.data() == nullptr);
    CHECK(v1.lengths() == array<len_type,2>{0, 0});
    CHECK(v1.strides() == array<stride_type,2>{0, 0});

    marray<double,3> v2{4, 2, 5};
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data()[0] == 0);

    marray<double,3> v21{4, 2, 5};
    CHECK(v21.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v21.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v21.data()[0] == 0);

    marray<double,3> v3(array<char,3>{4, 2, 5});
    CHECK(v3.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v3.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v3.data()[0] == 0);

    marray<double,3> v31({4, 2, 5}, 1.0);
    CHECK(v31.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v31.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v31.data()[0] == 1);

    marray<double,3> v32(array<char,3>{4, 2, 5}, 1.0);
    CHECK(v32.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v32.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v32.data()[0] == 1);

    marray<double,3> v4({4, 2, 5}, 1.0, COLUMN_MAJOR);
    CHECK(v4.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v4.strides() == array<stride_type,3>{1, 4, 8});
    CHECK(v4.data()[0] == 1);

    marray<double,3> v41({4, 2, 5}, COLUMN_MAJOR);
    CHECK(v41.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v41.strides() == array<stride_type,3>{1, 4, 8});
    CHECK(v41.data()[0] == 0);

    marray<double,3> v5(v0);
    CHECK(v5.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v5.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,3> v52(v01);
    CHECK(v52.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v52.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,3> v53(v02);
    CHECK(v53.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v53.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,3> v54(v03);
    CHECK(v54.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v54.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,3> v55(v04[0]);
    CHECK(v55.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v55.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,3> v51(v31);
    CHECK(v51.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v51.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v51.data()[0] == 1);

    marray<double,3> v6(marray<double,3>{4, 2, 5});
    CHECK(v6.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v6.strides() == array<stride_type,3>{10, 5, 1});

    marray<double,1> v7(std::vector<double>{1,2,3});
    CHECK(v7.lengths() == array<len_type,1>{3});
    CHECK(v7.strides() == array<stride_type,1>{1});
    CHECK(v7.data()[0] == 1);
}

TEST_CASE("varray::constructor")
{
    double data[40];

    marray_view<double> v0({4, 2, 5}, data);
    marray_view<const double> v01({4, 2, 5}, data);
    marray_view<int> v02({4, 2, 5}, (int*)data);
    marray<int> v03{4, 2, 5};

    marray<double> v1;
    CHECK(v1.dimension() == 0u);
    CHECK(v1.data() == nullptr);

    marray<double> v21{4, 2, 5};
    CHECK(v21.dimension() == 3u);
    CHECK(v21.lengths() == len_vector{4, 2, 5});
    CHECK(v21.strides() == stride_vector{10, 5, 1});
    CHECK(v21.data()[0] == 0);

    marray<double> v2{4, 2, 5};
    CHECK(v2.dimension() == 3u);
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data()[0] == 0);

    /*
     * The semantics for this have changed, now a 1-D copy of
     * the vector is created.
     *
    marray<double> v3(vector<char>{4, 2, 5});
    CHECK(v3.dimension() == 3u);
    CHECK(v3.lengths() == len_vector{4, 2, 5});
    CHECK(v3.strides() == stride_vector{10, 5, 1});
    CHECK(v3.data()[0] == 0);
     */

    marray<double> v31({4, 2, 5}, 1.0);
    CHECK(v31.dimension() == 3u);
    CHECK(v31.lengths() == len_vector{4, 2, 5});
    CHECK(v31.strides() == stride_vector{10, 5, 1});
    CHECK(v31.data()[0] == 1);

    marray<double> v32(vector<char>{4, 2, 5}, 1.0);
    CHECK(v32.dimension() == 3u);
    CHECK(v32.lengths() == len_vector{4, 2, 5});
    CHECK(v32.strides() == stride_vector{10, 5, 1});
    CHECK(v32.data()[0] == 1);

    marray<double> v4({4, 2, 5}, 1.0, COLUMN_MAJOR);
    CHECK(v4.dimension() == 3u);
    CHECK(v4.lengths() == len_vector{4, 2, 5});
    CHECK(v4.strides() == stride_vector{1, 4, 8});
    CHECK(v4.data()[0] == 1);

    marray<double> v41({4, 2, 5}, COLUMN_MAJOR);
    CHECK(v41.dimension() == 3u);
    CHECK(v41.lengths() == len_vector{4, 2, 5});
    CHECK(v41.strides() == stride_vector{1, 4, 8});
    CHECK(v41.data()[0] == 0);

    marray<double> v5(v0);
    CHECK(v5.dimension() == 3u);
    CHECK(v5.lengths() == len_vector{4, 2, 5});
    CHECK(v5.strides() == stride_vector{10, 5, 1});

    marray<double> v52(v01);
    CHECK(v52.dimension() == 3u);
    CHECK(v52.lengths() == len_vector{4, 2, 5});
    CHECK(v52.strides() == stride_vector{10, 5, 1});

    marray<double> v53(v02);
    CHECK(v53.dimension() == 3u);
    CHECK(v53.lengths() == len_vector{4, 2, 5});
    CHECK(v53.strides() == stride_vector{10, 5, 1});

    marray<double> v54(v03);
    CHECK(v54.dimension() == 3u);
    CHECK(v54.lengths() == len_vector{4, 2, 5});
    CHECK(v54.strides() == stride_vector{10, 5, 1});

    marray<double> v51(v31);
    CHECK(v51.dimension() == 3u);
    CHECK(v51.lengths() == len_vector{4, 2, 5});
    CHECK(v51.strides() == stride_vector{10, 5, 1});
    CHECK(v51.data()[0] == 1);

    marray<double> v6(marray<double>{4, 2, 5});
    CHECK(v6.dimension() == 3u);
    CHECK(v6.lengths() == len_vector{4, 2, 5});
    CHECK(v6.strides() == stride_vector{10, 5, 1});

    marray<double> v7(std::vector<double>{1,2,3});
    CHECK(v7.dimension() == 1u);
    CHECK(v7.lengths() == len_vector{3});
    CHECK(v7.strides() == stride_vector{1});
    CHECK(v7.data()[0] == 1);
}

TEST_CASE("marray::reset")
{
    double data[40];

    marray_view<double,3> v0({4, 2, 5}, data);
    marray_view<const double,3> v01({4, 2, 5}, data);
    marray_view<int,3> v02({4, 2, 5}, (int*)data);
    marray<int,3> v03{4, 2, 5};
    marray_view<double,4> v04({3, 4, 2, 5}, data);

    marray<double,3> v1;
    marray<double,3> v2({4, 2, 5}, 1.0);

    v1.reset({4, 2, 5});
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data()[0] == 0);

    v1.reset(array<char,3>{4, 2, 5});
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data()[0] == 0);

    v1.reset({4, 2, 5}, 1.0);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset(array<char,3>{4, 2, 5}, 1.0);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset({4, 2, 5}, 1.0, COLUMN_MAJOR);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{1, 4, 8});
    CHECK(v1.data()[0] == 1);

    v1.reset({4, 2, 5}, COLUMN_MAJOR);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{1, 4, 8});
    CHECK(v1.data()[0] == 0);

    v1.reset(v0);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v01);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v02);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v03);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v04[0]);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v2);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset(marray<double,3>{4, 2, 5});
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset();
    CHECK(v1.data() == nullptr);
    CHECK(v1.lengths() == array<len_type,3>{0, 0, 0});
    CHECK(v1.strides() == array<stride_type,3>{0, 0, 0});
}

TEST_CASE("varray::reset")
{
    double data[40];

    marray_view<double> v0({4, 2, 5}, data);
    marray_view<const double> v01({4, 2, 5}, data);
    marray_view<int> v02({4, 2, 5}, (int*)data);
    marray<int> v03{4, 2, 5};

    marray<double> v1;
    marray<double> v2({4, 2, 5}, 1.0);

    v1.reset({4, 2, 5});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data()[0] == 0);

    /*
     * The semantics for this have changed, now a 1-D copy of
     * the vector is created.
     *
    v1.reset(vector<char>{4, 2, 5});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data()[0] == 0);
     */

    v1.reset({4, 2, 5}, 1.0);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset(vector<char>{4, 2, 5}, 1.0);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset({4, 2, 5}, 1.0, COLUMN_MAJOR);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{1, 4, 8});
    CHECK(v1.data()[0] == 1);

    v1.reset({4, 2, 5}, COLUMN_MAJOR);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{1, 4, 8});
    CHECK(v1.data()[0] == 0);

    v1.reset(v0);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(v01);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(v02);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(v03);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(v2);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data()[0] == 1);

    v1.reset(marray<double>{4, 2, 5});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset();
    CHECK(v1.dimension() == 0u);
    CHECK(v1.data() == nullptr);
}

TEST_CASE("marray::initialize")
{
    marray<double,3> v1({3, 2, 3}, ROW_MAJOR);
    marray<double,3> v2({3, 2, 3}, COLUMN_MAJOR);

    v1 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    CHECK(*(const std::array<double,18>*)v1.data() ==
          std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                 6, 7, 8, 9,10,11,
                                12,13,14,15,16,17});

    v2 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    CHECK(*(const std::array<double,18>*)v2.data() ==
          std::array<double,18>{ 0, 6,12, 3, 9,15,
                                 1, 7,13, 4,10,16,
                                 2, 8,14, 5,11,17});

    marray<double,3> v4({{{ 0, 1, 2},
                          { 3, 4, 5}},
                         {{ 6, 7, 8},
                          { 9,10,11}},
                         {{12,13,14},
                          {15,16,17}}});

    CHECK(*(const std::array<double,18>*)v4.data() ==
          std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                 6, 7, 8, 9,10,11,
                                12,13,14,15,16,17});

    marray<double,3> v5 = {{{ 0, 1, 2},
                            { 3, 4, 5}},
                           {{ 6, 7, 8},
                            { 9,10,11}},
                           {{12,13,14},
                            {15,16,17}}};

    CHECK(*(const std::array<double,18>*)v5.data() ==
          std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                 6, 7, 8, 9,10,11,
                                12,13,14,15,16,17});
}

TEST_CASE("varray::assign")
{
    double data1[6] = {0, 1, 2,
                      3, 4, 5};
    int data2[6] = {0, 1, 2,
                    3, 4, 5};

    marray<double> v1{2, 3};

    v1 = marray_view<double>({2, 3}, data1);
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{0, 1, 2,
                                                          3, 4, 5});

    v1 = 1.0;
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{1, 1, 1,
                                                          1, 1, 1});

    v1 = marray_view<int>({2, 3}, data2);
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{0, 1, 2,
                                                          3, 4, 5});

    v1 = 1;
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{1, 1, 1,
                                                          1, 1, 1});

    v1 = marray_view<const double>({2, 3}, data1);
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{0, 1, 2,
                                                          3, 4, 5});
}

TEST_CASE("marray::resize")
{
    double data[6] = {0, 1, 2,
                      3, 4, 5};

    marray<double,2> v1(marray_view<const double,2>({2, 3}, data));

    v1.resize({2, 2});
    CHECK(v1.lengths() == array<len_type,2>{2, 2});
    CHECK(v1.strides() == array<stride_type,2>{2, 1});
    CHECK(*(array<double,4>*)v1.data() == array<double,4>{0, 1,
                                                          3, 4});

    v1.resize(array<char,2>{3, 4}, 1);
    CHECK(v1.lengths() == array<len_type,2>{3, 4});
    CHECK(v1.strides() == array<stride_type,2>{4, 1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{0, 1, 1, 1,
                                                            3, 4, 1, 1,
                                                            1, 1, 1, 1});
}

TEST_CASE("varray::resize")
{
    double data[6] = {0, 1, 2,
                      3, 4, 5};

    marray<double> v1(marray_view<const double>({2, 3}, data));

    v1.resize({2, 2});
    CHECK(v1.lengths() == len_vector{2, 2});
    CHECK(v1.strides() == stride_vector{2, 1});
    CHECK(*(array<double,4>*)v1.data() == array<double,4>{0, 1,
                                                          3, 4});

    v1.resize(vector<char>{3, 4}, 1);
    CHECK(v1.lengths() == len_vector{3, 4});
    CHECK(v1.strides() == stride_vector{4, 1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{0, 1, 1, 1,
                                                            3, 4, 1, 1,
                                                            1, 1, 1, 1});
}

TEST_CASE("marray::push_pop")
{
    double data1[6] = {0, 1, 2,
                       3, 4, 5};
    double data2[3] = {6, 7, 8};
    double data3[3] = {-1, -1, -1};

    marray<double,2> v1(marray_view<const double,2>({2, 3}, data1));
    marray_view<const double,1> v2({3}, data2);
    marray_view<const double,1> v3({3}, data3);

    v1.push_back(0, v2);
    CHECK(v1.lengths() == array<len_type,2>{3, 3});
    CHECK(v1.strides() == array<stride_type,2>{3, 1});
    CHECK(*(array<double,9>*)v1.data() == array<double,9>{0, 1, 2,
                                                          3, 4, 5,
                                                          6, 7, 8});

    v1.push_back(1, v3);
    CHECK(v1.lengths() == array<len_type,2>{3, 4});
    CHECK(v1.strides() == array<stride_type,2>{4, 1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{0, 1, 2, -1,
                                                            3, 4, 5, -1,
                                                            6, 7, 8, -1});

    v1.pop_back(0);
    CHECK(v1.lengths() == array<len_type,2>{2, 4});
    CHECK(v1.strides() == array<stride_type,2>{4, 1});
    CHECK(*(array<double,8>*)v1.data() == array<double,8>{0, 1, 2, -1,
                                                          3, 4, 5, -1});

    v1.pop_back(1);
    CHECK(v1.lengths() == array<len_type,2>{2, 3});
    CHECK(v1.strides() == array<stride_type,2>{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{0, 1, 2,
                                                          3, 4, 5});

    marray<double,1> v4(marray_view<const double,1>({6}, data1));

    v4.push_back(6);

    CHECK(v4.lengths() == array<len_type,1>{7});
    CHECK(v4.strides() == array<stride_type,1>{1});
    CHECK(*(array<double,7>*)v4.data() == array<double,7>{0, 1, 2, 3, 4, 5, 6});

    v4.pop_back();
    v4.pop_back();

    CHECK(v4.lengths() == array<len_type,1>{5});
    CHECK(v4.strides() == array<stride_type,1>{1});
    CHECK(*(array<double,5>*)v4.data() == array<double,5>{0, 1, 2, 3, 4});
}

TEST_CASE("varray::push_pop")
{
    double data1[6] = {0, 1, 2,
                       3, 4, 5};
    double data2[3] = {6, 7, 8};
    double data3[3] = {-1, -1, -1};

    marray<double> v1(marray_view<const double>({2, 3}, data1));
    marray_view<const double> v2({3}, data2);
    marray_view<const double> v3({3}, data3);

    v1.push_back(0, v2);
    CHECK(v1.lengths() == len_vector{3, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,9>*)v1.data() == array<double,9>{0, 1, 2,
                                                          3, 4, 5,
                                                          6, 7, 8});

    v1.push_back(1, v3);
    CHECK(v1.lengths() == len_vector{3, 4});
    CHECK(v1.strides() == stride_vector{4, 1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{0, 1, 2, -1,
                                                            3, 4, 5, -1,
                                                            6, 7, 8, -1});

    v1.pop_back(0);
    CHECK(v1.lengths() == len_vector{2, 4});
    CHECK(v1.strides() == stride_vector{4, 1});
    CHECK(*(array<double,8>*)v1.data() == array<double,8>{0, 1, 2, -1,
                                                          3, 4, 5, -1});

    v1.pop_back(1);
    CHECK(v1.lengths() == len_vector{2, 3});
    CHECK(v1.strides() == stride_vector{3, 1});
    CHECK(*(array<double,6>*)v1.data() == array<double,6>{0, 1, 2,
                                                          3, 4, 5});
}

TEST_CASE("marray::view")
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.cview();
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.view();
    CHECK(v3.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v3.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v3.data() == v1.data());

    auto v4 = std::as_const(v1).view();
    CHECK(v4.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v4.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v4.data() == v1.data());
}

TEST_CASE("varray::view")
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.cview();
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.view();
    CHECK(v3.lengths() == len_vector{4, 2, 5});
    CHECK(v3.strides() == stride_vector{10, 5, 1});
    CHECK(v3.data() == v1.data());

    auto v4 = std::as_const(v1).view();
    CHECK(v4.lengths() == len_vector{4, 2, 5});
    CHECK(v4.strides() == stride_vector{10, 5, 1});
    CHECK(v4.data() == v1.data());
}

TEST_CASE("marray::permuted")
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.permuted({1, 0, 2});
    CHECK(v2.lengths() == array<len_type,3>{2, 4, 5});
    CHECK(v2.strides() == array<stride_type,3>{5, 10, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.permuted(array<char,3>{2, 0, 1});
    CHECK(v3.lengths() == array<len_type,3>{5, 4, 2});
    CHECK(v3.strides() == array<stride_type,3>{1, 10, 5});
    CHECK(v3.data() == v1.data());

    auto v4 = std::as_const(v1).permuted(1, 0, 2);
    CHECK(v4.lengths() == array<len_type,3>{2, 4, 5});
    CHECK(v4.strides() == array<stride_type,3>{5, 10, 1});
    CHECK(v4.data() == v1.data());

    auto v5 = v1.permuted(1, 0, 2);
    CHECK(v5.lengths() == array<len_type,3>{2, 4, 5});
    CHECK(v5.strides() == array<stride_type,3>{5, 10, 1});
    CHECK(v5.data() == v1.data());
}

TEST_CASE("varray::permuted")
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.permuted({1, 0, 2});
    CHECK(v2.lengths() == len_vector{2, 4, 5});
    CHECK(v2.strides() == stride_vector{5, 10, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.permuted(vector<char>{2, 0, 1});
    CHECK(v3.lengths() == len_vector{5, 4, 2});
    CHECK(v3.strides() == stride_vector{1, 10, 5});
    CHECK(v3.data() == v1.data());

    auto v4 = std::as_const(v1).permuted(1, 0, 2);
    CHECK(v4.lengths() == len_vector{2, 4, 5});
    CHECK(v4.strides() == stride_vector{5, 10, 1});
    CHECK(v4.data() == v1.data());

    auto v5 = v1.permuted(1, 0, 2);
    CHECK(v5.lengths() == len_vector{2, 4, 5});
    CHECK(v5.strides() == stride_vector{5, 10, 1});
    CHECK(v5.data() == v1.data());
}

TEST_CASE("marray::transposed")
{
    marray<double,2> v1{4, 8};

    auto v2 = v1.transposed();
    CHECK(v2.lengths() == array<len_type,2>{8, 4});
    CHECK(v2.strides() == array<stride_type,2>{1, 8});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.T();
    CHECK(v3.lengths() == array<len_type,2>{8, 4});
    CHECK(v3.strides() == array<stride_type,2>{1, 8});
    CHECK(v3.data() == v1.data());

    auto v4 = std::as_const(v1).T();
    CHECK(v4.lengths() == array<len_type,2>{8, 4});
    CHECK(v4.strides() == array<stride_type,2>{1, 8});
    CHECK(v4.data() == v1.data());
}

TEST_CASE("marray::lowered")
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.lowered<3>({1, 2});
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.lowered<2>(array<char,1>{1});
    CHECK(v3.lengths() == array<len_type,2>{4, 10});
    CHECK(v3.strides() == array<stride_type,2>{10, 1});
    CHECK(v3.data() == v1.data());

    auto v4 = v1.lowered<1>({});
    CHECK(v4.lengths() == array<len_type,1>{40});
    CHECK(v4.strides() == array<stride_type,1>{1});
    CHECK(v4.data() == v1.data());

    auto v5 = std::as_const(v1).lowered(1, 2);
    CHECK(v5.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v5.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data() == v1.data());
}

TEST_CASE("varray::lowered")
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.lowered({1, 2});
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.lowered(vector<char>{1});
    CHECK(v3.lengths() == len_vector{4, 10});
    CHECK(v3.strides() == stride_vector{10, 1});
    CHECK(v3.data() == v1.data());

    auto v3b = v1.lowered(1);
    CHECK(v3b.lengths() == array<len_type,2>{4, 10});
    CHECK(v3b.strides() == array<stride_type,2>{10, 1});
    CHECK(v3b.data() == v1.data());

    auto v4 = v1.lowered();
    CHECK(v4.lengths() == array<len_type,1>{40});
    CHECK(v4.strides() == array<stride_type,1>{1});
    CHECK(v4.data() == v1.data());

    auto v5 = std::as_const(v1).lowered(1, 2);
    CHECK(v5.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v5.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data() == v1.data());
}

TEST_CASE("marray::rotate")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double,2> v1(marray_view<const double,2>({4, 3}, data.data()));

    rotate(v1, 1, 1);
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6,
                                                            10,11, 9});

    rotate(v1, 0, -1);
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{10,11, 9,
                                                             1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6});

    rotate(v1, {4,3});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{10,11, 9,
                                                             1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6});

    rotate(v1, array<char,2>{1,1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 2, 0, 1,
                                                             5, 3, 4,
                                                             8, 6, 7,
                                                            11, 9,10});
}

TEST_CASE("varray::rotate")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double> v1(marray_view<const double>({4, 3}, data.data()));

    rotate(v1, 1, 1);
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6,
                                                            10,11, 9});

    rotate(v1, 0, -1);
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{10,11, 9,
                                                             1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6});

    rotate(v1, {4,3});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{10,11, 9,
                                                             1, 2, 0,
                                                             4, 5, 3,
                                                             7, 8, 6});

    rotate(v1, vector<char>{1,1});
    CHECK(*(array<double,12>*)v1.data() == array<double,12>{ 2, 0, 1,
                                                             5, 3, 4,
                                                             8, 6, 7,
                                                            11, 9,10});
}

TEST_CASE("marray::front_back")
{
    marray<double,1> v1({8}, 0);

    CHECK(&v1.cfront() == v1.data());
    CHECK(&v1.front() == v1.data());
    CHECK(&(std::as_const(v1).front()) == v1.data());
    CHECK(&v1.cback() == v1.data()+7);
    CHECK(&v1.back() == v1.data()+7);
    CHECK(&(std::as_const(v1).back()) == v1.data()+7);

    marray<double,3> v2{4, 2, 5};

    auto v3 = v2.cfront(0);
    CHECK(v3.lengths() == array<len_type,2>{2, 5});
    CHECK(v3.strides() == array<stride_type,2>{5, 1});
    CHECK(v3.data() == v2.data());

    auto v4 = v2.front(1);
    CHECK(v4.lengths() == array<len_type,2>{4, 5});
    CHECK(v4.strides() == array<stride_type,2>{10, 1});
    CHECK(v4.data() == v2.data());

    auto v5 = std::as_const(v2).front(1);
    CHECK(v5.lengths() == array<len_type,2>{4, 5});
    CHECK(v5.strides() == array<stride_type,2>{10, 1});
    CHECK(v5.data() == v2.data());

    auto v6 = v2.cback(0);
    CHECK(v6.lengths() == array<len_type,2>{2, 5});
    CHECK(v6.strides() == array<stride_type,2>{5, 1});
    CHECK(v6.data() == v2.data() + 30);

    auto v7 = v2.back(1);
    CHECK(v7.lengths() == array<len_type,2>{4, 5});
    CHECK(v7.strides() == array<stride_type,2>{10, 1});
    CHECK(v7.data() == v2.data() + 5);

    auto v8 = std::as_const(v2).back(1);
    CHECK(v8.lengths() == array<len_type,2>{4, 5});
    CHECK(v8.strides() == array<stride_type,2>{10, 1});
    CHECK(v8.data() == v2.data() + 5);
}

TEST_CASE("varray::front_back")
{
    marray<double> v2{4, 2, 5};

    auto v3 = v2.cfront(0);
    CHECK(v3.lengths() == len_vector{2, 5});
    CHECK(v3.strides() == stride_vector{5, 1});
    CHECK(v3.data() == v2.data());

    auto v4 = v2.front(1);
    CHECK(v4.lengths() == len_vector{4, 5});
    CHECK(v4.strides() == stride_vector{10, 1});
    CHECK(v4.data() == v2.data());

    auto v5 = std::as_const(v2).front(1);
    CHECK(v5.lengths() == len_vector{4, 5});
    CHECK(v5.strides() == stride_vector{10, 1});
    CHECK(v5.data() == v2.data());

    auto v6 = v2.cback(0);
    CHECK(v6.lengths() == len_vector{2, 5});
    CHECK(v6.strides() == stride_vector{5, 1});
    CHECK(v6.data() == v2.data() + 30);

    auto v7 = v2.back(1);
    CHECK(v7.lengths() == len_vector{4, 5});
    CHECK(v7.strides() == stride_vector{10, 1});
    CHECK(v7.data() == v2.data() + 5);

    auto v8 = std::as_const(v2).back(1);
    CHECK(v8.lengths() == len_vector{4, 5});
    CHECK(v8.strides() == stride_vector{10, 1});
    CHECK(v8.data() == v2.data() + 5);
}

TEST_CASE("marray::access")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double,2> v1(marray_view<const double,2>({4, 3}, data.data()));

    CHECK(v1(0, 0) ==  0);
    CHECK(v1(1, 2) ==  5);
    CHECK(v1(3, 1) == 10);
    CHECK(std::as_const(v1)(3, 1) == 10);

    CHECK(v1[0][0] ==  0);
    CHECK(v1[1][2] ==  5);
    CHECK(v1[3][1] == 10);
    CHECK(std::as_const(v1)[3][1] == 10);

    auto v2 = view(v1(slice::all, range(2)));
    CHECK(v2.lengths() == array<len_type,2>{4, 2});
    CHECK(v2.strides() == array<stride_type,2>{3, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = view(v1(range(0, 4, 2), 1));
    CHECK(v3.lengths() == array<len_type,1>{2});
    CHECK(v3.strides() == array<stride_type,1>{6});
    CHECK(v3.data() == v1.data() + 1);

    auto v4 = view(v1[slice::all][range(2)]);
    CHECK(v4.lengths() == array<len_type,2>{4, 2});
    CHECK(v4.strides() == array<stride_type,2>{3, 1});
    CHECK(v4.data() == v1.data());

    auto v5 = view(v1[range(1,3)]);
    CHECK(v5.lengths() == array<len_type,2>{2,3});
    CHECK(v5.strides() == array<stride_type,2>{3,1});
    CHECK(v5.data() == v1.data() + 3);

    auto v6 = view(v1[2]);
    CHECK(v6.lengths() == array<len_type,1>{3});
    CHECK(v6.strides() == array<stride_type,1>{1});
    CHECK(v6.data() == v1.data() + 6);

    auto v7 = view(std::as_const(v1)(slice::all, range(2)));
    CHECK(v7.lengths() == array<len_type,2>{4, 2});
    CHECK(v7.strides() == array<stride_type,2>{3, 1});
    CHECK(v7.data() == v1.data());

    auto v8 = view(std::as_const(v1)(range(0, 4, 2), 1));
    CHECK(v8.lengths() == array<len_type,1>{2});
    CHECK(v8.strides() == array<stride_type,1>{6});
    CHECK(v8.data() == v1.data() + 1);

    auto v9 = view(std::as_const(v1)[slice::all][range(2)]);
    CHECK(v9.lengths() == array<len_type,2>{4, 2});
    CHECK(v9.strides() == array<stride_type,2>{3, 1});
    CHECK(v9.data() == v1.data());

    auto v10 = view(std::as_const(v1)[range(1,3)]);
    CHECK(v10.lengths() == array<len_type,2>{2,3});
    CHECK(v10.strides() == array<stride_type,2>{3,1});
    CHECK(v10.data() == v1.data() + 3);

    auto v11 = view(std::as_const(v1)[2]);
    CHECK(v11.lengths() == array<len_type,1>{3});
    CHECK(v11.strides() == array<stride_type,1>{1});
    CHECK(v11.data() == v1.data() + 6);
}

TEST_CASE("varray::access")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double> v1(marray_view<const double>({4, 3}, data.data()));

    CHECK(v1(0, 0) ==  0);
    CHECK(v1(1, 2) ==  5);
    CHECK(v1(3, 1) == 10);
    CHECK(std::as_const(v1)(3, 1) == 10);

    auto v2 = view<DYNAMIC>(v1(slice::all, range(2)));
    CHECK(v2.lengths() == len_vector{4, 2});
    CHECK(v2.strides() == stride_vector{3, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = view<DYNAMIC>(v1(range(0, 4, 2), 1));
    CHECK(v3.lengths() == len_vector{2});
    CHECK(v3.strides() == stride_vector{6});
    CHECK(v3.data() == v1.data() + 1);

    auto v4 = view<DYNAMIC>(std::as_const(v1)(slice::all, range(2)));
    CHECK(v4.lengths() == len_vector{4, 2});
    CHECK(v4.strides() == stride_vector{3, 1});
    CHECK(v4.data() == v1.data());

    auto v5 = view<DYNAMIC>(std::as_const(v1)(range(0, 4, 2), 1));
    CHECK(v5.lengths() == len_vector{2});
    CHECK(v5.strides() == stride_vector{6});
    CHECK(v5.data() == v1.data() + 1);
}

TEST_CASE("marray::iteration")
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray<double,2> v1 = {{{ 0, 1, 2},
                            { 3, 4, 5},
                            { 6, 7, 8},
                            { 9,10,11}}};
    const marray<double,2> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, len_type i, len_type j)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v2.for_each_element(
    [&](const double& v, len_type i, len_type j)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);
}

TEST_CASE("varray::iteration")
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray<double> v1{4, 3};
    copy_n(&data[0][0], 12, v1.data());
    const marray<double> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, const len_vector& pos)
    {
        CHECK(pos.size() == 2u);
        len_type i = pos[0];
        len_type j = pos[1];
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v2.for_each_element(
    [&](const double& v, const len_vector& pos)
    {
        CHECK(pos.size() == 2u);
        len_type i = pos[0];
        len_type j = pos[1];
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v1.for_each_element<2>(
    [&](double& v, len_type i, len_type j)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);

    visited = {};
    v2.for_each_element<2>(
    [&](const double& v, len_type i, len_type j)
    {
        CHECK(i >= 0);
        CHECK(i < 4);
        CHECK(j >= 0);
        CHECK(j < 3);
        CHECK(data[i][j] == v);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    for (len_type j = 0;j < 3;j++)
        CHECK(visited[i][j]);
}

TEST_CASE("marray::swap")
{
    marray<double,3> v1{4, 2, 5};
    marray<double,3> v2{3, 8, 3};

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v2.data() == data1);
    CHECK(v1.lengths() == array<len_type,3>{3, 8, 3});
    CHECK(v1.strides() == array<stride_type,3>{24, 3, 1});
    CHECK(v1.data() == data2);

    swap(v2, v1);

    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(v1.data() == data1);
    CHECK(v2.lengths() == array<len_type,3>{3, 8, 3});
    CHECK(v2.strides() == array<stride_type,3>{24, 3, 1});
    CHECK(v2.data() == data2);
}

TEST_CASE("varray::swap")
{
    marray<double> v1{4, 2, 5};
    marray<double> v2{3, 8};

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data() == data1);
    CHECK(v1.lengths() == len_vector{3, 8});
    CHECK(v1.strides() == stride_vector{8, 1});
    CHECK(v1.data() == data2);

    swap(v2, v1);

    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data() == data1);
    CHECK(v2.lengths() == len_vector{3, 8});
    CHECK(v2.strides() == stride_vector{8, 1});
    CHECK(v2.data() == data2);
}

TEST_CASE("marray::print")
{
    marray<double,3> v1 =
    {
     {
      {0, 1, 2},
      {3, 4, 5}
     },
     {
      {6, 7, 8},
      {9, 10, 11}
     }
    };

    std::string expected =
R"XXX({
 {
  {0, 1, 2},
  {3, 4, 5}
 },
 {
  {6, 7, 8},
  {9, 10, 11}
 }
})XXX";

    std::ostringstream oss;
    oss << v1;

    CHECK(oss.str() == expected);
}

TEST_CASE("varray::print")
{
    double data[2][2][3] =
    {
     {
      {0, 1, 2},
      {3, 4, 5}
     },
     {
      {6, 7, 8},
      {9, 10, 11}
     }
    };

    marray_view<double> v0({2,2,3}, (double*)data, ROW_MAJOR);
    marray<double> v1(v0, ROW_MAJOR);

    string expected =
R"XXX({
 {
  {0, 1, 2},
  {3, 4, 5}
 },
 {
  {6, 7, 8},
  {9, 10, 11}
 }
})XXX";

    ostringstream oss;
    oss << v1;

    CHECK(oss.str() == expected);
}
