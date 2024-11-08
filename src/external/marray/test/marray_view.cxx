#include "marray.hpp"
#include "rotate.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;
using namespace MArray::slice;

TEST_CASE("marray_view::constructor")
{
    double tmp;
    double* data = &tmp;
    std::vector<double> x{1, 2, 3};

    marray<double,3> v0{4, 2, 5};
    marray<double,4> v01{3, 4, 2, 5};

    marray_view<double,2> v1;
    CHECK(v1.data() == nullptr);
    CHECK(v1.lengths() == array<len_type,2>{0, 0});
    CHECK(v1.strides() == array<stride_type,2>{0, 0});

    marray_view<double,3> v2({4, 2, 5}, data);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v3(array<char,3>{4, 2, 5}, data);
    CHECK(v3.data() == data);
    CHECK(v3.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v3.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v4({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v4.data() == data);
    CHECK(v4.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v4.strides() == array<stride_type,3>{1, 4, 8});

    marray_view<double,3> v5(v2);
    CHECK(v5.data() == data);
    CHECK(v5.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v5.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v51(v0);
    CHECK(v51.data() == v0.data());
    CHECK(v51.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v51.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v52(v01[0]);
    CHECK(v52.data() == v01.data());
    CHECK(v52.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v52.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v6(marray_view<double,3>({4, 2, 5}, data));
    CHECK(v6.data() == data);
    CHECK(v6.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v6.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<double,3> v7({4, 2, 5}, data, {3, 8, 24});
    CHECK(v7.data() == data);
    CHECK(v7.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v7.strides() == array<stride_type,3>{3, 8, 24});

    marray_view<double,3> v8(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    CHECK(v8.data() == data);
    CHECK(v8.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v8.strides() == array<stride_type,3>{3, 8, 24});

    marray_view<const double,2> v9;
    CHECK(v9.data() == nullptr);
    CHECK(v9.lengths() == array<len_type,2>{0, 0});
    CHECK(v9.strides() == array<stride_type,2>{0, 0});

    marray_view<const double,3> v10({4, 2, 5}, data);
    CHECK(v10.data() == data);
    CHECK(v10.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v10.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v11(array<char,3>{4, 2, 5}, data);
    CHECK(v11.data() == data);
    CHECK(v11.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v11.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v12({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v12.data() == data);
    CHECK(v12.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v12.strides() == array<stride_type,3>{1, 4, 8});

    marray_view<const double,3> v13(v2);
    CHECK(v13.data() == data);
    CHECK(v13.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v13.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v131(v0);
    CHECK(v131.data() == v0.data());
    CHECK(v131.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v131.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v132(v01[0]);
    CHECK(v132.data() == v01.data());
    CHECK(v132.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v132.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v14(v10);
    CHECK(v14.data() == data);
    CHECK(v14.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v14.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v15(marray_view<double,3>({4, 2, 5}, data));
    CHECK(v15.data() == data);
    CHECK(v15.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v15.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v16(marray_view<const double,3>({4, 2, 5}, data));
    CHECK(v16.data() == data);
    CHECK(v16.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v16.strides() == array<stride_type,3>{10, 5, 1});

    marray_view<const double,3> v17({4, 2, 5}, data, {3, 8, 24});
    CHECK(v17.data() == data);
    CHECK(v17.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v17.strides() == array<stride_type,3>{3, 8, 24});

    marray_view<const double,3> v18(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    CHECK(v18.data() == data);
    CHECK(v18.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v18.strides() == array<stride_type,3>{3, 8, 24});

    marray_view<double,1> v19(x);
    CHECK(v19.data() == x.data());
    CHECK(v19.lengths() == array<len_type,1>{3});
    CHECK(v19.strides() == array<stride_type,1>{1});

    marray_view<const double,1> v20(std::as_const(x));
    CHECK(v20.data() == x.data());
    CHECK(v20.lengths() == array<len_type,1>{3});
    CHECK(v20.strides() == array<stride_type,1>{1});
}

TEST_CASE("varray_view::constructor")
{
    double tmp;
    double* data = &tmp;
    std::vector<double> x{1, 2, 3};

    marray<double> v0{4, 2, 5};

    marray_view<double> v1;
    CHECK(v1.dimension() == 0u);
    CHECK(v1.data() == nullptr);

    marray_view<double> v2({4, 2, 5}, data);
    CHECK(v2.dimension() == 3u);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});

    marray_view<double> v3(vector<char>{4, 2, 5}, data);
    CHECK(v3.dimension() == 3u);
    CHECK(v3.data() == data);
    CHECK(v3.lengths() == len_vector{4, 2, 5});
    CHECK(v3.strides() == stride_vector{10, 5, 1});

    marray_view<double> v4({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v4.dimension() == 3u);
    CHECK(v4.data() == data);
    CHECK(v4.lengths() == len_vector{4, 2, 5});
    CHECK(v4.strides() == stride_vector{1, 4, 8});

    marray_view<double> v5(v2);
    CHECK(v5.dimension() == 3u);
    CHECK(v5.data() == data);
    CHECK(v5.lengths() == len_vector{4, 2, 5});
    CHECK(v5.strides() == stride_vector{10, 5, 1});

    marray_view<double> v51(v0);
    CHECK(v51.dimension() == 3u);
    CHECK(v51.data() == v0.data());
    CHECK(v51.lengths() == len_vector{4, 2, 5});
    CHECK(v51.strides() == stride_vector{10, 5, 1});

    marray_view<double> v6(marray_view<double>({4, 2, 5}, data));
    CHECK(v6.dimension() == 3u);
    CHECK(v6.data() == data);
    CHECK(v6.lengths() == len_vector{4, 2, 5});
    CHECK(v6.strides() == stride_vector{10, 5, 1});

    marray_view<double> v7({4, 2, 5}, data, {3, 8, 24});
    CHECK(v7.dimension() == 3u);
    CHECK(v7.data() == data);
    CHECK(v7.lengths() == len_vector{4, 2, 5});
    CHECK(v7.strides() == stride_vector{3, 8, 24});

    marray_view<double> v8(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    CHECK(v8.dimension() == 3u);
    CHECK(v8.data() == data);
    CHECK(v8.lengths() == len_vector{4, 2, 5});
    CHECK(v8.strides() == stride_vector{3, 8, 24});

    marray_view<const double> v9;
    CHECK(v9.dimension() == 0u);
    CHECK(v9.data() == nullptr);

    marray_view<const double> v10({4, 2, 5}, data);
    CHECK(v10.dimension() == 3u);
    CHECK(v10.data() == data);
    CHECK(v10.lengths() == len_vector{4, 2, 5});
    CHECK(v10.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v11(vector<char>{4, 2, 5}, data);
    CHECK(v11.dimension() == 3u);
    CHECK(v11.data() == data);
    CHECK(v11.lengths() == len_vector{4, 2, 5});
    CHECK(v11.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v12({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v12.dimension() == 3u);
    CHECK(v12.data() == data);
    CHECK(v12.lengths() == len_vector{4, 2, 5});
    CHECK(v12.strides() == stride_vector{1, 4, 8});

    marray_view<const double> v13(v2);
    CHECK(v13.dimension() == 3u);
    CHECK(v13.data() == data);
    CHECK(v13.lengths() == len_vector{4, 2, 5});
    CHECK(v13.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v131(v0);
    CHECK(v131.dimension() == 3u);
    CHECK(v131.data() == v0.data());
    CHECK(v131.lengths() == len_vector{4, 2, 5});
    CHECK(v131.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v14(v10);
    CHECK(v14.dimension() == 3u);
    CHECK(v14.data() == data);
    CHECK(v14.lengths() == len_vector{4, 2, 5});
    CHECK(v14.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v15(marray_view<double>({4, 2, 5}, data));
    CHECK(v15.dimension() == 3u);
    CHECK(v15.data() == data);
    CHECK(v15.lengths() == len_vector{4, 2, 5});
    CHECK(v15.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v16(marray_view<const double>({4, 2, 5}, data));
    CHECK(v16.dimension() == 3u);
    CHECK(v16.data() == data);
    CHECK(v16.lengths() == len_vector{4, 2, 5});
    CHECK(v16.strides() == stride_vector{10, 5, 1});

    marray_view<const double> v17({4, 2, 5}, data, {3, 8, 24});
    CHECK(v17.dimension() == 3u);
    CHECK(v17.data() == data);
    CHECK(v17.lengths() == len_vector{4, 2, 5});
    CHECK(v17.strides() == stride_vector{3, 8, 24});

    marray_view<const double> v18(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    CHECK(v18.dimension() == 3u);
    CHECK(v18.data() == data);
    CHECK(v18.lengths() == len_vector{4, 2, 5});
    CHECK(v18.strides() == stride_vector{3, 8, 24});

    marray_view<double> v19(x);
    CHECK(v19.dimension() == 1u);
    CHECK(v19.data() == x.data());
    CHECK(v19.lengths() == len_vector{3});
    CHECK(v19.strides() == stride_vector{1});

    marray_view<const double> v20(std::as_const(x));
    CHECK(v20.dimension() == 1u);
    CHECK(v20.data() == x.data());
    CHECK(v20.lengths() == len_vector{3});
    CHECK(v20.strides() == stride_vector{1});
}

TEST_CASE("marray_view::reset")
{
    double tmp;
    double* data = &tmp;
    std::vector<double> x{1, 2, 3};

    marray_view<double,3> v1;
    marray_view<const double,3> v2;
    marray_view<double,3> v3({4, 2, 5}, data);
    marray_view<const double,3> v4({4, 2, 5}, data);
    marray_view<double,1> v5;
    marray_view<const double,1> v6;
    marray<double,3> v0{4, 2, 5};
    marray<double,4> v01{3, 4, 2, 5};

    v1.reset({4, 2, 5}, data);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(array<char,3>{4, 2, 5}, data);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{1, 4, 8});

    v1.reset(v3);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v0);
    CHECK(v1.data() == v0.data());
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(v01[0]);
    CHECK(v1.data() == v01.data());
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset(marray_view<double,3>({4, 2, 5}, data));
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{10, 5, 1});

    v1.reset({4, 2, 5}, data, {3, 8, 24});
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{3, 8, 24});

    v1.reset(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v1.strides() == array<stride_type,3>{3, 8, 24});

    v1.reset();
    CHECK(v1.data() == nullptr);
    CHECK(v1.lengths() == array<len_type,3>{0, 0, 0});
    CHECK(v1.strides() == array<stride_type,3>{0, 0, 0});

    v2.reset({4, 2, 5}, data);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(array<char,3>{4, 2, 5}, data);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{1, 4, 8});

    v2.reset(v3);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(v0);
    CHECK(v2.data() == v0.data());
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(v01[0]);
    CHECK(v2.data() == v01.data());
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(v4);
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(marray_view<double,3>({4, 2, 5}, data));
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset(marray_view<const double,3>({4, 2, 5}, data));
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{10, 5, 1});

    v2.reset({4, 2, 5}, data, {3, 8, 24});
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{3, 8, 24});

    v2.reset(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    CHECK(v2.data() == data);
    CHECK(v2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(v2.strides() == array<stride_type,3>{3, 8, 24});

    v2.reset();
    CHECK(v1.data() == nullptr);
    CHECK(v2.lengths() == array<len_type,3>{0, 0, 0});
    CHECK(v2.strides() == array<stride_type,3>{0, 0, 0});

    v5.reset(x);
    CHECK(v5.data() == x.data());
    CHECK(v5.lengths() == array<len_type,1>{3});
    CHECK(v5.strides() == array<stride_type,1>{1});

    v6.reset(x);
    CHECK(v6.data() == x.data());
    CHECK(v6.lengths() == array<len_type,1>{3});
    CHECK(v6.strides() == array<stride_type,1>{1});
}

TEST_CASE("varray_view::reset")
{
    double tmp;
    double* data = &tmp;
    std::vector<double> x{1, 2, 3};

    marray_view<double> v1;
    marray_view<double> v2({4, 2, 5}, data);
    marray<double> v0{4, 2, 5};

    v1.reset({4, 2, 5}, data);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(vector<char>{4, 2, 5}, data);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{1, 4, 8});

    v1.reset(v2);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(v0);
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == v0.data());
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset(marray_view<double>({4, 2, 5}, data));
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});

    v1.reset({4, 2, 5}, data, {3, 8, 24});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{3, 8, 24});

    v1.reset(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    CHECK(v1.dimension() == 3u);
    CHECK(v1.data() == data);
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{3, 8, 24});

    v1.reset();
    CHECK(v1.dimension() == 0u);
    CHECK(v1.data() == nullptr);

    v1.reset(x);
    CHECK(v1.dimension() == 1u);
    CHECK(v1.data() == x.data());
    CHECK(v1.lengths() == len_vector{3});
    CHECK(v1.strides() == stride_vector{1});

    marray_view<const double> v9;
    marray_view<const double> v10({4, 2, 5}, data);

    CHECK(v9.dimension() == 0u);
    CHECK(v9.data() == nullptr);

    v9.reset({4, 2, 5}, data);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset(vector<char>{4, 2, 5}, data);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset({4, 2, 5}, data, COLUMN_MAJOR);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{1, 4, 8});

    v9.reset(v2);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset(v0);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == v0.data());
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset(v10);
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset(marray_view<double>({4, 2, 5}, data));
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset(marray_view<const double>({4, 2, 5}, data));
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{10, 5, 1});

    v9.reset({4, 2, 5}, data, {3, 8, 24});
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{3, 8, 24});

    v9.reset(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    CHECK(v9.dimension() == 3u);
    CHECK(v9.data() == data);
    CHECK(v9.lengths() == len_vector{4, 2, 5});
    CHECK(v9.strides() == stride_vector{3, 8, 24});

    v9.reset();
    CHECK(v9.dimension() == 0u);
    CHECK(v9.data() == nullptr);

    v9.reset(std::as_const(x));
    CHECK(v9.dimension() == 1u);
    CHECK(v9.data() == x.data());
    CHECK(v9.lengths() == len_vector{3});
    CHECK(v9.strides() == stride_vector{1});
}

TEST_CASE("marray_view::initialize")
{
    std::array<double,18> data = {};

    marray_view<double,3> v1({3, 2, 3}, data.data(), ROW_MAJOR);
    marray_view<double,3> v2({3, 2, 3}, data.data(), COLUMN_MAJOR);

    v1 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    CHECK(data == std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                         6, 7, 8, 9,10,11,
                                        12,13,14,15,16,17});

    v2 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    CHECK(data == std::array<double,18>{ 0, 6,12, 3, 9,15,
                                         1, 7,13, 4,10,16,
                                         2, 8,14, 5,11,17});
}

TEST_CASE("marray_view::assign")
{
    double data1[6] = {0, 1, 2,
                       3, 4, 5};
    int data2[6] = {0, 1, 2,
                    3, 4, 5};

    marray<double> v0{2, 3};
    marray_view<double> v1(v0);

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

TEST_CASE("marray_view::shift")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1_({4, 6, 5}, data);
    marray_view<double,3> v1 = v1_[range(2)][range(3,6)][range(2)];

    auto v2 = v1.shifted(0, 2);
    CHECK(v2.lengths() == v1.lengths());
    CHECK(v2.strides() == v1.strides());
    CHECK(v2.data() == v1.data() + 2*v1.stride(0));

    auto v3 = v1.shifted(1, -3);
    CHECK(v3.lengths() == v1.lengths());
    CHECK(v3.strides() == v1.strides());
    CHECK(v3.data() == v1.data() - 3*v1.stride(1));

    auto v4 = v1.shifted_down(2);
    CHECK(v4.lengths() == v1.lengths());
    CHECK(v4.strides() == v1.strides());
    CHECK(v4.data() == v1.data() + v1.length(2)*v1.stride(2));

    auto v5 = v2.shifted_up(0);
    CHECK(v5.lengths() == v1.lengths());
    CHECK(v5.strides() == v1.strides());
    CHECK(v5.data() == v2.data() - v1.length(0)*v1.stride(0));

    auto ref = v1.data();
    v1.shift(0, 2);
    CHECK(v1.data() == ref + 2*v1.stride(0));

    ref = v1.data();
    v1.shift(1, -3);
    CHECK(v1.data() == ref - 3*v1.stride(1));

    ref = v1.data();
    v1.shift_down(2);
    CHECK(v1.data() == ref + v1.length(2)*v1.stride(2));

    ref = v1.data();
    v1.shift_up(0);
    CHECK(v1.data() == ref - v1.length(0)*v1.stride(0));
}

TEST_CASE("varray_view::shift")
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1_({4, 6, 5}, data);
    marray_view<double> v1 = v1_(range(2),range(3,6),range(2));

    auto v2 = v1.shifted(0, 2);
    CHECK(v2.lengths() == v1.lengths());
    CHECK(v2.strides() == v1.strides());
    CHECK(v2.data() == v1.data() + 2*v1.stride(0));

    auto v3 = v1.shifted(1, -3);
    CHECK(v3.lengths() == v1.lengths());
    CHECK(v3.strides() == v1.strides());
    CHECK(v3.data() == v1.data() - 3*v1.stride(1));

    auto v4 = v1.shifted_down(2);
    CHECK(v4.lengths() == v1.lengths());
    CHECK(v4.strides() == v1.strides());
    CHECK(v4.data() == v1.data() + v1.length(2)*v1.stride(2));

    auto v5 = v2.shifted_up(0);
    CHECK(v5.lengths() == v1.lengths());
    CHECK(v5.strides() == v1.strides());
    CHECK(v5.data() == v2.data() - v1.length(0)*v1.stride(0));

    auto ref = v1.data();
    v1.shift(0, 2);
    CHECK(v1.data() == ref + 2*v1.stride(0));

    ref = v1.data();
    v1.shift(1, -3);
    CHECK(v1.data() == ref - 3*v1.stride(1));

    ref = v1.data();
    v1.shift_down(2);
    CHECK(v1.data() == ref + v1.length(2)*v1.stride(2));

    ref = v1.data();
    v1.shift_up(0);
    CHECK(v1.data() == ref - v1.length(0)*v1.stride(0));
}

TEST_CASE("marray_view::permute")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1({4, 2, 5}, data);

    auto v2 = v1.permuted({1, 0, 2});
    CHECK(v2.lengths() == array<len_type,3>{2, 4, 5});
    CHECK(v2.strides() == array<stride_type,3>{5, 10, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.permuted(array<char,3>{2, 0, 1});
    CHECK(v3.lengths() == array<len_type,3>{5, 4, 2});
    CHECK(v3.strides() == array<stride_type,3>{1, 10, 5});
    CHECK(v3.data() == v1.data());

    v1.permute({1, 0, 2});
    CHECK(v1.lengths() == array<len_type,3>{2, 4, 5});
    CHECK(v1.strides() == array<stride_type,3>{5, 10, 1});
    CHECK(v1.data() == data);

    v1.permute(array<char,3>{2, 0, 1});
    CHECK(v1.lengths() == array<len_type,3>{5, 2, 4});
    CHECK(v1.strides() == array<stride_type,3>{1, 5, 10});
    CHECK(v1.data() == data);

    v1.permute(0, 2, 1);
    CHECK(v1.lengths() == array<len_type,3>{5, 4, 2});
    CHECK(v1.strides() == array<stride_type,3>{1, 10, 5});
    CHECK(v1.data() == data);
}

TEST_CASE("varray_view::permute")
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1({4, 2, 5}, data);

    auto v2 = v1.permuted({1, 0, 2});
    CHECK(v2.lengths() == len_vector{2, 4, 5});
    CHECK(v2.strides() == stride_vector{5, 10, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.permuted(vector<char>{2, 0, 1});
    CHECK(v3.lengths() == len_vector{5, 4, 2});
    CHECK(v3.strides() == stride_vector{1, 10, 5});
    CHECK(v3.data() == v1.data());

    v1.permute({1, 0, 2});
    CHECK(v1.lengths() == len_vector{2, 4, 5});
    CHECK(v1.strides() == stride_vector{5, 10, 1});
    CHECK(v1.data() == data);

    v1.permute(vector<char>{2, 0, 1});
    CHECK(v1.lengths() == len_vector{5, 2, 4});
    CHECK(v1.strides() == stride_vector{1, 5, 10});
    CHECK(v1.data() == data);

    v1.permute(0, 2, 1);
    CHECK(v1.lengths() == len_vector{5, 4, 2});
    CHECK(v1.strides() == stride_vector{1, 10, 5});
    CHECK(v1.data() == data);
}

TEST_CASE("marray_view::transpose")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,2> v1({4, 8}, data);

    auto v2 = v1.transposed();
    CHECK(v2.lengths() == array<len_type,2>{8, 4});
    CHECK(v2.strides() == array<stride_type,2>{1, 8});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.T();
    CHECK(v3.lengths() == array<len_type,2>{8, 4});
    CHECK(v3.strides() == array<stride_type,2>{1, 8});
    CHECK(v3.data() == v1.data());

    v1.transpose();
    CHECK(v1.lengths() == array<len_type,2>{8, 4});
    CHECK(v1.strides() == array<stride_type,2>{1, 8});
    CHECK(v1.data() == data);
}

TEST_CASE("marray_view::lowered")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1({4, 2, 5}, data);

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
}

TEST_CASE("marray_view::lower")
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1({4, 2, 5}, data);

    auto v2 = v1.lowered({1, 2});
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = v1.lowered(vector<char>{1});
    CHECK(v3.lengths() == len_vector{4, 10});
    CHECK(v3.strides() == stride_vector{10, 1});
    CHECK(v3.data() == v1.data());

    auto v4 = v1.lowered({});
    CHECK(v4.lengths() == len_vector{40});
    CHECK(v4.strides() == stride_vector{1});
    CHECK(v4.data() == v1.data());

    v1.lower({1, 2});
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data() == data);

    v1.lower(vector<char>{1});
    CHECK(v1.lengths() == len_vector{4, 10});
    CHECK(v1.strides() == stride_vector{10, 1});
    CHECK(v1.data() == data);

    v1.lower({});
    CHECK(v1.lengths() == len_vector{40});
    CHECK(v1.strides() == stride_vector{1});
    CHECK(v1.data() == data);
}

TEST_CASE("marray_view::fix")
{
    marray<double> v1{4, 2, 5};
    marray<double> v2_{4, 2, 5};
    marray_view<double> v2(v2_);

    auto m1 = v1.view<3>();
    CHECK(m1.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(m1.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(m1.data() == v1.data());

    auto m2 = v2.view<3>();
    CHECK(m2.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(m2.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(m2.data() == v2.data());

    auto m3 = view<3>(v1);
    CHECK(m3.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(m3.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(m3.data() == v1.data());

    auto m4= view<3>(v2);
    CHECK(m4.lengths() == array<len_type,3>{4, 2, 5});
    CHECK(m4.strides() == array<stride_type,3>{10, 5, 1});
    CHECK(m4.data() == v2.data());
}

TEST_CASE("marray_view::vary")
{
    marray<double,3> m1{4, 2, 5};
    marray<double,3> m2_{4, 2, 5};
    marray_view<double,3> m2(m2_);

    auto v1 = m1.view<DYNAMIC>();
    CHECK(v1.lengths() == len_vector{4, 2, 5});
    CHECK(v1.strides() == stride_vector{10, 5, 1});
    CHECK(v1.data() == m1.data());

    auto v2 = m2.view<DYNAMIC>();
    CHECK(v2.lengths() == len_vector{4, 2, 5});
    CHECK(v2.strides() == stride_vector{10, 5, 1});
    CHECK(v2.data() == m2.data());

    auto v3 = m2[0].view<DYNAMIC>();
    CHECK(v3.lengths() == len_vector{2, 5});
    CHECK(v3.strides() == stride_vector{5, 1});
    CHECK(v3.data() == m2.data());

    auto v4 = m2[slice::all].view<DYNAMIC>();
    CHECK(v4.lengths() == len_vector{4, 2, 5});
    CHECK(v4.strides() == stride_vector{10, 5, 1});
    CHECK(v4.data() == m2.data());

    auto v5 = view<DYNAMIC>(m1);
    CHECK(v5.lengths() == len_vector{4, 2, 5});
    CHECK(v5.strides() == stride_vector{10, 5, 1});
    CHECK(v5.data() == m1.data());

    auto v6 = view<DYNAMIC>(m2);
    CHECK(v6.lengths() == len_vector{4, 2, 5});
    CHECK(v6.strides() == stride_vector{10, 5, 1});
    CHECK(v6.data() == m2.data());

    auto v7 = view<DYNAMIC>(m2[0]);
    CHECK(v7.lengths() == len_vector{2, 5});
    CHECK(v7.strides() == stride_vector{5, 1});
    CHECK(v7.data() == m2.data());

    auto v8 = view<DYNAMIC>(m2[slice::all]);
    CHECK(v8.lengths() == len_vector{4, 2, 5});
    CHECK(v8.strides() == stride_vector{10, 5, 1});
    CHECK(v8.data() == m2.data());
}

TEST_CASE("marray_view::rotate")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double,2> v1({4, 3}, data.data());

    rotate(v1, 1, 1);
    CHECK(data == array<double,12>{ 1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6,
                                   10,11, 9});

    rotate(v1, 0, -1);
    CHECK(data == array<double,12>{10,11, 9,
                                    1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6});

    rotate(v1, {4,3});
    CHECK(data == array<double,12>{10,11, 9,
                                    1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6});

    rotate(v1, array<char,2>{1,1});
    CHECK(data == array<double,12>{ 2, 0, 1,
                                    5, 3, 4,
                                    8, 6, 7,
                                   11, 9,10});
}

TEST_CASE("varray_view::rotate")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double> v1({4, 3}, data.data());

    rotate(v1, 1, 1);
    CHECK(data == array<double,12>{ 1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6,
                                   10,11, 9});

    rotate(v1, 0, -1);
    CHECK(data == array<double,12>{10,11, 9,
                                    1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6});

    rotate(v1, {4,3});
    CHECK(data == array<double,12>{10,11, 9,
                                    1, 2, 0,
                                    4, 5, 3,
                                    7, 8, 6});

    rotate(v1, vector<char>{1,1});
    CHECK(data == array<double,12>{ 2, 0, 1,
                                    5, 3, 4,
                                    8, 6, 7,
                                   11, 9,10});
}

TEST_CASE("marray_view::front_back")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,1> v1({8}, data);

    CHECK(&v1.cfront() == data);
    CHECK(&v1.front() == data);
    CHECK(&v1.cback() == data+7);
    CHECK(&v1.back() == data+7);

    marray_view<double,3> v2({4, 2, 5}, data);

    auto v3 = v2.cfront(0);
    CHECK(v3.lengths() == array<len_type,2>{2, 5});
    CHECK(v3.strides() == array<stride_type,2>{5, 1});
    CHECK(v3.data() == data);

    auto v4 = v2.front(1);
    CHECK(v4.lengths() == array<len_type,2>{4, 5});
    CHECK(v4.strides() == array<stride_type,2>{10, 1});
    CHECK(v4.data() == data);

    auto v5 = v2.cback(0);
    CHECK(v5.lengths() == array<len_type,2>{2, 5});
    CHECK(v5.strides() == array<stride_type,2>{5, 1});
    CHECK(v5.data() == data + 30);

    auto v6 = v2.back(1);
    CHECK(v6.lengths() == array<len_type,2>{4, 5});
    CHECK(v6.strides() == array<stride_type,2>{10, 1});
    CHECK(v6.data() == data + 5);
}

TEST_CASE("varray_view::front_back")
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v2({4, 2, 5}, data);

    auto v3 = v2.cfront(0);
    CHECK(v3.lengths() == len_vector{2, 5});
    CHECK(v3.strides() == stride_vector{5, 1});
    CHECK(v3.data() == data);

    auto v4 = v2.front(1);
    CHECK(v4.lengths() == len_vector{4, 5});
    CHECK(v4.strides() == stride_vector{10, 1});
    CHECK(v4.data() == data);

    auto v5 = v2.cback(0);
    CHECK(v5.lengths() == len_vector{2, 5});
    CHECK(v5.strides() == stride_vector{5, 1});
    CHECK(v5.data() == data + 30);

    auto v6 = v2.back(1);
    CHECK(v6.lengths() == len_vector{4, 5});
    CHECK(v6.strides() == stride_vector{10, 1});
    CHECK(v6.data() == data + 5);
}

TEST_CASE("marray_view::access")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double,2> v1({4, 3}, data.data());

    CHECK(v1(0, 0) ==  0);
    CHECK(v1(1, 2) ==  5);
    CHECK(v1(3, 1) == 10);

    CHECK(v1[0][0] ==  0);
    CHECK(v1[1][2] ==  5);
    CHECK(v1[3][1] == 10);

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
}

TEST_CASE("varray_view::access")
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double> v1({4, 3}, data.data());

    CHECK(v1(0, 0) ==  0);
    CHECK(v1(1, 2) ==  5);
    CHECK(v1(3, 1) == 10);

    auto v2 = view<DYNAMIC>(v1(slice::all, range(2)));
    CHECK(v2.lengths() == len_vector{4, 2});
    CHECK(v2.strides() == stride_vector{3, 1});
    CHECK(v2.data() == v1.data());

    auto v3 = view<DYNAMIC>(v1(range(0, 4, 2), 1));
    CHECK(v3.lengths() == len_vector{2});
    CHECK(v3.strides() == stride_vector{6});
    CHECK(v3.data() == v1.data() + 1);
}

TEST_CASE("marray_view::iteration")
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray_view<double,2> v1({4, 3}, &data[0][0]);
    marray_view<const double,2> v2({4, 3}, &data[0][0]);

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

TEST_CASE("varray_view::iteration")
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray_view<double> v1({4, 3}, &data[0][0]);
    marray_view<const double> v2({4, 3}, &data[0][0]);

    visited = {};
    v1.for_each_element(
    [&](double& v, const len_vector& pos)
    {
        CHECK(2u == pos.size());
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
        CHECK(2u == pos.size());
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

TEST_CASE("marray_view::length_stride")
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1_({100, 2, 6}, data);
    marray_view<double,3> v1 = v1_[range(4)][all][all];

    CHECK(v1.length(0) == 4);
    CHECK(v1.length(0, 2) == 4);
    CHECK(v1.length(0) == 2);

    CHECK(v1.length(2) == 6);
    CHECK(v1.length(2, 3) == 6);
    CHECK(v1.length(2) == 3);

    CHECK(v1.stride(0) == 12);
    CHECK(v1.stride(0, 24) == 12);
    CHECK(v1.stride(0) == 24);

    CHECK(v1.stride(2) == 1);
    CHECK(v1.stride(2, 2) == 1);
    CHECK(v1.stride(2) == 2);

    CHECK(v1.lengths() == array<len_type,3>{2, 2, 3});
    CHECK(v1.strides() == array<stride_type,3>{24, 6, 2});
}

TEST_CASE("varray_view::length_stride")
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1_({100, 2, 6}, data);
    marray_view<double> v1 = v1_(range(4),all,all);

    CHECK(v1.length(0) == 4);
    CHECK(v1.length(0, 2) == 4);
    CHECK(v1.length(0) == 2);

    CHECK(v1.length(2) == 6);
    CHECK(v1.length(2, 3) == 6);
    CHECK(v1.length(2) == 3);

    CHECK(v1.stride(0) == 12);
    CHECK(v1.stride(0, 24) == 12);
    CHECK(v1.stride(0) == 24);

    CHECK(v1.stride(2) == 1);
    CHECK(v1.stride(2, 2) == 1);
    CHECK(v1.stride(2) == 2);

    CHECK(v1.lengths() == len_vector{2, 2, 3});
    CHECK(v1.strides() == stride_vector{24, 6, 2});
}

TEST_CASE("marray_view::swap")
{
    double tmp1, tmp2;
    double* data1 = &tmp1;
    double* data2 = &tmp2;

    marray_view<double,3> v1({4, 2, 5}, data1);
    marray_view<double,3> v2({3, 8, 3}, data2);

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

TEST_CASE("varray_view::swap")
{
    double tmp1, tmp2;
    double* data1 = &tmp1;
    double* data2 = &tmp2;

    marray_view<double> v1({4, 2, 5}, data1);
    marray_view<double> v2({3, 8}, data2);

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

TEST_CASE("marray_view::print")
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

    marray_view<double,3> v1({2,2,3}, (double*)data, ROW_MAJOR);

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

TEST_CASE("varray_view::print")
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

    marray_view<double> v1({2,2,3}, (double*)data, ROW_MAJOR);

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
