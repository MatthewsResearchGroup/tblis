#include "index_iterator.hpp"

#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

TEST_CASE("miterator::next")
{
    len_type off1, off2;

    index_iterator<0,1> m1(vector<int>{}, vector<int>{});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 0);

    index_iterator<1,2> m2(vector<int>{5}, vector<int>{1}, vector<int>{2});

    off1 = 0;
    off2 = 0;
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 2);
    CHECK(off2 == 4);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 3);
    CHECK(off2 == 6);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 4);
    CHECK(off2 == 8);
    CHECK_FALSE(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);

    index_iterator<2,1> m3(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
    CHECK(m3.next(off1));
    CHECK(off1 == 2);
    CHECK(m3.next(off1));
    CHECK(off1 == 3);
    CHECK(m3.next(off1));
    CHECK(off1 == 4);
    CHECK(m3.next(off1));
    CHECK(off1 == 5);
    CHECK_FALSE(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);

    index_iterator<1,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
}

TEST_CASE("viterator::next")
{
    len_type off1, off2;

    index_iterator<DYNAMIC,1> m1(vector<int>{}, vector<int>{});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 0);

    index_iterator<DYNAMIC,2> m2(vector<int>{5}, vector<int>{1}, vector<int>{2});

    off1 = 0;
    off2 = 0;
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 2);
    CHECK(off2 == 4);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 3);
    CHECK(off2 == 6);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 4);
    CHECK(off2 == 8);
    CHECK_FALSE(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);

    index_iterator<DYNAMIC,1> m3(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
    CHECK(m3.next(off1));
    CHECK(off1 == 2);
    CHECK(m3.next(off1));
    CHECK(off1 == 3);
    CHECK(m3.next(off1));
    CHECK(off1 == 4);
    CHECK(m3.next(off1));
    CHECK(off1 == 5);
    CHECK_FALSE(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);

    index_iterator<DYNAMIC,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
}

TEST_CASE("miterator::reset")
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 1);
    CHECK(m1.next(off1));
    CHECK(off1 == 2);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);

    off1 = 0;
    m1.reset();
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 1);
    CHECK(m1.next(off1));
    CHECK(off1 == 2);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    index_iterator<1,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
}

TEST_CASE("viterator::reset")
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 1);
    CHECK(m1.next(off1));
    CHECK(off1 == 2);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);

    off1 = 0;
    m1.reset();
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 1);
    CHECK(m1.next(off1));
    CHECK(off1 == 2);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    index_iterator<DYNAMIC,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m4.next(off1));
    CHECK(off1 == 0);
}

TEST_CASE("miterator::position")
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m1.position(3, off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    off1 = 0;
    CHECK(m1.position(vector<int>{1,1}, off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    CHECK(m1.position() == array<len_type,2>{1,2});
}

TEST_CASE("viterator::position")
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    CHECK(m1.position(3, off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    off1 = 0;
    CHECK(m1.position(vector<int>{1,1}, off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 3);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 5);

    CHECK(m1.position() == len_vector{1,2});
}

TEST_CASE("miterator::assign")
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<2,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    CHECK(off1 == 3);

    off1 = 0;
    m2.position(vector<int>{1,1}, off1);
    CHECK(off1 == 4);

    m1 = m2;

    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 6);
    CHECK(m1.next(off1));
    CHECK(off1 == 7);
}

TEST_CASE("viterator::assign")
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<DYNAMIC,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    CHECK(off1 == 3);

    off1 = 0;
    m2.position(vector<int>{1,1}, off1);
    CHECK(off1 == 4);

    m1 = m2;

    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 4);
    CHECK(m1.next(off1));
    CHECK(off1 == 6);
    CHECK(m1.next(off1));
    CHECK(off1 == 7);
}

TEST_CASE("miterator::swap")
{
    len_type off1, off2;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<2,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    CHECK(off1 == 3);

    off2 = 0;
    m2.position(vector<int>{1,1}, off2);
    CHECK(off2 == 4);

    m1.swap(m2);

    CHECK(m1.next(off2));
    CHECK(off2 == 4);
    CHECK(m1.next(off2));
    CHECK(off2 == 6);
    CHECK(m1.next(off2));
    CHECK(off2 == 7);

    CHECK(m2.next(off1));
    CHECK(off1 == 3);
    CHECK(m2.next(off1));
    CHECK(off1 == 4);
    CHECK(m2.next(off1));
    CHECK(off1 == 5);
}

TEST_CASE("viterator::swap")
{
    len_type off1, off2;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<DYNAMIC,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    CHECK(off1 == 3);

    off2 = 0;
    m2.position(vector<int>{1,1}, off2);
    CHECK(off2 == 4);

    m1.swap(m2);

    CHECK(m1.next(off2));
    CHECK(off2 == 4);
    CHECK(m1.next(off2));
    CHECK(off2 == 6);
    CHECK(m1.next(off2));
    CHECK(off2 == 7);

    CHECK(m2.next(off1));
    CHECK(off1 == 3);
    CHECK(m2.next(off1));
    CHECK(off1 == 4);
    CHECK(m2.next(off1));
    CHECK(off1 == 5);
}

TEST_CASE("miterator::make_iterator")
{
    len_type off1, off2;

    auto m1 = make_iterator(array<len_type,0>{}, array<stride_type,0>{});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 0);

    auto m2 = make_iterator(array<len_type,1>{5}, array<stride_type,1>{1}, array<stride_type,1>{2});

    off1 = 0;
    off2 = 0;
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 2);
    CHECK(off2 == 4);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 3);
    CHECK(off2 == 6);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 4);
    CHECK(off2 == 8);
    CHECK_FALSE(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);

    auto m3 = make_iterator(array<len_type,2>{2,3}, array<stride_type,2>{1,2});

    off1 = 0;
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
    CHECK(m3.next(off1));
    CHECK(off1 == 2);
    CHECK(m3.next(off1));
    CHECK(off1 == 3);
    CHECK(m3.next(off1));
    CHECK(off1 == 4);
    CHECK(m3.next(off1));
    CHECK(off1 == 5);
    CHECK_FALSE(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
}

TEST_CASE("viterator::make_iterator")
{
    len_type off1, off2;

    auto m1 = make_iterator(vector<len_type>{}, vector<stride_type>{});

    off1 = 0;
    CHECK(m1.next(off1));
    CHECK(off1 == 0);
    CHECK_FALSE(m1.next(off1));
    CHECK(off1 == 0);
    CHECK(m1.next(off1));
    CHECK(off1 == 0);

    auto m2 = make_iterator(vector<len_type>{5}, vector<stride_type>{1}, vector<stride_type>{2});

    off1 = 0;
    off2 = 0;
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 2);
    CHECK(off2 == 4);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 3);
    CHECK(off2 == 6);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 4);
    CHECK(off2 == 8);
    CHECK_FALSE(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 0);
    CHECK(off2 == 0);
    CHECK(m2.next(off1, off2));
    CHECK(off1 == 1);
    CHECK(off2 == 2);

    auto m3 = make_iterator(vector<len_type>{2,3}, vector<stride_type>{1,2});

    off1 = 0;
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
    CHECK(m3.next(off1));
    CHECK(off1 == 2);
    CHECK(m3.next(off1));
    CHECK(off1 == 3);
    CHECK(m3.next(off1));
    CHECK(off1 == 4);
    CHECK(m3.next(off1));
    CHECK(off1 == 5);
    CHECK_FALSE(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 0);
    CHECK(m3.next(off1));
    CHECK(off1 == 1);
}
