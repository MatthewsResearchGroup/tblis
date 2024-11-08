#include "range.hpp"
#include "catch_amalgamated.hpp"

using namespace std;
using namespace MArray;

TEST_CASE("range::iterator")
{
    range_t<int>::iterator it1;
    range_t<int>::iterator it2(0, 2);
    range_t<int>::iterator it3(5, 1);

    CHECK(it1 == it1);
    CHECK(it2 == it2);
    CHECK(it3 == it3);
    CHECK(it1 != it2);
    CHECK(it1 != it3);
    CHECK(it2 != it3);
    CHECK(*it1 == 0);
    CHECK(*it2 == 0);
    CHECK(*it3 == 5);
    ++it1;
    ++it2;
    ++it3;
    CHECK(*it1 == 0);
    CHECK(*it2 == 2);
    CHECK(*it3 == 6);
    CHECK(*it1++ == 0);
    CHECK(*it2++ == 2);
    CHECK(*it3++ == 6);
    CHECK(*it1 == 0);
    CHECK(*it2 == 4);
    CHECK(*it3 == 7);
    --it1;
    --it2;
    --it3;
    CHECK(*it1 == 0);
    CHECK(*it2 == 2);
    CHECK(*it3 == 6);
    CHECK(*it1-- == 0);
    CHECK(*it2-- == 2);
    CHECK(*it3-- == 6);
    CHECK(*it1 == 0);
    CHECK(*it2 == 0);
    CHECK(*it3 == 5);
    CHECK(*(it1 += 2) == 0);
    CHECK(*(it2 += 2) == 4);
    CHECK(*(it3 += 2) == 7);
    CHECK(*(it1 -= 3) == 0);
    CHECK(*(it2 -= 3) == -2);
    CHECK(*(it3 -= 3) == 4);
    CHECK(*(it1 + 1) == 0);
    CHECK(*(it2 + 1) == 0);
    CHECK(*(it3 + 1) == 5);
    CHECK(*(1 + it1) == 0);
    CHECK(*(1 + it2) == 0);
    CHECK(*(1 + it3) == 5);
    CHECK(*(it1 - 1) == 0);
    CHECK(*(it2 - 1) == -4);
    CHECK(*(it3 - 1) == 3);
    CHECK(it1 > it2);
    CHECK(it1 >= it1);
    CHECK(it1 <= it1);
    CHECK(it1 <= it3);
    CHECK(it1 < it3);
    CHECK_FALSE(it1 < it2);
    CHECK_FALSE(it1 < it1);
    CHECK_FALSE(it2 >= it1);
    CHECK_FALSE(it3 <= it2);
    CHECK_FALSE(it1 > it3);
    CHECK(it1[4] == 0);
    CHECK(it2[4] == 6);
    CHECK(it3[4] == 8);
    swap(it1, it3);
    CHECK(*it1 == 4);
    CHECK(*it2 == -2);
    CHECK(*it3 == 0);
    CHECK(it1[4] == 8);
    CHECK(it2[4] == 6);
    CHECK(it3[4] == 0);

    range_t<int>::iterator it4(0, 2);
    range_t<int>::iterator it5(4, 2);
    range_t<int>::iterator it6(8, 2);
    CHECK(it4-it4 == 0);
    CHECK(it4-it5 == -2);
    CHECK(it4-it6 == -4);
    CHECK(it5-it5 == 0);
    CHECK(it5-it6 == -2);
    CHECK(it6-it6 == 0);
}

TEST_CASE("range::range_t")
{
    range_t<int> r1;
    range_t<int> r2(0, 5, 2);
    range_t<int> r3(r2);
    range_t<int> r4(range_t<int>(1, 3, 1));

    CHECK(r1.size() == 0);
    CHECK(r1.end() == r1.begin());

    CHECK(r2.front() == 0);
    CHECK(r2.back() == 4);
    CHECK(r2.step() == 2);
    CHECK(r2.size() == 3);
    CHECK(r2.begin() == range_t<int>::iterator(0,2));
    CHECK(r2.end() == range_t<int>::iterator(6,2));
    CHECK(r2[3] == 6);

    CHECK(r3.front() == 0);
    CHECK(r3.back() == 4);
    CHECK(r3.step() == 2);
    CHECK(r3.size() == 3);
    CHECK(r3.begin() == range_t<int>::iterator(0,2));
    CHECK(r3.end() == range_t<int>::iterator(6,2));
    CHECK(r3[3] == 6);

    CHECK(r4.front() == 1);
    CHECK(r4.back() == 2);
    CHECK(r4.step() == 1);
    CHECK(r4.size() == 2);
    CHECK(r4.begin() == range_t<int>::iterator(1,1));
    CHECK(r4.end() == range_t<int>::iterator(3,1));
    CHECK(r4[3] == 4);

    r1 = r2;
    r2 = range_t<int>(3, 7, 1);

    CHECK(r1.front() == 0);
    CHECK(r1.back() == 4);
    CHECK(r1.step() == 2);
    CHECK(r1.size() == 3);
    CHECK(r1.begin() == range_t<int>::iterator(0,2));
    CHECK(r1.end() == range_t<int>::iterator(6,2));
    CHECK(r1[3] == 6);

    CHECK(r2.front() == 3);
    CHECK(r2.back() == 6);
    CHECK(r2.step() == 1);
    CHECK(r2.size() == 4);
    CHECK(r2.begin() == range_t<int>::iterator(3,1));
    CHECK(r2.end() == range_t<int>::iterator(7,1));
    CHECK(r2[3] == 6);

    auto r22 = r1.reverse();
    CHECK(r22.front() == 4);
    CHECK(r22.back() == 0);
    CHECK(r22.step() == -2);
    CHECK(r22.size() == 3);
    CHECK(r22.begin() == range_t<int>::iterator(4,-2));
    CHECK(r22.end() == range_t<int>::iterator(-2,-2));
    CHECK(r22[1] == 2);

    range_t<int> r5(4);
    CHECK(r5.front() == 0);
    CHECK(r5.back() == 3);
    CHECK(r5.step() == 1);
    CHECK(r5.size() == 4);

    r5 += 2;
    CHECK(r5.front() == 2);
    CHECK(r5.back() == 5);
    CHECK(r5.step() == 1);
    CHECK(r5.size() == 4);

    r5 -= 1;
    CHECK(r5.front() == 1);
    CHECK(r5.back() == 4);
    CHECK(r5.step() == 1);
    CHECK(r5.size() == 4);

    auto r6 = r5+4;
    CHECK(r6.front() == 5);
    CHECK(r6.back() == 8);
    CHECK(r6.step() == 1);
    CHECK(r6.size() == 4);

    auto r7 = 4+r5;
    CHECK(r7.front() == 5);
    CHECK(r7.back() == 8);
    CHECK(r7.step() == 1);
    CHECK(r7.size() == 4);

    auto r8 = r7-5;
    CHECK(r8.front() == 0);
    CHECK(r8.back() == 3);
    CHECK(r8.step() == 1);
    CHECK(r8.size() == 4);

    auto r9 = 6-r5;
    CHECK(r9.front() == 5);
    CHECK(r9.back() == 2);
    CHECK(r9.step() == -1);
    CHECK(r9.size() == 4);
}

TEST_CASE("range::vector")
{
    range_t<int> r1(0, 4, 1);
    range_t<int> r2(2, 7, 2);
    vector<int> v1 = r1;
    vector<int> v2 = r2;
    CHECK(v1 == vector<int>{0,1,2,3});
    CHECK(v2 == vector<int>{2,4,6});
}

TEST_CASE("range::string")
{
    range_t<char> r('A', 'E', 1);
    string s = r;
    CHECK(s == "ABCD");
}

TEST_CASE("range::range")
{
    vector<int> v1 = range(4);
    vector<int> v2 = range(1,4);
    vector<int> v3 = range(2,9,2);
    CHECK(v1 == vector<int>{0,1,2,3});
    CHECK(v2 == vector<int>{1,2,3});
    CHECK(v3 == vector<int>{2,4,6,8});
}

TEST_CASE("range::reversed_range")
{
    vector<int> v1 = reversed_range(4);
    vector<int> v2 = reversed_range(1,4);
    vector<int> v3 = reversed_range(2,10,2);
    CHECK(v1 == vector<int>{3,2,1,0});
    CHECK(v2 == vector<int>{3,2,1});
    CHECK(v3 == vector<int>{8,6,4,2});
}

TEST_CASE("range::rangeN")
{
    vector<int> v2 = rangeN(1,4);
    vector<int> v3 = rangeN(2,9,2);
    CHECK(v2 == vector<int>{1,2,3,4});
    CHECK(v3 == vector<int>{2,4,6,8,10,12,14,16,18});
}

TEST_CASE("range::reversed_rangeN")
{
    vector<int> v2 = reversed_rangeN(6,4);
    vector<int> v3 = reversed_rangeN(18,9,2);
    CHECK(v2 == vector<int>{5,4,3,2});
    CHECK(v3 == vector<int>{16,14,12,10,8,6,4,2,0});
}
