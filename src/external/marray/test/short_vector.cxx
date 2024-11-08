#include "short_vector.hpp"
#include "catch_amalgamated.hpp"

#include <list>
#include <vector>

using namespace std;
using namespace MArray;

TEST_CASE("short_vector::constructor")
{
    short_vector<int, 4> v1;
    CHECK(v1.size() == 0u);
    CHECK(v1.capacity() == 4u);

    short_vector<int, 4> v2(2);
    CHECK(v2.size() == 2u);
    CHECK(v2.capacity() == 4u);

    short_vector<int, 4> v3(6);
    CHECK(v3.size() == 6u);
    CHECK(v3.capacity() >= 6u);

    vector<int> v{0, 2};
    short_vector<int, 4> v4(v.begin(), v.end());
    CHECK(v4.size() == 2u);
    CHECK(v4.capacity() == 4u);

    short_vector<int, 4> v5(v4);
    CHECK(v5.size() == 2u);
    CHECK(v5.capacity() == 4u);
    CHECK(v4 == v5);

    short_vector<int, 4> v6(short_vector<int, 4>(3));
    CHECK(v6.size() == 3u);
    CHECK(v6.capacity() == 4u);

    short_vector<int, 4> v7{0, 2, 4};
    CHECK(v7.size() == 3u);
    CHECK(v7.capacity() == 4u);

    short_vector<int, 4> v8{0, 1, 2, 3, 4, 5, 6};
    CHECK(v8.size() == 7u);
    CHECK(v8.capacity() >= 7u);
}

TEST_CASE("short_vector::assignment")
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2;
    CHECK(v1 != v2);
    v2 = v1;
    CHECK(v2 == v1);
    v2.clear();
    CHECK(v1 != v2);
    v2 = short_vector<int, 4>{0, 2, 4};
    CHECK(v2 == v1);

    short_vector<int, 4> v3{0, 2, 4, 6, 9, 2, 5};
    short_vector<int, 4> v4;
    CHECK(v3 != v4);
    v4 = v3;
    CHECK(v4 == v3);
    v4 = v1;
    CHECK(v4 == v1);
    v2 = v3;
    CHECK(v2 == v3);
    v4.clear();
    CHECK(v3 != v4);
    v4 = short_vector<int, 4>{0, 2, 4};
    CHECK(v4 == v1);
    v4 = short_vector<int, 4>{0, 2, 4, 6, 9, 2, 5};
    CHECK(v4 == v3);
    v2 = short_vector<int, 4>{0, 2, 4, 6, 9, 2, 5};
    CHECK(v2 == v3);

    short_vector<int, 4> v5;
    v5 = {0, 2, 4};
    CHECK(v5 == v1);
    v5 = {0, 2, 4, 6, 9, 2, 5};
    CHECK(v5 == v3);
    v5 = {0, 2, 4};
    CHECK(v5 == v1);
}

TEST_CASE("short_vector::assign")
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};
    short_vector<int, 4> v3{5, 5, 5};
    short_vector<int, 4> v4{5, 5, 5, 5, 5, 5, 5};

    short_vector<int, 4> v5;
    v5.assign({0, 2, 4});
    CHECK(v5 == v1);
    v5.assign({0, 2, 4, 6, 9, 2, 5});
    CHECK(v5 == v2);
    v5.assign({0, 2, 4});
    CHECK(v5 == v1);

    short_vector<int, 4> v6;
    v6.assign(v1.begin(), v1.end());
    CHECK(v6 == v1);
    v6.assign(v2.begin(), v2.end());
    CHECK(v6 == v2);
    v6.assign(v1.begin(), v1.end());
    CHECK(v6 == v1);

    short_vector<int, 4> v7;
    v7.assign(3, 5);
    CHECK(v7 == v3);
    v7.assign(7, 5);
    CHECK(v7 == v4);
}

TEST_CASE("short_vector::access")
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};

    CHECK(v1.at(0) == 0);
    CHECK(v1.at(2) == 4);
    CHECK(v1[0] == 0);
    CHECK(v1[2] == 4);
    CHECK_THROWS_AS(v1.at(-1), out_of_range);
    CHECK_THROWS_AS(v1.at(3), out_of_range);
    CHECK(v1.front() == 0);
    CHECK(v1.back() == 4);
    CHECK(v1.data()[0] == 0);
    CHECK(v1.data()[2] == 4);

    CHECK(v2.at(0) == 0);
    CHECK(v2.at(2) == 4);
    CHECK(v2.at(4) == 9);
    CHECK(v2.at(5) == 2);
    CHECK(v2[0] == 0);
    CHECK(v2[2] == 4);
    CHECK(v2[4] == 9);
    CHECK(v2[5] == 2);
    CHECK_THROWS_AS(v2.at(-1), out_of_range);
    CHECK_THROWS_AS(v2.at(7), out_of_range);
    CHECK(v2.front() == 0);
    CHECK(v2.back() == 5);
    CHECK(v2.data()[0] == 0);
    CHECK(v2.data()[2] == 4);
    CHECK(v2.data()[4] == 9);
    CHECK(v2.data()[5] == 2);

    CHECK_FALSE(v1.data() == v2.data());
}

TEST_CASE("short_vector::iterator")
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};

    CHECK(*v1.begin() == 0);
    CHECK(*next(v1.begin()) == 2);
    CHECK(*prev(v1.end()) == 4);
    CHECK(*v1.rbegin() == 4);
    CHECK(*next(v1.rbegin()) == 2);
    CHECK(*prev(v1.rend()) == 0);
    CHECK(*v1.cbegin() == 0);
    CHECK(*next(v1.cbegin()) == 2);
    CHECK(*prev(v1.cend()) == 4);
    CHECK(*v1.crbegin() == 4);
    CHECK(*next(v1.crbegin()) == 2);
    CHECK(*prev(v1.crend()) == 0);

    CHECK(*v2.begin() == 0);
    CHECK(*next(v2.begin()) == 2);
    CHECK(*prev(v2.end()) == 5);
    CHECK(*v2.rbegin() == 5);
    CHECK(*next(v2.rbegin()) == 2);
    CHECK(*prev(v2.rend()) == 0);
    CHECK(*v2.cbegin() == 0);
    CHECK(*next(v2.cbegin()) == 2);
    CHECK(*prev(v2.cend()) == 5);
    CHECK(*v2.crbegin() == 5);
    CHECK(*next(v2.crbegin()) == 2);
    CHECK(*prev(v2.crend()) == 0);
}

TEST_CASE("short_vector::size_capacity_empty")
{
    short_vector<int, 4> v1;

    CHECK(v1.size() == 0u);
    CHECK(v1.capacity() == 4u);
    CHECK(v1.empty());

    v1 = {0, 3, 4};

    CHECK(v1.size() == 3u);
    CHECK(v1.capacity() == 4u);
    CHECK_FALSE(v1.empty());

    v1 = {0, 3, 4, 0, 4, 2, 4, 8};

    CHECK(v1.size() == 8u);
    CHECK(v1.capacity() >= 8u);
    CHECK_FALSE(v1.empty());

    v1.clear();

    CHECK(v1.size() == 0u);
    CHECK(v1.capacity() >= 8u);
    CHECK(v1.empty());

    v1 = {0, 3, 4, 0, 4};
    v1.shrink_to_fit();

    CHECK(v1.size() == 5u);
    CHECK(v1.capacity() == 5u);
    CHECK_FALSE(v1.empty());

    v1.clear();
    v1.shrink_to_fit();

    CHECK(v1.size() == 0u);
    CHECK(v1.capacity() == 4u);
    CHECK(v1.empty());

    v1.reserve(567);

    CHECK(v1.size() == 0u);
    CHECK(v1.capacity() >= 567u);
    CHECK(v1.empty());
}

TEST_CASE("short_vector::insert_emplace")
{
    short_vector<int, 4> v1;

    v1.insert(v1.begin(), 5);
    CHECK(v1 == short_vector<int, 4>{5});
    v1.insert(v1.begin(), 4);
    CHECK(v1 == short_vector<int, 4>{4, 5});
    v1.insert(v1.end(), 6);
    CHECK(v1 == short_vector<int, 4>{4, 5, 6});
    v1.insert(next(v1.begin()), 3);
    CHECK(v1 == short_vector<int, 4>{4, 3, 5, 6});
    v1.insert(v1.end(), 7);
    CHECK(v1 == short_vector<int, 4>{4, 3, 5, 6, 7});
    v1.insert(next(v1.begin()), 2);
    CHECK(v1 == short_vector<int, 4>{4, 2, 3, 5, 6, 7});
    v1.insert(v1.begin(), 1);
    CHECK(v1 == short_vector<int, 4>{1, 4, 2, 3, 5, 6, 7});
    v1.emplace(v1.begin(), 0);
    CHECK(v1 == short_vector<int, 4>{0, 1, 4, 2, 3, 5, 6, 7});
    v1.emplace(v1.begin()+4, 8);
    CHECK(v1 == short_vector<int, 4>{0, 1, 4, 2, 8, 3, 5, 6, 7});
    v1.emplace(v1.end(), 4);
    CHECK(v1 == short_vector<int, 4>{0, 1, 4, 2, 8, 3, 5, 6, 7, 4});

    short_vector<int, 4> v2;

    v2.insert(v2.begin(), 2, 5);
    CHECK(v2 == short_vector<int, 4>{5, 5});
    v2.insert(v2.begin(), 2, 4);
    CHECK(v2 == short_vector<int, 4>{4, 4, 5, 5});
    v2.insert(v2.end(), 2, 6);
    CHECK(v2 == short_vector<int, 4>{4, 4, 5, 5, 6, 6});
    v2.insert(next(v2.begin()), 3, 3);
    CHECK(v2 == short_vector<int, 4>{4, 3, 3, 3, 4, 5, 5, 6, 6});
    v2.insert(v2.end(), 1, 7);
    CHECK(v2 == short_vector<int, 4>{4, 3, 3, 3, 4, 5, 5, 6, 6, 7});

    short_vector<int, 4> v3;
    vector<int> v{1, 2};
    list<int> l{3, 4};

    v3.insert(v3.begin(), v.begin(), v.end());
    CHECK(v3 == short_vector<int, 4>{1, 2});
    v3.insert(v3.begin(), l.begin(), l.end());
    CHECK(v3 == short_vector<int, 4>{3, 4, 1, 2});
    v3.insert(v3.end(), l.begin(), l.end());
    CHECK(v3 == short_vector<int, 4>{3, 4, 1, 2, 3, 4});
    v3.insert(next(v3.begin()), l.begin(), l.end());
    CHECK(v3 == short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 4});
    v3.insert(prev(v3.end()), v.begin(), v.end());
    CHECK(v3 == short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 1, 2, 4});

    short_vector<int, 4> v4;

    v4.insert(v4.begin(), {1, 2});
    CHECK(v4 == short_vector<int, 4>{1, 2});
    v4.insert(v4.begin(), {3, 4});
    CHECK(v4 == short_vector<int, 4>{3, 4, 1, 2});
    v4.insert(v4.end(), {3, 4});
    CHECK(v4 == short_vector<int, 4>{3, 4, 1, 2, 3, 4});
    v4.insert(next(v4.begin()), {3, 4});
    CHECK(v4 == short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 4});
    v4.insert(prev(v4.end()), {1, 2});
    CHECK(v4 == short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 1, 2, 4});
}

TEST_CASE("short_vector::erase")
{
    short_vector<int, 4> v1{1, 2, 3, 4, 5, 6, 7, 8, 9};

    v1.erase(v1.begin());
    CHECK(v1 == short_vector<int, 4>{2, 3, 4, 5, 6, 7, 8, 9});
    v1.erase(prev(v1.end()));
    CHECK(v1 == short_vector<int, 4>{2, 3, 4, 5, 6, 7, 8});
    v1.erase(v1.begin()+3);
    CHECK(v1 == short_vector<int, 4>{2, 3, 4, 6, 7, 8});
    v1.erase(v1.begin(), v1.begin()+2);
    CHECK(v1 == short_vector<int, 4>{4, 6, 7, 8});
    v1.erase(v1.begin()+1, v1.begin()+3);
    CHECK(v1 == short_vector<int, 4>{4, 8});
}

TEST_CASE("short_vector::push_pop")
{
    short_vector<int, 4> v1;

    v1.push_back(2);
    CHECK(v1 == short_vector<int, 4>{2});
    v1.push_back(3);
    CHECK(v1 == short_vector<int, 4>{2, 3});
    v1.emplace_back(1);
    CHECK(v1 == short_vector<int, 4>{2, 3, 1});
    v1.emplace_back(4);
    CHECK(v1 == short_vector<int, 4>{2, 3, 1, 4});
    v1.push_back(4);
    CHECK(v1 == short_vector<int, 4>{2, 3, 1, 4, 4});
    v1.pop_back();
    CHECK(v1 == short_vector<int, 4>{2, 3, 1, 4});
    v1.pop_back();
    CHECK(v1 == short_vector<int, 4>{2, 3, 1});
    v1.pop_back();
    CHECK(v1 == short_vector<int, 4>{2, 3});
    v1.pop_back();
    CHECK(v1 == short_vector<int, 4>{2});
}

TEST_CASE("short_vector::resize")
{
    short_vector<int, 4> v1;

    v1.resize(2);
    CHECK(v1 == short_vector<int, 4>{0, 0});
    v1.resize(5, 1);
    CHECK(v1 == short_vector<int, 4>{0, 0, 1, 1, 1});
    v1.resize(3);
    CHECK(v1 == short_vector<int, 4>{0, 0, 1});
    v1.resize(6, 4);
    CHECK(v1 == short_vector<int, 4>{0, 0, 1, 4, 4, 4});
    v1.resize(1, 8);
    CHECK(v1 == short_vector<int, 4>{0});
}

TEST_CASE("short_vector::swap")
{
    short_vector<int, 4> v1{1, 2, 3};
    short_vector<int, 4> v2{9, 2};
    short_vector<int, 4> v3{0, 5, 9, 4, 5};
    short_vector<int, 4> v4{6, 5, 4, 3, 2, 1, 0};

    v1.swap(v2);
    v3.swap(v4);
    CHECK(v2 == short_vector<int, 4>{1, 2, 3});
    CHECK(v1 == short_vector<int, 4>{9, 2});
    CHECK(v4 == short_vector<int, 4>{0, 5, 9, 4, 5});
    CHECK(v3 == short_vector<int, 4>{6, 5, 4, 3, 2, 1, 0});
    swap(v1, v3);
    swap(v4, v2);
    CHECK(v4 == short_vector<int, 4>{1, 2, 3});
    CHECK(v3 == short_vector<int, 4>{9, 2});
    CHECK(v2 == short_vector<int, 4>{0, 5, 9, 4, 5});
    CHECK(v1 == short_vector<int, 4>{6, 5, 4, 3, 2, 1, 0});
}

TEST_CASE("short_vector::operators")
{
    short_vector<int, 4> v1{0, 1};
    short_vector<int, 4> v2{0, 2, 4};

    CHECK(v1 == v1);
    CHECK(v2 == v2);
    CHECK(v1 != v2);
    CHECK(v2 > v1);
    CHECK(v1 >= v1);
    CHECK(v1 <= v1);
    CHECK(v1 <= v2);
    CHECK(v1 < v2);
    CHECK_FALSE(v2 < v1);
    CHECK_FALSE(v1 < v1);
    CHECK_FALSE(v1 >= v2);
    CHECK_FALSE(v2 <= v1);
    CHECK_FALSE(v1 > v2);
}
