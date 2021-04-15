#ifndef MARRAY_ROTATE_HPP
#define MARRAY_ROTATE_HPP

#include "../marray/miterator.hpp"
#include "../marray/utility.hpp"
#include "../marray/viterator.hpp"

namespace MArray
{

template <typename Array>
void rotate(Array& array, std::initializer_list<len_type> shift)
{
    rotate<Array, std::initializer_list<len_type>>(array, shift);
}

template <typename Array, typename U, typename=detail::enable_if_container_of_t<U,len_type>>
void rotate(Array& array, const U& shift)
{
    MARRAY_ASSERT(shift.size() == array.dimension());

    auto it = shift.begin();
    for (auto i : range(array.dimension()))
    {
        rotate(array, i, *it);
        ++it;
    }
}

template <typename Array>
void rotate(Array& array, int dim, len_type shift)
{
    MARRAY_ASSERT(dim >= 0 && dim < array.dimension());

    len_type n = array.length(dim);
    stride_type s = array.stride(dim);

    if (n == 0) return;

    shift = shift%n;
    if (shift < 0) shift += n;

    if (shift == 0) return;

    auto len = array.lengths();
    auto stride = array.strides();
    len[dim] = 1;

    auto p = array.data();
    auto it = make_iterator(len, stride);
    while (it.next(p))
    {
        auto a = p;
        auto b = p+(shift-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }

        a = p+shift*s;
        b = p+(n-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }

        a = p;
        b = p+(n-1)*s;
        while (a < b)
        {
            std::iter_swap(a, b);
            a += s;
            b -= s;
        }
    }
}

}

#endif //MARRAY_ROTATE_HPP
