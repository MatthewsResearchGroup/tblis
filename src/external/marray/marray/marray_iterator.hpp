#ifndef MARRAY_MARRAY_ITERATOR_HPP
#define MARRAY_MARRAY_ITERATOR_HPP

#include "types.hpp"

namespace MArray
{

template <typename Array>
class marray_iterator
{
    protected:
        Array* array_ = nullptr;
        int dim_ = 0;
        len_type i_ = 0;

    public:
        typedef std::random_access_iterator_tag iterator_category;
        typedef decltype((*array_).slice(0,0)) value_type;
        typedef len_type difference_type;
        typedef typename std::remove_reference<value_type>::type* pointer;
        typedef value_type& reference;

        marray_iterator(Array& array, int dim, len_type i)
        : array_(&array), dim_(dim), i_(i) {}

        bool operator==(const marray_iterator& other) const
        {
            return dim_ == other.dim_ && i_ == other.i_;
        }

        bool operator!=(const marray_iterator& other) const
        {
            return !(*this == other);
        }

        value_type operator*() const
        {
            return (*array_).slice(dim_, i_);
        }

        marray_iterator& operator++()
        {
            i_++;
            return *this;
        }

        marray_iterator operator++(int)
        {
            marray_iterator old(*this);
            i_++;
            return old;
        }

        marray_iterator& operator--()
        {
            i_--;
            return *this;
        }

        marray_iterator operator--(int)
        {
            marray_iterator old(*this);
            i_--;
            return old;
        }

        marray_iterator& operator+=(difference_type n)
        {
            i_ += n;
            return *this;
        }

        marray_iterator operator+(difference_type n) const
        {
            return marray_iterator(*array_, dim_, i_+n);
        }

        friend marray_iterator operator+(difference_type n, const marray_iterator& i)
        {
            return marray_iterator(*i.array_, i.dim_, i.i_+n);
        }

        marray_iterator& operator-=(difference_type n)
        {
            i_ -= n;
            return *this;
        }

        marray_iterator operator-(difference_type n) const
        {
            return marray_iterator(*array_, dim_, i_-n);
        }

        difference_type operator-(const marray_iterator& other) const
        {
            return i_ - other.i_;
        }

        bool operator<(const marray_iterator& other) const
        {
            return i_ < other.i_;
        }

        bool operator<=(const marray_iterator& other) const
        {
            return !(other < *this);
        }

        bool operator>(const marray_iterator& other) const
        {
            return other < *this;
        }

        bool operator>=(const marray_iterator& other) const
        {
            return !(*this < other);
        }

        value_type operator[](difference_type n) const
        {
            return (*array_).slice(dim_, i_+n);
        }

        friend void swap(marray_iterator& a, marray_iterator& b)
        {
            using std::swap;
            swap(a.array_, b.array_);
            swap(a.dim_, b.dim_);
            swap(a.i_, b.i_);
        }
};

}

#endif //MARRAY_MARRAY_ITERATOR_HPP
