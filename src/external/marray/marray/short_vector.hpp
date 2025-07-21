#ifndef MARRAY_SHORT_VECTOR_HPP
#define MARRAY_SHORT_VECTOR_HPP

#include <array>
#include <cstdlib>
#include <memory>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#ifndef MARRAY_ASSERT
#ifdef MARRAY_ENABLE_ASSERTS
#define MARRAY_ASSERT(e) assert(e)
#else
#define MARRAY_ASSERT(e) ((void)0)
#endif
#endif

namespace MArray
{

template <typename T, size_t N, typename Allocator=std::allocator<T>>
class short_vector
{
    protected:
        typedef std::allocator_traits<Allocator> _alloc_traits;

    public:
        typedef T value_type;
        typedef Allocator allocator_type;
        typedef size_t size_type;
        typedef ptrdiff_t difference_type;
        typedef value_type& reference;
        typedef const value_type& const_reference;
        typedef typename _alloc_traits::pointer pointer;
        typedef typename _alloc_traits::const_pointer const_pointer;
        typedef pointer iterator;
        typedef const_pointer const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        short_vector()
        : _size(0), _alloc(Allocator(), _local_data()) {}

        explicit short_vector(const Allocator& alloc)
        : _size(0), _alloc(alloc, _local_data()) {}

        short_vector(size_type count, const T& value,
                     const Allocator& alloc = Allocator())
        : _size(0), _alloc(alloc, _local_data())
        {
            assign(count, value);
        }

        explicit short_vector(size_type count)
        : _size(0), _alloc(Allocator(), _local_data())
        {
            assign(count, T());
        }

        template <typename Iterator>
        short_vector(Iterator first, Iterator last,
                     const Allocator& alloc = Allocator())
        : _size(0), _alloc(alloc, _local_data())
        {
            assign(first, last);
        }

        short_vector(const short_vector& other)
        : _size(0), _alloc(_alloc_traits::select_on_container_copy_construction(other._alloc),
                           _local_data())
        {
            assign(other.begin(), other.end());
        }

        short_vector(const short_vector& other, const Allocator& alloc)
        : _size(0), _alloc(alloc, _local_data())
        {
            assign(other.begin(), other.end());
        }

        short_vector(short_vector&& other)
        : _size(other._size), _alloc(std::move(other._alloc), _local_data())
        {
            if (other._is_local())
            {
                _uninitialized_move_n_if(other.data(), other.size(), data());
                other.clear();
            }
            else
            {
                _capacity = other._capacity;
                _alloc._data = other._alloc._data;
                other._size = 0;
                other._alloc._data = other._local_data();
            }
        }

        short_vector(short_vector&& other, const Allocator& alloc)
        : _size(other._size), _alloc(alloc, _local_data())
        {
            if (other._is_local() || alloc != other._alloc)
            {
                reserve(size());
                _uninitialized_move_n_if(other.data(), size(), data());
                other.clear();
            }
            else
            {
                _capacity = other._capacity;
                _alloc._data = other._alloc._data;
                other._size = 0;
                other._alloc._data = other._local_data();
            }
        }

        short_vector(std::initializer_list<T> init,
                     const Allocator& alloc = Allocator())
        : _size(0), _alloc(alloc, _local_data())
        {
            assign(init.begin(), init.end());
        }

        ~short_vector()
        {
            _destroy(begin(), end());
            if (!_is_local())
                _alloc_traits::deallocate(_alloc, data(), capacity());
        }

        short_vector& operator=(const short_vector& other)
        {
            if (_alloc_traits::propagate_on_container_copy_assignment::value)
                _alloc = other._alloc;

            assign(other.begin(), other.end());

            return *this;
        }

        short_vector& operator=(short_vector&& other)
        {
            if (_alloc_traits::propagate_on_container_move_assignment::value)
                _alloc = std::move(other._alloc);

            if (other._is_local() || (!_alloc_traits::propagate_on_container_move_assignment::value && _alloc != other._alloc))
            {
                assign(other.begin(), other.end());
                other.clear();
            }
            else if (_is_local())
            {
                clear();
                _size = other._size;
                _capacity = other._capacity;
                _alloc._data = other._alloc._data;
                other._size = 0;
                other._alloc._data = other._local_data();
            }
            else
            {
                using std::swap;
                swap(_size, other._size);
                swap(_capacity, other._capacity);
                swap(_alloc._data, other._alloc._data);
            }

            return *this;
        }

        short_vector& operator=(std::initializer_list<T> ilist)
        {
            assign(ilist.begin(), ilist.end());
            return *this;
        }

        void assign(size_type count, const T& value)
        {
            if (capacity() < count)
            {
                clear();
                reserve(count);
            }

            if (size() > count)
                erase(begin()+count, end());
            std::fill_n(begin(), std::min(count, size()), value);
            if (count > size())
                _construct_n(end(), count-size(), value);

            _size = count;
        }

        template <typename Iterator>
        typename std::enable_if<std::is_convertible<
            typename std::iterator_traits<Iterator>::iterator_category,
            std::input_iterator_tag>::value>::type
        assign(Iterator first, Iterator last)
        {
            _assign(first, last, typename std::iterator_traits<Iterator>::iterator_category());
        }

        void assign(std::initializer_list<T> ilist)
        {
            assign(ilist.begin(), ilist.end());
        }

        allocator_type get_allocator() const
        {
            return _alloc;
        }

        reference at(size_type pos)
        {
            if (pos >= size())
                throw std::out_of_range("short_vector: out-of-range");
            return data()[pos];
        }

        const_reference at(size_type pos) const
        {
            if (pos >= size())
                throw std::out_of_range("short_vector: out-of-range");
            return data()[pos];
        }

        reference operator[](size_type pos)
        {
            MARRAY_ASSERT(pos < size());
            return data()[pos];
        }

        const_reference operator[](size_type pos) const
        {
            MARRAY_ASSERT(pos < size());
            return data()[pos];
        }

        reference front()
        {
            return data()[0];
        }

        const_reference front() const
        {
            return data()[0];
        }

        reference back()
        {
            return data()[size()-1];
        }

        const_reference back() const
        {
            return data()[size()-1];
        }

        pointer data()
        {
            return _alloc._data;
        }

        const_pointer data() const
        {
            return _alloc._data;
        }

        iterator begin()
        {
            return data();
        }

        const_iterator begin() const
        {
            return data();
        }

        const_iterator cbegin() const
        {
            return data();
        }

        iterator end()
        {
            return data()+size();
        }

        const_iterator end() const
        {
            return data()+size();
        }

        const_iterator cend() const
        {
            return data()+size();
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator(end());
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator(end());
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator(end());
        }

        reverse_iterator rend()
        {
            return reverse_iterator(begin());
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator(begin());
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator(begin());
        }

        bool empty() const
        {
            return size() == 0;
        }

        size_type size() const
        {
            return _size;
        }

        size_type max_size() const
        {
            return std::numeric_limits<size_type>::max();
        }

        void reserve(size_type new_cap)
        {
            if (new_cap > capacity())
                _set_capacity(_new_capacity(capacity(), new_cap));
        }

        size_type capacity() const
        {
            return (_is_local() ? N : _capacity);
        }

        void shrink_to_fit()
        {
            _set_capacity(size());
        }

        void clear()
        {
            erase(begin(), end());
        }

        iterator insert(const_iterator pos, const T& value)
        {
            return _emplace(pos, 1, value);
        }

        iterator insert(const_iterator pos, T&& value)
        {
            return _emplace(pos, 1, std::move(value));
        }

        iterator insert(const_iterator pos, size_type count, const T& value)
        {
            return _emplace(pos, count, value);
        }

        template <typename Iterator>
        typename std::enable_if<std::is_convertible<
            typename std::iterator_traits<Iterator>::iterator_category,
            std::input_iterator_tag>::value, iterator>::type
        insert(const_iterator pos, Iterator first, Iterator last)
        {
            return _insert(pos, first, last, typename std::iterator_traits<Iterator>::iterator_category());
        }

        iterator insert(const_iterator pos, std::initializer_list<T> ilist)
        {
            return insert(pos, ilist.begin(), ilist.end());
        }

        template <typename... Args>
        iterator emplace(const_iterator pos, Args&&... args)
        {
            return _emplace(pos, 1, std::forward<Args>(args)...);
        }

        iterator erase(const_iterator pos)
        {
            return erase(pos, pos+1);
        }

        iterator erase(const_iterator first, const_iterator last)
        {
            size_type n = last-first;
            iterator cur = begin() + (first-cbegin());
            std::move(last, cend(), cur);
            _destroy(end()-n, end());
            _size -= n;
            return cur;
        }

        void push_back(const T& value)
        {
            _emplace(end(), 1, value);
        }

        void push_back(T&& value)
        {
            _emplace(end(), 1, std::move(value));
        }

        template <typename... Args>
        void emplace_back(Args&&... args)
        {
            _emplace(end(), 1, std::forward<Args>(args)...);
        }

        void pop_back()
        {
            erase(end()-1);
        }

        void resize(size_type count)
        {
            resize(count, T());
        }

        void resize(size_type count, const value_type& value)
        {
            reserve(count);
            if (count < size())
                _destroy(begin()+count, end());
            else if (count > size())
                _construct_n(end(), count-size(), value);
            _size = count;
        }

        void swap(short_vector& other)
        {
            using std::swap;

            if (_alloc_traits::propagate_on_container_swap::value)
                swap(_alloc, other._alloc);

            if ((_is_local() && other._is_local()) ||
                (!_alloc_traits::propagate_on_container_swap::value && _alloc != other._alloc))
            {
                std::swap_ranges(begin(), begin()+std::min(size(), other.size()), other.begin());
                if (size() < other.size())
                {
                    _uninitialized_move_n_if(other.begin()+size(), other.size()-size(), end());
                    _destroy(other.begin()+size(), other.end());
                }
                else if (size() > other.size())
                {
                    _uninitialized_move_n_if(begin()+other.size(), size()-other.size(), other.end());
                    _destroy(begin()+other.size(), end());
                }
            }
            else if (_is_local())
            {
                size_type other_cap = other.capacity();
                pointer other_data = other.data();
                other._alloc._data = other._local_data();
                _uninitialized_move_n_if(begin(), size(), other.begin());
                _destroy(begin(), end());
                _capacity = other_cap;
                _alloc._data = other_data;
            }
            else if (other._is_local())
            {
                size_type this_cap = capacity();
                pointer this_data = data();
                _alloc._data = _local_data();
                _uninitialized_move_n_if(other.begin(), other.size(), begin());
                _destroy(other.begin(), other.end());
                other._capacity = this_cap;
                other._alloc._data = this_data;
            }
            else
            {
                swap(_capacity, other._capacity);
                swap(_alloc._data, other._alloc._data);
            }

            swap(_size, other._size);
        }

        friend void swap(short_vector& lhs, short_vector& rhs)
        {
            lhs.swap(rhs);
        }

        friend bool operator==(const short_vector& lhs,
                               const short_vector& rhs)
        {
            return lhs.size() == rhs.size() &&
                std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }

        friend bool operator!=(const short_vector& lhs,
                               const short_vector& rhs)
        {
            return !(lhs == rhs);
        }

        friend bool operator<(const short_vector& lhs,
                              const short_vector& rhs)
        {
            return std::lexicographical_compare(lhs.begin(), lhs.end(),
                                                rhs.begin(), rhs.end());
        }

        friend bool operator<=(const short_vector& lhs,
                               const short_vector& rhs)
        {
            return !(rhs < lhs);
        }

        friend bool operator>(const short_vector& lhs,
                              const short_vector& rhs)
        {
            return (rhs < lhs);
        }

        friend bool operator>=(const short_vector& lhs,
                               const short_vector& rhs)
        {
            return !(lhs < rhs);
        }

    protected:
        void _set_capacity(size_type new_cap)
        {
            size_type old_cap = capacity();
            pointer old_data = data();

            if (new_cap <= N)
            {
                if (_is_local()) return;
                _alloc._data = _local_data();
            }
            else
            {
                _alloc._data = _alloc_traits::allocate(_alloc, new_cap);
            }

            _uninitialized_move_n_if(old_data, size(), data());
            _destroy(old_data, old_data+size());

            if (old_data != _local_data())
                _alloc_traits::deallocate(_alloc, old_data, old_cap);

            if (!_is_local())
                _capacity = new_cap;
        }

        template <typename Iterator>
        void _assign(Iterator first, Iterator last, std::random_access_iterator_tag)
        {
            size_type count = last-first;

            if (capacity() < count)
            {
                clear();
                reserve(count);
            }

            if (size() > count)
                _destroy(begin()+count, end());

            std::copy_n(first, std::min(count, size()), begin());

            if (count > size())
                _construct(first+size(), last, end());

            _size = count;
        }

        template <typename Iterator>
        void _assign(Iterator first, Iterator last, std::input_iterator_tag)
        {
            iterator it = begin();

            for (;it != end() && first != last;++it, ++first)
                *it = *first;

            erase(it, end());

            for (;first != last;++first)
                emplace_back(*first);
        }

        template <typename Iterator>
        void _destroy(Iterator first, Iterator last)
        {
            if (first == last) return;
            for (--last;last != first;--last)
                _alloc_traits::destroy(_alloc, last);
        }

        template <typename Iterator>
        iterator _construct(Iterator first, Iterator last, iterator result)
        {
            iterator cur = result;

            try
            {
                for (;first != last;++first, ++cur)
                    _alloc_traits::construct(_alloc, cur, *first);
            }
            catch (...)
            {
                _destroy(result, cur);
                throw;
            }

            return result;
        }

        template <typename... Args>
        iterator _construct_n(iterator result, size_type n, Args&&... args)
        {
            iterator last = result+n;
            iterator cur = result;

            try
            {
                for (;cur != last;++cur)
                    _alloc_traits::construct(_alloc, cur, std::forward<Args>(args)...);
            }
            catch (...)
            {
                _destroy(result, cur);
                throw;
            }

            return result;
        }

        template <typename T_=T>
        typename std::enable_if<std::is_nothrow_move_constructible<T_>::value ||
                                !std::is_copy_constructible<T_>::value,iterator>::type
        _uninitialized_move_n_if(iterator first, size_type n, iterator result)
        {
            auto it = std::make_move_iterator(first);
            return _construct(it, it+n, result);
        }

        template <typename T_=T>
        typename std::enable_if<!std::is_nothrow_move_constructible<T_>::value,iterator>::type
        _uninitialized_move_n_if(iterator first, size_type n, iterator result)
        {
            return _construct(first, first+n, result);
        }

        template <typename... Args>
        iterator _emplace(const_iterator cpos, size_type n, Args&&... args)
        {
            size_type off = cpos-begin();

            reserve(size()+n);
            iterator pos = begin()+off;

            size_type n_tail = end()-pos;
            size_type n_uninit = std::min(n, n_tail);
            size_type n_construct = std::max(n, n_tail)-n_tail;
            size_type n_move = n_tail-n_uninit;
            size_type n_fill = n-n_construct;

            _uninitialized_move_n_if(end()-n_uninit, n_uninit, end()+n_construct);
            _construct_n(end(), n_construct, std::forward<Args>(args)...);
            std::move_backward(pos, pos+n_move, end());
            std::fill_n(pos, n_fill, T(std::forward<Args>(args)...));

            _size += n;

            return pos+n;
        }

        template <typename Iterator>
        iterator _insert(const_iterator cpos, Iterator first, Iterator last,
                         std::random_access_iterator_tag)
        {
            size_type n = last-first;
            size_type off = cpos-cbegin();

            reserve(size()+n);
            iterator pos = begin()+off;

            size_type n_tail = end()-pos;
            size_type n_uninit = std::min(n, n_tail);
            size_type n_construct = std::max(n, n_tail)-n_tail;
            size_type n_move = n_tail-n_uninit;
            size_type n_copy = n-n_construct;

            _uninitialized_move_n_if(end()-n_uninit, n_uninit, end()+n_construct);
            _construct(last-n_construct, last, end());
            std::move_backward(pos, pos+n_move, end());
            std::copy_n(first, n_copy, pos);

            _size += n;

            return pos+n;
        }

        template <typename Iterator>
        iterator _insert(const_iterator cpos, Iterator first, Iterator last,
                         std::input_iterator_tag)
        {
            size_type off = cpos-cbegin();
            size_type cur = off;

            for (;first != last;++first)
            {
                reserve(size()+1);
                iterator pos = begin()+cur;

                if (pos == end())
                {
                    _construct_n(end(), 1, *first);
                }
                else
                {
                    _uninitialized_move_n_if(pos, 1, end());
                    *pos = *first;
                }

                ++_size;
                ++cur;
            }

            iterator pos = begin()+off;
            size_type n_tail = size()-cur;
            size_type n = cur-off;

            if (n_tail > 0)
                std::rotate(pos+n, pos+n+n_tail-(n%n_tail), pos+n+n_tail);

            return pos;
        }

        static size_type _new_capacity(size_type capacity, size_type count)
        {
            return std::max(2*capacity, count);
        }

        pointer _local_data()
        {
            return &_fixed[0];
        }

        const_pointer _local_data() const
        {
            return &_fixed[0];
        }

        bool _is_local() const
        {
            return _alloc._data == _local_data();
        }

        size_type _size;
        struct _data_s : public Allocator
        {
            _data_s(Allocator alloc, pointer data)
            : Allocator(alloc), _data(data) {}

            _data_s(const _data_s& other, pointer data)
            : Allocator(other), _data(data) {}

            _data_s(_data_s&& other, pointer data)
            : Allocator(std::move(other)), _data(data) {}

            _data_s& operator=(const _data_s& other)
            {
                Allocator::operator=(other);
                return *this;
            }

            _data_s& operator=(_data_s&& other)
            {
                Allocator::operator=(std::move(other));
                return *this;
            }

            friend void swap(_data_s& a, _data_s& b)
            {
                using std::swap;
                swap(static_cast<Allocator&>(a), static_cast<Allocator&>(b));
            }

            pointer _data;
        } _alloc;

        union
        {
            std::array<T,N> _fixed;
            size_type _capacity;
        };
};

}

#endif //MARRAY_SHORT_VECTOR_HPP
