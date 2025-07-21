#ifndef _TBLIS_ALIGNED_ALLOCATOR_HPP_
#define _TBLIS_ALIGNED_ALLOCATOR_HPP_

#include <cstdlib>
#include <new>

#if TBLIS_HAVE_HBWMALLOC_H
#include <hbwmalloc.h>
#endif

namespace tblis
{

template <typename T, size_t N=8> struct aligned_allocator
{
    typedef T value_type;

    aligned_allocator() {}

    template <typename U, size_t M>
    aligned_allocator(const aligned_allocator<U, M>& other)
    {
        (void)other;
    }

    T* allocate(size_t n)
    {
        if (n == 0) return nullptr;

        void* ptr;
#if TBLIS_HAVE_HBWMALLOC_H
        int ret = hbw_posix_memalign(&ptr, N, n*sizeof(T));
#else
        int ret = posix_memalign(&ptr, N, n*sizeof(T));
#endif
        if (ret != 0) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, size_t n)
    {
        (void)n;

        if (!ptr) return;

#if TBLIS_HAVE_HBWMALLOC_H
        hbw_free(ptr);
#else
        free(ptr);
#endif
    }

    template<class U>
    struct rebind { typedef aligned_allocator<U, N> other; };
};

template <typename T, size_t N, typename U, size_t M>
bool operator==(const aligned_allocator<T, N>&, const aligned_allocator<U, M>&) { return true; }

template <typename T, size_t N, typename U, size_t M>
bool operator!=(const aligned_allocator<T, N>&, const aligned_allocator<U, M>&) { return false; }

}

#endif
