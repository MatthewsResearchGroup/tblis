#ifndef _TBLIS_STACK_ALLOCATOR_HPP_
#define _TBLIS_STACK_ALLOCATOR_HPP_

#include <stdexcept>
#include <cstddef>
#include <vector>
#include <memory>

template <typename T>
class stack_allocator
{
    protected:
        class stack
        {
            public:
                stack(const stack&) = delete;

                stack(stack&&) = default;

                stack& operator=(const stack&) = delete;

                stack& operator=(stack&&) = default;

                stack(size_t size)
                : _mem(::operator new(size)), _size(size), _pos(0) {}

                ~stack()
                {
                    if (_pos > 0) throw std::runtime_error("attempting to delete a stack which is not empty");
                    ::operator delete(_mem);
                }

                template <typename U>
                U* allocate(size_t size, size_t align = alignof(U))
                {
                    align = std::min(align, alignof(size_t));

                    size_t size_bytes = size*sizeof(U);
                    char* ptr = align_to(_mem+_pos+2*sizeof(size_t), align);

                    if (ptr+size_bytes > _mem+_size)
                        return nullptr;

                    prev_pos(ptr) = _pos;
                    ptr_size(ptr) = size;

                    _pos = size_t(ptr-_mem)+size_bytes;
                    return (U*)ptr;
                }

                template <typename U>
                bool deallocate(U* t_ptr, size_t size)
                {
                    size_t size_bytes = size*sizeof(U);
                    char *ptr = (char*)t_ptr;

                    if (_pos != size_t(ptr-_mem)+size_bytes)
                        throw std::runtime_error("attempting to free memory which is not at the top of the stack");

                    if (ptr_size(ptr) != size)
                        throw std::runtime_error("attempting to free memory of wrong size");

                    _pos = prev_pos(ptr);

                    return _pos > 0;
                }

            protected:
                static char* align_to(char* pos, size_t align)
                {
                    return pos + (align-1) - uintptr_t(pos-1)%align;
                }

                size_t& prev_pos(char* ptr)
                {
                    return *(size_t*)(ptr-2*sizeof(size_t));
                }

                size_t& ptr_size(char* ptr)
                {
                    return *(size_t*)(ptr-sizeof(size_t));
                }

                char* _mem;
                size_t _size;
                size_t _pos;
        };

    public:
        typedef T value_type;

        stack_allocator(size_t stack_size, bool expandable = true, size_t alignment = alignof(T))
        : _stack_size(stack_size), _expandable(expandable), _align(alignment),
          _stacks(std::make_shared<std::vector<stack>>()), _stack(0)
        {
            _stacks->emplace_back(_stack_size*sizeof(T));
        }

        stack_allocator(const stack_allocator& other) = default;

        stack_allocator(stack_allocator&& other) = default;

        stack_allocator& operator=(const stack_allocator& other) = default;

        stack_allocator& operator=(stack_allocator&& other) = default;

        template <typename U>
        stack_allocator(const stack_allocator<U>& other)
        : _stack_size(other._stack_size), _expandable(other._expandable), _align(std::max(alignof(T),other._align)),
          _stacks(other._stacks), _stack(other._stack) {}

        template <typename U>
        stack_allocator(stack_allocator<U>&& other)
        : _stack_size(other._stack_size), _expandable(other._expandable), _align(std::max(alignof(T),other._align)),
          _stacks(std::move(other._stacks)), _stack(other._stack) {}

        template <typename U>
        friend bool operator==(const stack_allocator& a, const stack_allocator<U>& b)
        {
            return a._stacks == b._stacks;
        }

        template <typename U>
        friend bool operator!=(const stack_allocator& a, const stack_allocator<U>& b)
        {
            return a._stacks != b._stacks;
        }

        T* allocate(size_t size)
        {
            if (size*sizeof(T) > _stack_size)
                throw std::bad_alloc("stack is not large enough");

            T* ptr = _stacks->at(_stack).template allocate<T>(size, _align);

            if (!ptr)
            {
                if (++_stack >= _stacks->size())
                {
                    if (!_expandable)
                        throw std::bad_alloc("stack space exhausted");
                    _stacks->emplace_back(_stack_size);
                }

                ptr = _stacks->at(_stack).template allocate<T>(size, _align);
                if (!ptr)
                    throw std::bad_alloc("error allocating on new stack");
            }

            return ptr;
        }

        void deallocate(T* ptr, size_t size)
        {
            if (!_stacks->at(_stack).deallocate(ptr, size))
            {
                if (_stack > 0) _stack--;
            }
        }

    protected:
        size_t _stack_size;
        bool _expandable;
        size_t _align;
        std::shared_ptr<std::vector<stack>> _stacks;
        int _stack;
};

#endif
