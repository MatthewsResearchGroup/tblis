#ifndef MARRAY_ARRAY_1D_HPP
#define MARRAY_ARRAY_1D_HPP

#include <type_traits>
#include <initializer_list>
#include <vector>
#include <array>

#include "detail/utility.hpp"

namespace MArray
{
namespace detail
{

template <typename T>
struct is_row : std::false_type {};

template <typename T, typename D, bool O>
struct is_row<marray_base<T, 1, D, O>> : std::true_type {};

template <typename T>
struct is_row<marray_view<T, 1>> : std::true_type {};

template <typename T, typename A>
struct is_row<marray<T, 1, A>> : std::true_type {};

template <typename T, typename U, typename=void>
struct is_row_of : std::false_type {};

template <typename T, typename U>
struct is_row_of<T, U, std::enable_if_t<is_row<T>::value &&
    std::is_assignable<U&,typename T::value_type>::value>> : std::true_type {};

template <typename T, typename U>
struct is_1d_container_of :
    std::integral_constant<bool, is_row_of<T,U>::value ||
                                 is_container_of<T,U>::value> {};

template <typename T, typename U, typename V=void>
using enable_if_1d_container_of_t = std::enable_if_t<is_1d_container_of<T,U>::value, V>;

template <typename T>
std::enable_if_t<is_container<T>::value,len_type>
length(const T& len)
{
    return len.size();
}

template <typename T>
std::enable_if_t<is_row<T>::value,len_type>
length(const T& len)
{
    return len.length();
}

}

/**
 * Adaptor class which can capture a container, initializer_list, @ref row, or @ref row_view with elements
 * convertible to a specified type.
 *
 * This class is used to capture a variety of acceptable argument types without relying on template parameters.
 * For example:
 *
 * @code{.cxx}
 * void foo(const array_1d<int>&);
 *
 * // An initializer_list<int> (sort of...actually narrowing conversions are allowed):
 * foo({1, 4, 4l, 3u});
 *
 * // A linear container:
 * std::vector<long> v{1, 5, 2};
 * foo(v);
 *
 * // A non-linear container:
 * std::list<int> l{1, 6, 2, 0};
 * foo(l);
 *
 * // A row (1-d tensor):
 * row<int> r{3};
 * r[0] = 0;
 * r[1] = 4;
 * r[2] = 2;
 * foo(r);
 *
 * // A row_view:
 * row_view<int> rv({v.size()}, v.data());
 * foo(rv);
 * @endcode
 *
 * @tparam T  Elements of the supplied container, initializer list, etc. must be convertible to this type.
 *
 * @ingroup util
 */
template <typename T>
class array_1d
{
    protected:
        struct len_wrapper
        {
            T len;

            template <typename U, typename=std::enable_if_t<std::is_convertible_v<U,T>>>
            len_wrapper(const U& len) : len{(T)len} {}

            static auto extract(std::initializer_list<len_wrapper> il)
            {
                short_vector<T,MARRAY_OPT_NDIM> v;
                for (auto& lw : il) v.push_back(lw.len);
                return v;
            }
        };

        struct adaptor_base
        {
            len_type len;

            adaptor_base(len_type len) : len(len) {}

            virtual ~adaptor_base() {}

            virtual void slurp(T*) const = 0;

            virtual adaptor_base& copy(adaptor_base& other) = 0;

            virtual adaptor_base& move(adaptor_base& other) = 0;
        };

        template <typename U>
        struct adaptor : adaptor_base
        {
            U data;
            using adaptor_base::len;

            adaptor(U data)
            : adaptor_base(detail::length(data)), data(std::move(data)) {}

            virtual void slurp(T* x) const override
            {
                std::copy_n(data.begin(), len, x);
            }

            virtual adaptor_base& copy(adaptor_base& other) override
            {
                return *(new (static_cast<adaptor*>(&other)) adaptor(*this));
            }

            virtual adaptor_base& move(adaptor_base& other) override
            {
                return *(new (static_cast<adaptor*>(&other)) adaptor(std::move(*this)));
            }
        };

        static constexpr size_t _s1 = sizeof(adaptor<T*&>);
        static constexpr size_t _s2 = sizeof(adaptor<short_vector<T,MARRAY_OPT_NDIM>>);
        static constexpr size_t max_adaptor_size = (_s1 > _s2 ? _s1 : _s2);

        std::aligned_storage_t<max_adaptor_size> raw_adaptor_;
        adaptor_base& adaptor_;

        template <typename U>
        adaptor_base& adapt(U&& data)
        {
            return *(new (&raw_adaptor_) adaptor<U>(data));
        }

    public:
        array_1d()
        : adaptor_(adapt(std::array<T,0>{})) {}

        array_1d(const array_1d& other)
        : adaptor_(other.adaptor_.copy(reinterpret_cast<adaptor_base&>(raw_adaptor_))) {}

        array_1d(array_1d&& other)
        : adaptor_(other.adaptor_.move(reinterpret_cast<adaptor_base&>(raw_adaptor_))) {}

        array_1d(std::initializer_list<len_wrapper> il)
        : adaptor_(adapt(len_wrapper::extract(il))) {}

        template <typename U, typename=detail::enable_if_1d_container_of_t<U,T>>
        array_1d(const U& data)
        : adaptor_(adapt(data)) {}

        ~array_1d() { adaptor_.~adaptor_base(); }

        template <size_t N>
        void slurp(std::array<T, N>& x) const
        {
            MARRAY_ASSERT((len_type)N >= size());
            adaptor_.slurp(x.data());
        }

        void slurp(std::vector<T>& x) const
        {
            x.resize(size());
            adaptor_.slurp(x.data());
        }

        template <size_t N>
        void slurp(short_vector<T, N>& x) const
        {
            x.resize(size());
            adaptor_.slurp(x.data());
        }

        void slurp(row<T>& x) const
        {
            x.reset({size()});
            adaptor_.slurp(x.data());
        }

        len_type size() const { return adaptor_.len; }
};

}

#endif //MARRAY_ARRAY_1D_HPP
