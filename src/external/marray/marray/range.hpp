#ifndef MARRAY_RANGE_HPP
#define MARRAY_RANGE_HPP

#include <type_traits>
#include <iterator>
#include <limits>

#include "detail/utility.hpp"

#define MARRAY_ASSERT_RANGE_IN(x, from, to) \
MARRAY_ASSERT((x).size() >= 0); \
if ((x).size()) \
{ \
    MARRAY_ASSERT((x).front() >= (from) && (x).front() < (to)); \
    MARRAY_ASSERT((x).back() >= (from) && (x).back() < (to)); \
}

namespace MArray
{

namespace detail
{

template <typename T, bool=std::is_enum_v<T>>
struct underlying_type_if;

template <typename T>
struct underlying_type_if<T, false>
{
    typedef std::make_signed_t<T> type;
};

template <typename T>
struct underlying_type_if<T, true>
{
    typedef std::make_signed_t<std::underlying_type_t<T>> type;
};

template <typename... Ts> struct are_numeric;

template <> struct are_numeric<> : std::true_type {};

template <typename T, typename... Ts> struct are_numeric<T, Ts...>
: std::integral_constant<bool, (std::is_integral<T>::value ||
                                std::is_enum<T>::value) &&
                               are_numeric<Ts...>::value> {};

template <typename... Ts> using enable_if_numeric =
    std::enable_if_t<are_numeric<Ts...>::value>;

}

template <typename T>
class range_t
{
    static_assert(std::is_integral<T>::value, "The type must be integral.");

    template <typename U> friend class range_t;

    protected:
        // NB: ordering the fields from_, to_, delta_ triggers a strange bug on
        // clang 16 (https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGEgBykrgAyeAyYAHI%2BAEaYxCAAzKQADqgKhE4MHt6%2BASlpGQKh4VEssfFJdpgOmUIETMQE2T5%2BXIFVNQJ1DQTFkTFxibb1jc25bcM9faXliQCUtqhexMjsHOYJYcjeWADUJgluBACeyZgA%2BgTETIQKB9gmGgCCG1s7mPuHhHFMRMT3jxeZk2DG2Xj2BzcADdqn8Ac9AQRMCxkgYkZ8jqdGKwPgAVeFPbZMBQKXbXIwXAiAkwAdisz12jN2yWIqCRDkw6BAgKZvN2uN2VFZLAO9KefKZAqI51FPIl/N2WFo9RlCTFcolAHpNfyEHhSQB3EgAa1JxN2mFUZw5XI1fO1CqVKtlDPlDoFQtQIrVdt57t20pdL1djOSXmitDwyG5Ib5RJJu2%2B1zhsd5tLF8r5LLZsM5MfFmflAqhYlVGcLEoFTqYZepqflYYjUfzFYlXnSRkTSOTJHOoiRwBIx0%2BABFdgoCFyQOT0F7zkxkKsSeck79e/VgEHW3z22FgLsS95KVjR/yt9umbvO/gqFQ4oxVpcTwcxxOp8krjeqJdzxfdlf91SMJu1PXEACpfwvACyUwO9iAfD4XzPMwADYg19Vs0AYCdLRZLsfj%2BCA5l2EAD1LCANAWRVMGVGsIC4Yj01pEc6wLP8sJwq1iHwntiAgYsxFIR0aPqYjSMPc4IEPKjq0k6tGLpZjWL/RlolQTxdlQM5eJfF8IA4ggeLXf5UM0ggEDiYiDIwismPrP94IIZYGDI2hzlHJC2Qs4gADoJP2VDzBQ6jaPc3SEjHLy4h82TIO3JSEXsws1I0rSCJIMAwD0gyjLhUyouIKyBAnGzCzstiVMZRznNc9zMs88zov82k3BakKVV2eqIrM7yYpEms4tbBLgwq7dDy8Y8zk07TjLAojdmspLM3KyreWq4gXIkwbbJpFjEtG1tVzy4K0p0yxrCI0rlsUpbW2a87utin1bordaXLA8z9W2srdtKpTnpGzMDNw7jyWASl%2BIDVAxPauiGKE6UICIBSrF%2B/bMyOkhdliYAwnmxaDsZFbWze3KSAooTZLmb7htKzHuNcfHiqpJLidezAnI2sm%2BOlSn%2BvOamAeuvaXl%2BgHEWRVFfkQw4TjOZg2DPBIHmeJgvCIMlDHByGGGphEboq0mwcpSF8WViBdfQtGRqhVA8HQQV1IgYDdkt/Xy0Zd09VJBQECWWgHe0dtDPghQvGVRMXKYcc8BRegArMRyAEdzDMO01Y11RT2Ni3BY9x3uIgDPUETEjdlUYipLt9A5g0K2RY4BZaE4ABWXg/G4XhUE4VrzsscclhWRDgR4UgCE0RuFmNRIW58hIUMkFCaUkDQ2jaFuEmkZuOEkduJ9IbuOF4BQQA0MeJ4WOBYCQNAUToOJyEoW/knv%2BIiSMLgUI0M%2BaGVOIT4gNEfe0QwgNGOJwUeIDmDEGOAAeWiNoWEEDeC3zYIIWBDBaDgI4FoUgWBoheGAG4MQtAT6dzwciLW4hyH4HgjUGEZDcGWmqOrNYo9gI0WQfoPA0RrgwI8FgfeVxY5cJhMQNSSgRyUKMJGIwF8%2BAGGAAoAAangTABpYHyy4fwQQIgxDsCkDIQQigVDqBwToPQBg5GmD7jYSM0QT6QAWFpRwxVOAAFo3wvhsVYSwXAaS7HcbAhIXcxHEHtpgRxRFbA0VhJkFwDB3CeBaHoEIYR%2BhlEGFwfI6RXFZGSeMHJhQGDTAGPEbJHQ8ndFGAU1oMT7BVJGL0dJMwsmTBqTkOpE4pgtLKRIBYChB6rH6foVue9zEH04OXfwKF3ELwWlY/cn8fIaBWbsCAuBCBYw2AxXg49zFzCnokLgc9V4AE4zmoUkP4a5XAEgJBbqMne4zcGH2Pqfc%2BBzSBX0QCAJYBAwwEEfvpL0L96DEAiDiTgqgZlzMkAsrWuxlmrKYfgP49s9A6OEKIcQhisUmLUPvXQ2SDTXGSMgpuYzSAd1eZwWB6tAWaSoNM2Z8z35LJQistZEAPB33BQFBIuzPlaEOaQae9y573KldK6VTzd7Uv3m82wHz9kispRwMwLyu6cD2RfBYYj0jOEkEAA%3D%3D%3D)
        T delta_;
        T from_;
        T to_;

        typedef T value_type;
        typedef T size_type;

    public:
        class iterator
        {
            protected:
                T val_;
                T delta_;

            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = T;
                using difference_type = std::ptrdiff_t;
                using pointer = T*;
                using reference = T&;

                constexpr iterator() : val_(0), delta_(0) {}

                constexpr iterator(T val, T delta) : val_(val), delta_(delta) {}

                bool operator==(const iterator& other) const
                {
                    return val_ == other.val_ && delta_ == other.delta_;
                }

                auto operator<=>(const iterator& other) const
                {
                    return val_ <=> other.val_;
                }

                value_type operator*() const
                {
                    return val_;
                }

                iterator& operator++()
                {
                    val_ += delta_;
                    return *this;
                }

                iterator operator++(int)
                {
                    iterator old(*this);
                    val_ += delta_;
                    return old;
                }

                iterator& operator--()
                {
                    val_ -= delta_;
                    return *this;
                }

                iterator operator--(int)
                {
                    iterator old(*this);
                    val_ -= delta_;
                    return old;
                }

                iterator& operator+=(difference_type n)
                {
                    val_ += n*delta_;
                    return *this;
                }

                iterator operator+(difference_type n) const
                {
                    return iterator(val_+n*delta_, delta_);
                }

                friend iterator operator+(difference_type n, const iterator& i)
                {
                    return iterator(i.val_+n*i.delta_, i.delta_);
                }

                iterator& operator-=(difference_type n)
                {
                    val_ -= n*delta_;
                    return *this;
                }

                iterator operator-(difference_type n) const
                {
                    return iterator(val_-n*delta_, delta_);
                }

                difference_type operator-(const iterator& other) const
                {
                    return (val_-other.val_)/delta_;
                }

                value_type operator[](difference_type n) const
                {
                    return val_+n*delta_;
                }

                friend void swap(iterator& a, iterator& b)
                {
                    using std::swap;
                    swap(a.val_, b.val_);
                    swap(a.delta_, b.delta_);
                }
        };

        constexpr range_t()
        : delta_(1), from_(0), to_(0) {}

        constexpr range_t(T to)
        : delta_(1), from_(0), to_(to) {}

        constexpr range_t(T from, T to)
        : delta_(1), from_(from), to_(to) {}

        constexpr range_t(T from, T to, T delta) __attribute__((always_inline))
        : delta_(delta), from_(from)
        {
            if (delta > 0)
                to_ = from+((to-from+delta-1)/delta)*delta;
            else if (delta < 0)
                to_ = from+((to-from+delta+1)/delta)*delta;
        }

        range_t(const range_t&) = default;

        template <typename U>
        range_t(const range_t<U>& other)
        : from_(other.from_), to_(other.to_), delta_(other.delta_) {}

        range_t(range_t&&) = default;

        range_t& operator=(const range_t&) = default;

        template <typename U>
        range_t& operator=(const range_t<U>& other)
        {
            from_ = other.from_;
            to_ = other.to_;
            delta_ = other.delta_;
            return *this;
        }

        range_t& operator=(range_t&&) = default;

        value_type step() const
        {
            return delta_;
        }

        size_type size() const
        {
            return (delta_ == 0 ? std::numeric_limits<size_type>::max() :
                    (to_-from_)/delta_);
        }

        iterator begin() const
        {
            return iterator(from_, delta_);
        }

        iterator end() const
        {
            return iterator(to_, delta_);
        }

        value_type front() const
        {
            return from_;
        }

        value_type back() const
        {
            return to_-delta_;
        }

        value_type from() const
        {
            return from_;
        }

        value_type to() const
        {
            return to_;
        }

        value_type operator[](size_type n) const
        {
            return from_+n*delta_;
        }

        template <typename U, typename=
            typename std::enable_if<detail::is_container<U>::value>::type>
        operator U() const
        {
            return U(begin(), end());
        }

        range_t& operator+=(T shift)
        {
            from_ += shift;
            to_ += shift;
            return *this;
        }

        range_t& operator-=(T shift)
        {
            from_ -= shift;
            to_ -= shift;
            return *this;
        }

        range_t operator+(T shift) const
        {
            range_t shifted(*this);
            shifted += shift;
            return shifted;
        }

        range_t operator-(T shift) const
        {
            range_t shifted(*this);
            shifted -= shift;
            return shifted;
        }

        /**
         * Shift a range to the right.
         *
         * @param shift     The amount to shift, in units of the range step size.
         *
         * @param other     The range to shift.
         *
         * @return          The shifted range. After shifting, `other[i]` becomes
         *                  `other[i] + shift*other.step()`.
         *
         * @ingroup range
         */
        friend range_t operator+(T shift, const range_t& other)
        {
            return other + shift;
        }

        /**
         * Shift a range to the left.
         *
         * @param shift     The amount to shift, in units of the range step size.
         *
         * @param other     The range to shift.
         *
         * @return          The shifted range. After shifting, `other[i]` becomes
         *                  `other[i] - shift*other.step()`.
         *
         * @ingroup range
         */
        friend range_t operator-(T shift, const range_t& other)
        {
            return range_t(shift - other.from_, shift - other.to_, -other.delta_);
        }

        /**
         * Concatenate two adjacent ranges.
         *
         * @param other     The range to concatenate with `this`. The ranges must be
         *                  adjacent, meaning that `this.to() == other.from()`, and
         *                  both ranges must have the same step.
         *
         * @return          The combined range.
         */
        range_t operator|(const range_t& other) const
        {
            MARRAY_ASSERT(step() == other.step());
            MARRAY_ASSERT(to() == other.from());
            return range_t{from(), other.to(), step()};
        }

        /**
         * Append an element to a range.
         *
         * @param lhs   The range to which to append.
         *
         * @param rhs   A range element. Must be equal to `lhs.to()`, that is, the element
         *              immediately following the last element of `lhs` during iteration.
         *
         * @return      The combined range.
         *
         * @ingroup range
         */
        friend range_t operator|(const range_t& lhs, T rhs)
        {
            MARRAY_ASSERT(lhs.to() == rhs);
            return range_t{lhs.from(), lhs.to()+lhs.step(), lhs.step()};
        }

        /**
         * Prepend an element to a range.
         *
         * @param lhs   A range element. Must be equal to `rhs.from()-rhs.step()`, that is,
         *              the element immediately preceding the first element of `rhs` during iteration.
         *
         * @param rhs   The range to which to prepend.
         *
         * @return      The combined range.
         *
         * @ingroup range
         */
        friend range_t operator|(T lhs, const range_t& rhs)
        {
            MARRAY_ASSERT(lhs+rhs.step() == rhs.from());
            return range_t{lhs, rhs.to(), rhs.step()};
        }

        /**
         * Create a range with reversed iteration order.
         *
         * @return  A range which iterates over the same set of elements as `this`,
         *          but in reverse order.
         */
        range_t reverse()
        {
            return range_t(back(), front()-step(), -step());
        }

        /**
         * Test for emptiness.
         *
         * @return  `size() == 0`.
         */
        bool empty() const
        {
            return size() == 0;
        }

        /**
         * Truth test for non-zero size.
         *
         * @return  `size() != 0`, or equivalently `!empty()`.
         */
        explicit operator bool() const
        {
            return size();
        }
};

/**
 * The range `[0,to)`.
 *
 * @param to    The value one higher than the last element in the range.
 *              This is equal to the number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @return      Range object, of the same type as `to`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename=
    detail::enable_if_numeric<T>>
auto range(T to)
{
    typedef typename detail::underlying_type_if<T>::type U;
    return range_t<U>{U(to)};
}

/**
 * The range `[from,to)`.
 *
 * @param from  The value of the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param to    The value one higher than the last element in the range.
 *              The number of elements is equal to `to-from`. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`
 *              and `to`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename=
    detail::enable_if_numeric<T,U>>
auto range(T from, U to)
{
    typedef decltype(std::declval<T>() + std::declval<U>()) V0;
    typedef typename detail::underlying_type_if<V0>::type V;
    if ((V)to < (V)from)
        to = from;
    return range_t<V>{(V)from, (V)to};
}

/**
 * The range `[from,to)` with spacing `delta`.
 *
 * @param from  The value of the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param to    The value higher than all elements in the range. Must be
 *              an integral or enum type.
 *
 * @param delta The distance between consecutive elements in the range.
 *              The number of elements is equal to `ceil((to-from)/delta)`. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`,
 *              `to`, and `delta`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename V, typename=
    detail::enable_if_numeric<T,U,V>>
auto range(T from, U to, V delta)
{
    typedef decltype(std::declval<T>() + std::declval<U>() + std::declval<V>()) W0;
    typedef typename detail::underlying_type_if<W0>::type W;
    if ((W() < (W)delta && (W)to < (W)from) ||
        ((W)delta < W() && (W)from < (W)to))
        to = from;
    return range_t<W>{(W)from, (W)to, (W)delta};
}

/**
 * The range `[from,from+N)`.
 *
 * @param from  The value of the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param N     The number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`
 *              and `N`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename=
    detail::enable_if_numeric<T,U>>
auto rangeN(T from, U N)
{
    typedef decltype(std::declval<T>() + std::declval<U>()) V0;
    typedef typename detail::underlying_type_if<V0>::type V;
    return range_t<V>{V(from), V(from+N)};
}

/**
 * The range `[from,from+N*delta)` with spacing `delta`.
 *
 * @param from  The value of the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param N     The number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @param delta The distance between consecutive elements in the range.
 *              Must be an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`,
 *              `N`, and `delta`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename V, typename=
    detail::enable_if_numeric<T,U,V>>
auto rangeN(T from, U N, V delta)
{
    typedef decltype(std::declval<T>() + std::declval<U>() + std::declval<V>()) W0;
    typedef typename detail::underlying_type_if<W0>::type W;
    return range_t<W>{W(from), W(from+N*delta), W(delta)};
}

/**
 * The range `[0,to)` in reverse order.
 *
 * @param to    The value one higher than the first element in the range.
 *              This is equal to the number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @return      Range object, of the same type as `to`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename=
    detail::enable_if_numeric<T>>
auto reversed_range(T to)
{
    return range(to).reverse();
}

/**
 * The range `[from,to)` in reverse order.
 *
 * @param from  The value of the last element in the range. Must be
 *              an integral or enum type.
 *
 * @param to    The value one higher than the first element in the range.
 *              The number of elements is equal to `to-from`. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`
 *              and `to`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename=
    detail::enable_if_numeric<T,U>>
auto reversed_range(T from, U to)
{
    return range(from, to).reverse();
}

/**
 * The range `[from,to)` with spacing `delta` in reverse order.
 *
 * @param from  The value of the last element in the range. Must be
 *              an integral or enum type.
 *
 * @param to    The value one higher than the first element in the range.
 *              Must be an integral or enum type.
 *
 * @param delta The distance between consecutive elements in the range.
 *              The number of elements is equal to `ceil(to-from)/delta)`. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`,
 *              `to`, and `delta`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename V, typename=
    detail::enable_if_numeric<T,U,V>>
auto reversed_range(T from, U to, V delta)
{
    return range(from, to, delta).reverse();
}

/**
 * The range `[to-N,to)` in reverse order.
 *
 * @param to    The value one larger than the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param N     The number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`
 *              and `N`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename=
    detail::enable_if_numeric<T,U>>
auto reversed_rangeN(T to, U N)
{
    return range(to-1, to-N-1, -1);
}

/**
 * The range `[to-N*delta,to)` with spacing `delta` in reverse order.
 *
 * @param to    The value one larger than the first element in the range. Must be
 *              an integral or enum type.
 *
 * @param N     The number of elements in the range. Must be
 *              an integral or enum type.
 *
 * @param delta The distance between consecutive elements in the range.
 *              Must be an integral or enum type.
 *
 * @return      Range object, whose type is the common arithmetic type of `from`
 *              and `N`, that can be used to index a tensor.
 *
 * @ingroup range
 */
template <typename T, typename U, typename V, typename=
    detail::enable_if_numeric<T,U,V>>
auto reversed_rangeN(T to, U N, V delta)
{
    return range(to-delta, to-(N+1)*delta, -delta);
}

} // namespace MArray

#endif //MARRAY_RANGE_HPP
