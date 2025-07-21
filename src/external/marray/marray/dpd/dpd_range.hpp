#ifndef MARRAY_DPD_RANGE_HPP
#define MARRAY_DPD_RANGE_HPP

#include "../range.hpp"
#include "../marray_base.hpp"
#include <type_traits>

namespace MArray
{

class dpd_index
{
    protected:
        int irrep_;
        len_type idx_;

    public:
        dpd_index(int irrep, len_type idx)
        : irrep_(irrep), idx_(idx) {}

        int irrep() const { return irrep_; }

        len_type idx() const { return idx_; }
};

class dpd_range : public std::array<range_t<len_type>, 8>
{
    public:
        dpd_range() : std::array<range_t<len_type>, 8>{} {}

        dpd_range(const array_1d<len_type>& to_)
        {
            MARRAY_ASSERT(to_.size() < 8);

            len_vector to; to_.slurp(to);

            for (auto i : range(to.size()))
                (*this)[i] = range(to[i]);
        }

        dpd_range(const array_1d<len_type>& from_,
                  const array_1d<len_type>& to_)
        {
            MARRAY_ASSERT(from_.size() == to_.size());
            MARRAY_ASSERT(from_.size() < 8);

            len_vector from; from_.slurp(from);
            len_vector to; to_.slurp(to);

            for (auto i : range(from.size()))
                (*this)[i] = range(from[i], to[i]);
        }

        dpd_range(const array_1d<len_type>& from_,
                  const array_1d<len_type>& to_,
                  const array_1d<len_type>& delta_)
        {
            MARRAY_ASSERT(from_.size() == to_.size());
            MARRAY_ASSERT(from_.size() == delta_.size());
            MARRAY_ASSERT(from_.size() < 8);

            len_vector from; from_.slurp(from);
            len_vector to; to_.slurp(to);
            len_vector delta; delta_.slurp(delta);

            for (auto i : range(from.size()))
                (*this)[i] = range(from[i], to[i], delta[i]);
        }

        dpd_range(int irrep, const range_t<len_type>& x)
        : std::array<range_t<len_type>, 8>{}
        {
            MARRAY_ASSERT(irrep >= 0 && irrep < 8);

            (*this)[irrep] = x;
        }

        dpd_range operator()(int irrep, const range_t<len_type>& x)
        {
            MARRAY_ASSERT(irrep >= 0 && irrep < 8);

            dpd_range ret(*this);
            ret[irrep] = x;
            return ret;
        }
};

namespace detail
{

template <typename T>
struct is_dpd_index_or_slice_helper : std::bool_constant<std::is_convertible_v<T,dpd_index> ||
                                                         std::is_convertible_v<T,dpd_range> ||
                                                         std::is_same_v<T,all_t>> {};

template <typename T>
struct is_dpd_index_or_slice : is_dpd_index_or_slice_helper<typename std::decay<T>::type> {};

template <typename... Args>
struct are_dpd_indices_or_slices;

template<>
struct are_dpd_indices_or_slices<> : std::true_type {};

template <typename Arg, typename... Args>
struct are_dpd_indices_or_slices<Arg, Args...> :
    std::conditional_t<is_dpd_index_or_slice<Arg>::value,
                       are_dpd_indices_or_slices<Args...>,
                       std::false_type> {};

template <typename... Args>
struct sliced_dimension;

template <>
struct sliced_dimension<>
{
    static constexpr int value = 0;
};

template <typename Arg, typename... Args>
struct sliced_dimension<Arg, Args...>
{
    static constexpr int value =
        !std::is_same<std::decay_t<Arg>,dpd_index>::value +
        sliced_dimension<Args...>::value;
};

}

}

#endif //MARRAY_DPD_RANGE_HPP
