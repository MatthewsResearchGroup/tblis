#ifndef MARRAY_DPD_MARRAY_BASE_HPP
#define MARRAY_DPD_MARRAY_BASE_HPP

#include "../marray.hpp"
#include "../array_2d.hpp"
#include "dpd_range.hpp"
#include <limits>
#include <climits>

namespace MArray
{

struct dpd_layout
{
    int type;

    constexpr explicit dpd_layout(int type) : type(type) {}

    dpd_layout(layout layout);

    layout base() const;

    bool operator==(const dpd_layout&) const = default;
};

struct balanced_column_major_layout : dpd_layout
{ constexpr balanced_column_major_layout() : dpd_layout(0) {} };
constexpr balanced_column_major_layout BALANCED_COLUMN_MAJOR;

struct balanced_row_major_layout : dpd_layout
{ constexpr balanced_row_major_layout() : dpd_layout(1) {} };
constexpr balanced_row_major_layout BALANCED_ROW_MAJOR;

struct blocked_column_major_layout : dpd_layout
{ constexpr blocked_column_major_layout() : dpd_layout(2) {} };
constexpr blocked_column_major_layout BLOCKED_COLUMN_MAJOR;

struct blocked_row_major_layout : dpd_layout
{ constexpr blocked_row_major_layout() : dpd_layout(3) {} };
constexpr blocked_row_major_layout BLOCKED_ROW_MAJOR;

struct prefix_column_major_layout : dpd_layout
{ constexpr prefix_column_major_layout() : dpd_layout(4) {} };
constexpr prefix_column_major_layout PREFIX_COLUMN_MAJOR;

struct prefix_row_major_layout : dpd_layout
{ constexpr prefix_row_major_layout() : dpd_layout(5) {} };
constexpr prefix_row_major_layout PREFIX_ROW_MAJOR;

constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(BALANCED)) BALANCED;
constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(BLOCKED)) BLOCKED;
constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(PREFIX)) PREFIX;

inline dpd_layout::dpd_layout(layout layout)
: type(layout == DEFAULT_LAYOUT ? MARRAY_DEFAULT_DPD_LAYOUT.type :
       layout == ROW_MAJOR      ? MARRAY_PASTE(MARRAY_DEFAULT_DPD_LAYOUT,_ROW_MAJOR).type
                                : MARRAY_PASTE(MARRAY_DEFAULT_DPD_LAYOUT,_COLUMN_MAJOR).type) {}

inline layout dpd_layout::base() const
{
    if (*this == BALANCED_COLUMN_MAJOR ||
        *this == BLOCKED_COLUMN_MAJOR ||
        *this == PREFIX_COLUMN_MAJOR)
    {
        return COLUMN_MAJOR;
    }
    else
    {
        return ROW_MAJOR;
    }
}

template <typename Type, typename Derived, bool Owner>
class dpd_marray_base;

template <typename Type>
class dpd_marray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class dpd_marray;

class irrep_iterator
{
    protected:
        const unsigned irrep_;
        const int ndim_;
        const unsigned irrep_bits_;
        const unsigned irrep_mask_;
        const unsigned nblock_;
        unsigned block_;

    public:
        irrep_iterator(int irrep, int nirrep, int ndim)
        : irrep_(irrep), ndim_(ndim), irrep_bits_(__builtin_popcount(nirrep-1)),
          irrep_mask_(nirrep-1), nblock_(ndim == 0 ? (irrep == 0 ? 1 : 0) : 1u << (irrep_bits_*(ndim-1)))
        {
            MARRAY_ASSERT(ndim > 0);
            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);
            MARRAY_ASSERT(irrep >= 0 && irrep < nirrep);
            reset();
        }

        bool next()
        {
            bool done = ++block_ == nblock_;
            if (done) reset();
            return !done;
        }

        int nblock() const { return nblock_; }

        void block(int block)
        {
            MARRAY_ASSERT(block >= 0 && block < (int)nblock_);
            block_ = block;
        }

        int block() const
        {
            return block_;
        }

        void reset()
        {
            block_ = std::numeric_limits<unsigned int>::max();
        }

        int irrep(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < ndim_);

            if (dim == 0)
            {
                auto irr0 = irrep_;
                for (auto i : range(ndim_-1))
                    irr0 ^= (block_ >> irrep_bits_*i) & irrep_mask_;
                return irr0;
            }

            return (block_ >> irrep_bits_*(dim-1)) & irrep_mask_;
        }
};

namespace detail
{

struct dpd_base
{
    dpd_stride_vector2 size_;
    dpd_len_vector len_;
    dpd_len_vector2 off_;
    dpd_stride_vector stride_;
    dim_vector leaf_;
    dim_vector2 parent_;
    dim_vector2 fixed_;
    dim_vector perm_;
    int irrep_ = 0;
    int nirrep_ = 0;

    void reset()
    {
        size_.clear();
        len_.clear();
        off_.clear();
        stride_.clear();
        leaf_.clear();
        parent_.clear();
        fixed_.clear();
        perm_.clear();
        irrep_ = 0;
        nirrep_ = 0;
    }

    void reset(const dpd_base& other)
    {
        size_ = other.size_;
        len_ = other.len_;
        off_ = other.off_;
        stride_ = other.stride_;
        leaf_ = other.leaf_;
        parent_ = other.parent_;
        fixed_ = other.fixed_;
        perm_ = other.perm_;
        irrep_ = other.irrep_;
        nirrep_ = other.nirrep_;
    }

    void reset(int irrep, int nirrep,
               const array_2d<len_type>& len,
               const array_1d<int>& depth_,
               layout layout)
    {
        MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                      nirrep == 4 || nirrep == 8);

        auto ndim = depth_.size();
        MARRAY_ASSERT(ndim > 0);
        MARRAY_ASSERT(irrep >= 0 && irrep < nirrep);
        MARRAY_ASSERT(len.length(1) >= nirrep);
        MARRAY_ASSERT(len.length(0) >= ndim);

        irrep_ = irrep;
        nirrep_ = nirrep;
        size_.resize(2*ndim-1);
        len.slurp(len_);
        off_.resize(2*ndim-1);
        stride_.resize(ndim, {1,1,1,1,1,1,1,1});
        leaf_.resize(ndim);
        parent_.resize(2*ndim-1);
        fixed_.resize(2*ndim-1);
        perm_.resize(ndim);

        dim_vector depth;
        depth_.slurp(depth);
        set_tree(depth);
        set_size(layout);
    }

    void swap(dpd_base& other)
    {
        using std::swap;
        swap(size_, other.size_);
        swap(len_, other.len_);
        swap(off_, other.off_);
        swap(stride_, other.stride_);
        swap(leaf_, other.leaf_);
        swap(parent_, other.parent_);
        swap(fixed_, other.fixed_);
        swap(perm_, other.perm_);
        swap(irrep_, other.irrep_);
        swap(nirrep_, other.nirrep_);
    }

    static dim_vector default_depth(dpd_layout layout, int ndim)
    {
        dim_vector depth;

        auto log2_floor = [&](int x)
        {
            return sizeof(x)*CHAR_BIT - __builtin_clz(x) - 1;
        };

        auto is_power_of_two = [&](int x)
        {
            return (x & (x-1)) == 0;
        };

        auto log2_ceil = [&](int x)
        {
            return log2_floor(x) + (is_power_of_two(x) ? 0 : 1);
        };

        if (layout == BALANCED_ROW_MAJOR ||
            layout == BALANCED_COLUMN_MAJOR)
        {
            int dl = log2_floor(ndim);
            int du = log2_ceil(ndim);
            int o = ndim - (1<<dl);
            int k = 2*((o+1)/2);
            int l = 2*(o/2);
            int h = (ndim+1)/2;

            depth.insert(depth.end(), k, du);
            depth.insert(depth.end(), h-k, dl);
            depth.insert(depth.end(), l, du);
            depth.insert(depth.end(), ndim-h-l, dl);
        }
        else if (layout == PREFIX_ROW_MAJOR ||
                 layout == PREFIX_COLUMN_MAJOR)
        {
            depth.push_back(ndim-1);
            for (auto i : range(1,ndim)) depth.push_back(ndim-i);
        }
        else //if (layout == BLOCKED_ROW_MAJOR ||
             //    layout == BLOCKED_COLUMN_MAJOR)
        {
            for (auto i : range(1,ndim)) depth.push_back(i);
            depth.push_back(ndim-1);
        }

        return depth;
    }

    void set_tree(dim_vector depth)
    {
        int ndim = leaf_.size();
        dim_vector node(ndim);
        len_vector leaf_idx = range(ndim);

        MARRAY_ASSERT((int)depth.size() == ndim);
        for (auto i : depth)
            MARRAY_ASSERT(i < ndim);

        auto pos = 0;
        for (auto d : reversed_range(ndim))
        {
            auto i = 0;
            while (i < (int)depth.size())
            {
                /*
                 * If we encounter a pair of nodes with depth greater than the
                 * current level, combine them into an interior node and
                 * assign the parent pointers.
                 */
                if (depth[i] == d+1)
                {
                    MARRAY_ASSERT(i+1 < (int)depth.size() && depth[i+1] == d+1);
                    parent_[node[i]] = parent_[node[i+1]] = pos;
                    depth.erase(depth.begin()+i+1);
                    depth[i]--;
                    node.erase(node.begin()+i+1);
                    node[i] = pos++;
                    leaf_idx.erase(leaf_idx.begin()+i+1);
                    leaf_idx[i] = -1;
                }
                /*
                 * For nodes on the current depth level, assign a position
                 * in the linearized tree.
                 */
                else if (depth[i] == d)
                {
                    node[i] = pos++;
                    if (leaf_idx[i] != -1) leaf_[leaf_idx[i]] = node[i];
                }

                i++;
            }
        }

        MARRAY_ASSERT(pos == 2*ndim-1);
        MARRAY_ASSERT(depth.size() == 1);
        MARRAY_ASSERT(depth[0] == 0);
    }

    void set_size(layout layout)
    {
        auto ndim = perm_.size();

        if (layout == COLUMN_MAJOR)
        {
            // Column major
            for (auto i : range(ndim))
            {
                std::copy_n(&len_[i][0], nirrep_, &size_[leaf_[i]][0]);
                perm_[i] = i;
            }
        }
        else
        {
            // Row major: reverse the dimensions and treat as
            // permuted column major
            for (auto i : range(ndim))
            {
                std::copy_n(&len_[i][0], nirrep_, &size_[leaf_[ndim-1-i]][0]);
                perm_[i] = ndim-1-i;
            }

            for (auto i : range(ndim/2))
            for (auto j : range(nirrep_))
                std::swap(len_[i][j], len_[ndim-1-i][j]);
        }

        for (auto i : range(ndim-1))
        {
            auto next = parent_[2*i];

            for (auto irr1 : range(nirrep_))
            {
                size_[next][irr1] = 0;
                for (auto irr2 : range(nirrep_))
                    size_[next][irr1] += size_[2*i][irr1^irr2]*size_[2*i+1][irr2];
            }
        }
    }

    template <typename U, typename V, typename W, typename X>
    void get_block(const U& irreps, V& len, W& data, X& stride) const
    {
        auto ndim = perm_.size();
        auto ndim2 = size_.size();

        MARRAY_ASSERT(irreps.size() >= ndim);
        MARRAY_ASSERT(len.size() >= ndim);
        MARRAY_ASSERT(stride.size() >= ndim);

        dim_vector2 dpd_irrep = fixed_;
        stride_vector2 dpd_stride(ndim2);
        dpd_stride[ndim2-1] = 1;

        for (auto i : range(ndim))
        {
            MARRAY_ASSERT(irreps[i] >= 0 && irreps[i] < nirrep_);
            dpd_irrep[leaf_[perm_[i]]] = irreps[i];
        }

        for (auto i : range(ndim2/2))
            dpd_irrep[parent_[2*i]] = dpd_irrep[2*i]^dpd_irrep[2*i+1];

        for (auto i : reversed_range(ndim2/2))
        {
            dpd_stride[2*i] = dpd_stride[parent_[2*i]];
            dpd_stride[2*i+1] = dpd_stride[2*i]*size_[2*i][dpd_irrep[2*i]];

            stride_type offset = off_[2*i][dpd_irrep[2*i]]*dpd_stride[2*i] +
                                 off_[2*i+1][dpd_irrep[2*i+1]]*dpd_stride[2*i+1];
            for (auto irr2 : range(dpd_irrep[2*i+1]))
            {
                auto irr1 = irr2^dpd_irrep[parent_[2*i]];
                offset += size_[2*i][irr1]*size_[2*i+1][irr2]*dpd_stride[2*i];
            }

            data += offset;
        }

        for (auto i : range(ndim))
        {
            stride[i] = dpd_stride[leaf_[perm_[i]]] *
                        stride_[perm_[i]][irreps[i]];
            len[i] = len_[perm_[i]][irreps[i]];
        }
    }

    void slice(int, const all_t&) {}

    void slice(int idx, const dpd_range& x)
    {
        auto ndim = perm_.size();

        MARRAY_ASSERT(idx < ndim);

        for (auto i : range(nirrep_))
        {
            MARRAY_ASSERT_RANGE_IN(x[i], 0, len_[perm_[idx]][i]);

            off_[leaf_[perm_[idx]]][i] += stride_[perm_[idx]][i] * x[i].front();
            stride_[perm_[idx]][i] *= x[i].step();
            len_[perm_[idx]][i] = x[i].size();
        }
    }

    void slice(int idx, const dpd_index& x)
    {
        auto ndim = perm_.size();

        MARRAY_ASSERT(idx < ndim);
        MARRAY_ASSERT(x.irrep() >= 0 && x.irrep() < nirrep_);
        MARRAY_ASSERT(x.idx() >= 0 && x.idx() < len_[perm_[idx]][x.irrep()]);

        irrep_ ^= x.irrep();
        off_[leaf_[perm_[idx]]][x.irrep()] += x.idx() * stride_[perm_[idx]][x.irrep()];
        fixed_[leaf_[perm_[idx]]] = x.irrep();
    }

    template <typename Slice1, typename Slice2, typename... Slices>
    void slice(int idx, const Slice1& s1, const Slice2& s2, const Slices&... slices)
    {
        slice(idx, s1);
        slice(idx+1, s2, slices...);
    }

    template <typename... Slices>
    std::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value>
    slice(const Slices&... slices)
    {
        slice(0, slices...);

        std::array<bool,sizeof...(Slices)> erase{std::is_same<Slices,dpd_index>::value...};
        dim_vector to_erase;
        for (auto i : reversed_range(sizeof...(Slices)))
            if (erase[i])
                to_erase.push_back(i);

        for (auto idx : to_erase)
        {
            leaf_.erase(leaf_.begin() + perm_[idx]);
            len_.erase(len_.begin() + perm_[idx]);
            stride_.erase(stride_.begin() + perm_[idx]);
        }

        for (auto idx : to_erase)
        {
            auto pidx = perm_[idx];
            perm_.erase(perm_.begin() + idx);
            for (auto& p : perm_)
                if (p > pidx)
                    p--;
        }
    }

    template <size_t N, typename A>
    void slice(const short_vector<dpd_range, N, A>& slices)
    {
        for (auto i : range(slices.size()))
            slice(i, slices[i]);
    }
};

}

template <typename Type, typename Derived, bool Owner>
class dpd_marray_base : protected detail::dpd_base
{
    template <typename, typename, bool> friend class dpd_marray_base;
    template <typename> friend class dpd_marray_view;
    template <typename, typename> friend class dpd_marray;
    template <typename, typename, bool> friend class indexed_dpd_marray_base;

    public:
        typedef Type value_type;
        typedef Type* pointer;
        typedef const Type* const_pointer;
        typedef Type& reference;
        typedef const Type& const_reference;

        typedef typename std::conditional<Owner,const Type,Type>::type ctype;
        typedef ctype& cref;
        typedef ctype* cptr;

    protected:
        typedef detail::dpd_base base;
        pointer data_ = nullptr;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            base::reset();
            data_ = nullptr;
        }

        template <typename U, typename D, bool O>
        void reset(const dpd_marray_base<U, D, O>& other)
        {
            base::reset(other);
            data_ = other.data();
        }

        template <typename U, typename D, bool O>
        void reset(dpd_marray_base<U, D, O>& other)
        {
            base::reset(other);
            data_ = other.data();
        }

        void reset(int irrep, int nirrep,
                   const array_2d<len_type>& len, pointer ptr,
                   dpd_layout layout = DEFAULT_LAYOUT)
        {
            reset(irrep, nirrep, len, ptr,
                  default_depth(layout, len.length(0)), layout.base());
        }

        void reset(int irrep, int nirrep,
                   const array_2d<len_type>& len, pointer ptr,
                   const array_1d<int>& depth, layout layout = DEFAULT_LAYOUT)
        {
            base::reset(irrep, nirrep, len, depth, layout);
            data_ = ptr;
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename View, typename Func>
        void for_each_block_(Func&& f) const
        {
            typedef typename View::pointer Ptr;

            auto ndim = dimension();
            const_pointer cptr;
            irrep_vector irreps(ndim);
            len_vector len(ndim);
            stride_vector stride(ndim);

            irrep_iterator it(irrep(), num_irreps(), ndim);
            while (it.next())
            {
                irreps[0] = irrep();
                for (auto i : range(1,ndim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                bool empty = false;
                for (auto i : range(1,ndim))
                    if (!length(i, irreps[i]))
                        empty = true;
                if (empty) continue;

                cptr = data();
                get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps);
            }
        }

        template <typename View, typename Func, int... I>
        void for_each_block_(Func&& f, std::integer_sequence<int, I...>) const
        {
            constexpr int NDim = sizeof...(I);
            typedef typename View::pointer Ptr;

            MARRAY_ASSERT(NDim == dimension());

            const_pointer cptr;
            std::array<int, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irrep_iterator it(irrep(), num_irreps(), NDim);
            while (it.next())
            {
                irreps[0] = irrep();
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                bool empty = false;
                for (auto i : range(1,NDim))
                    if (!length(i, irreps[i]))
                        empty = true;
                if (empty) continue;

                cptr = data();
                get_block(irreps, len, cptr, stride);

                detail::call(std::forward<Func>(f),
                             View(len, const_cast<Ptr>(cptr), stride),
                             irreps[I]...);
            }
        }

        template <typename Tp, typename Func>
        void for_each_element_(Func&& f) const
        {
            typedef Tp* Ptr;

            auto ndim = dimension();

            const_pointer cptr = data_;
            irrep_vector irreps(ndim);
            len_vector len(ndim);
            stride_vector stride(ndim);

            irrep_iterator it(irrep(), num_irreps(), ndim);
            while (it.next())
            {
                irreps[0] = irrep();
                for (auto i : range(1,ndim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                bool empty = false;
                for (auto i : range(1,ndim))
                    if (!length(i, irreps[i]))
                        empty = true;
                if (empty) continue;

                cptr = data();
                get_block(irreps, len, cptr, stride);

                auto it2 = make_iterator(len, stride);
                Ptr ptr = const_cast<Ptr>(cptr);
                while (it2.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                   irreps, it2.position());
            }
        }

        template <typename Tp, typename Func, int... I>
        void for_each_element_(Func&& f, std::integer_sequence<int, I...>) const
        {
            constexpr int NDim = sizeof...(I);
            typedef Tp* Ptr;

            MARRAY_ASSERT(NDim == dimension());

            const_pointer cptr = data_;
            std::array<int, NDim> irreps;
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            irrep_iterator it(irrep(), num_irreps(), NDim);
            while (it.next())
            {
                irreps[0] = irrep();
                for (auto i : range(1,NDim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                bool empty = false;
                for (auto i : range(1,NDim))
                    if (!length(i, irreps[i]))
                        empty = true;
                if (empty) continue;

                cptr = data();
                get_block(irreps, len, cptr, stride);

                auto it2 = make_iterator(len, stride);
                Ptr ptr = const_cast<Ptr>(cptr);
                while (it2.next(ptr)) detail::call(std::forward<Func>(f), *ptr,
                                                   irreps[I]..., it2.position()[I]...);
            }
        }

        void swap(dpd_marray_base& other)
        {
            using std::swap;
            base::swap(other);
            swap(data_, other.data_);
        }

    public:
        /***********************************************************************
         *
         * Static helper functions
         *
         **********************************************************************/

        static stride_type size(int irrep, const array_2d<len_type>& len)
        {
            if (len.length(0) == 0) return 1;

            //TODO: add alignment option

            matrix<len_type> len_;
            len.slurp(len_);

            auto nirrep = len_.length(1);
            auto ndim = len_.length(0);

            MARRAY_ASSERT(nirrep == 1 || nirrep == 2 ||
                          nirrep == 4 || nirrep == 8);

            std::array<stride_type, 8> size;

            for (auto irr : range(nirrep))
                size[irr] = len_[0][irr];

            for (auto i : range(1,ndim))
            {
                std::array<stride_type, 8> new_size = {};

                for (auto irr1 : range(nirrep))
                for (auto irr2 : range(nirrep))
                    new_size[irr1] += size[irr1^irr2]*len_[i][irr2];

                size = new_size;
            }

            return size[irrep];
        }

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const dpd_marray_base& other)
        {
            return operator=<>(other);
        }

        template <typename U, typename D, bool O>
        Derived& operator=(const dpd_marray_base<U, D, O>& other)
        {
            auto ndim = dimension();

            MARRAY_ASSERT(ndim == other.dimension());
            MARRAY_ASSERT(nirrep_ == other.nirrep_);
            MARRAY_ASSERT(irrep_ == other.irrep_);

            for (auto i : range(ndim))
                MARRAY_ASSERT(lengths(i) == other.lengths(i));

            irrep_iterator it(irrep(), num_irreps(), ndim);
            irrep_vector irreps(ndim);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,ndim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                (*this)(irreps) = other(irreps);
            }

            return static_cast<Derived&>(*this);
        }

        Derived& operator=(const Type& value)
        {
            auto ndim = dimension();

            irrep_iterator it(irrep(), num_irreps(), ndim);
            irrep_vector irreps(ndim);
            while (it.next())
            {
                irreps[0] = irrep_;
                for (auto i : range(1,ndim))
                    irreps[0] ^= irreps[i] = it.irrep(i);

                (*this)(irreps) = value;
            }

            return static_cast<Derived&>(*this);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        dpd_marray_view<const Type> cview() const
        {
            return const_cast<dpd_marray_base&>(*this).view();
        }

        dpd_marray_view<ctype> view() const
        {
            return const_cast<dpd_marray_base&>(*this).view();
        }

        dpd_marray_view<Type> view()
        {
            return *this;
        }

        friend dpd_marray_view<const Type> cview(const dpd_marray_base& x)
        {
            return x.view();
        }

        friend dpd_marray_view<ctype> view(const dpd_marray_base& x)
        {
            return x.view();
        }

        friend dpd_marray_view<Type> view(dpd_marray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        dpd_marray_view<ctype> permuted(const array_1d<int>& perm) const
        {
            return const_cast<dpd_marray_base&>(*this).permuted(perm);
        }

        dpd_marray_view<Type> permuted(const array_1d<int>& perm)
        {
            dpd_marray_view<Type> r(*this);
            r.permute(perm);
            return r;
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        template <typename... Irreps>
        std::enable_if_t<detail::are_assignable<int&, Irreps...>::value,
                            marray_view<ctype, sizeof...(Irreps)>>
        operator()(const Irreps&... irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps...);
        }

        template <typename... Irreps>
        std::enable_if_t<detail::are_assignable<int&, Irreps...>::value,
                            marray_view<Type, sizeof...(Irreps)>>
        operator()(const Irreps&... irreps_)
        {
            constexpr int NDim = sizeof...(Irreps);

            MARRAY_ASSERT(NDim == dimension());

            std::array<int, NDim> irreps{(int)irreps_...};
            std::array<len_type, NDim> len;
            std::array<stride_type, NDim> stride;

            pointer ptr = data();
            get_block(irreps, len, ptr, stride);

            return marray_view<Type, NDim>(len, ptr, stride);
        }

        marray_view<ctype> operator()(const array_1d<int>& irreps) const
        {
            return const_cast<dpd_marray_base&>(*this)(irreps);
        }

        marray_view<Type> operator()(const array_1d<int>& irreps_)
        {
            auto ndim = dimension();

            MARRAY_ASSERT(irreps_.size() == ndim);

            irrep_vector irreps;
            irreps_.slurp(irreps);

            len_vector len(ndim);
            stride_vector stride(ndim);

            auto irrep = 0;
            for (auto& i: irreps) irrep ^= i;
            MARRAY_ASSERT(irrep == irrep_);

            pointer ptr = data();
            get_block(irreps, len, ptr, stride);

            return marray_view<Type>(len, ptr, stride);
        }

        /***********************************************************************
         *
         * Slicing
         *
         **********************************************************************/

        template <typename... Slices>
        std::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            detail::sliced_dimension<Slices...>::value,
                            dpd_marray_view<ctype>>
        operator()(const Slices&... slices) const
        {
            return const_cast<dpd_marray_base&>(*this)(slices...);
        }

        template <typename... Slices>
        std::enable_if_t<detail::are_dpd_indices_or_slices<Slices...>::value &&
                            detail::sliced_dimension<Slices...>::value,
                            dpd_marray_view<Type>>
        operator()(const Slices&... slices)
        {
            dpd_marray_view<Type> ret(*this);
            ret.slice(slices...);
            return ret;
        }

        dpd_marray_view<ctype> operator()(const array_1d<dpd_range>& slices) const
        {
            return const_cast<dpd_marray_base&>(*this)(slices);
        }

        dpd_marray_view<Type> operator()(const array_1d<dpd_range>& slices_)
        {
            MARRAY_ASSERT(slices_.size() == dimension());

            short_vector<dpd_range, MARRAY_OPT_NDIM> slices;
            slices_.slurp(slices);

            dpd_marray_view<Type> ret(*this);
            ret.slice(slices);
            return ret;
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block_<marray_view<ctype>>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block_<marray_view<Type>>(std::forward<Func>(f));
        }

        template <int NDim, typename Func>
        void for_each_block(Func&& f) const
        {
            for_each_block_<marray_view<ctype, NDim>>(std::forward<Func>(f), std::make_integer_sequence<int, NDim>{});
        }

        template <int NDim, typename Func>
        void for_each_block(Func&& f)
        {
            for_each_block_<marray_view<Type, NDim>>(std::forward<Func>(f), std::make_integer_sequence<int, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element_<ctype>(std::forward<Func>(f));
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element_<Type>(std::forward<Func>(f));
        }

        template <int NDim, typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element_<ctype>(std::forward<Func>(f), std::make_integer_sequence<int, NDim>{});
        }

        template <int NDim, typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element_<Type>(std::forward<Func>(f), std::make_integer_sequence<int, NDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        const_pointer cdata() const
        {
            return const_cast<dpd_marray_base&>(*this).data();
        }

        cptr data() const
        {
            return const_cast<dpd_marray_base&>(*this).data();
        }

        pointer data()
        {
            return data_;
        }

        len_type length(int dim, int irrep) const
        {
            MARRAY_ASSERT(irrep >= 0 && irrep < num_irreps());
            return len_[permutation(dim)][irrep];
        }

        row_view<const len_type> lengths(int dim) const
        {
            return row_view<const len_type>{{num_irreps()}, &len_[permutation(dim)][0]};
        }

        matrix<len_type> lengths() const
        {
            matrix<len_type> len({dimension(), num_irreps()}, ROW_MAJOR);
            for (auto i : range(dimension()))
                for (auto j : range(num_irreps()))
                    len[i][j] = length(i, j);
            return len;
        }

        auto irrep() const
        {
            return irrep_;
        }

        auto num_irreps() const
        {
            return nirrep_;
        }

        auto& permutation() const
        {
            return perm_;
        }

        auto permutation(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            return perm_[dim];
        }

        int dimension() const
        {
            return perm_.size();
        }
};

}

#endif //MARRAY_DPD_marray_BASE_HPP
