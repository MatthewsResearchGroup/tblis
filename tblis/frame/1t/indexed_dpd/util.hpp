#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_UTIL_HPP_

#include <climits>

#include "tblis/frame/base/thread.h"
#include "tblis/frame/base/basic_types.h"

#include "marray/indexed_dpd/indexed_dpd_marray_view.hpp"

#include "tblis/frame/1t/dpd/util.hpp"
#include "tblis/frame/1t/indexed/util.hpp"

namespace tblis
{

using MArray::indexed_dpd_marray_view;

namespace internal
{

inline
auto block_to_full(type_t type, const communicator& comm, const cntx_t* cntx,
                   const indexed_dpd_marray_view<char>& A)
{
    auto ts = type_size[type];
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();
    auto dense_ndim_A = A.dense_dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    auto stride_A2 = MArray::detail::strides(len_A);
    auto size_A = stl_ext::prod(len_A);

    scalar factor_A(0.0, type);
    scalar zero(0.0, type);

    char* A2 = comm.master() ? new char[size_A*ts]() : nullptr;
    comm.broadcast_value(A2);

    auto dense_stride_A2 = stride_A2;
    dense_stride_A2.resize(dense_ndim_A);

    A[0].for_each_block(
    [&](auto&& local_A, auto&& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (auto i : range(A.num_indices()))
        {
            auto data_A = A.data(0) + (local_A.data()-A.data(0))*ts + (A.data(i) - A.data(0));
            auto idx_A = A.indices(i);

            factor_A.from(A.factors().data() + i*ts);

            auto data_A2 = A2;
            for (auto i : range(dense_ndim_A))
                data_A2 += off_A[i][irreps_A[i]]*stride_A2[i]*ts;
            for (auto i : range(dense_ndim_A,ndim_A))
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*stride_A2[i]*ts;

            add(type, comm, cntx, {}, {}, dense_len_A,
                factor_A, false,  data_A, {},  dense_stride_A,
                    zero, false, data_A2, {}, dense_stride_A2);
        }
    });

    return std::make_tuple(A2, len_A, stride_A2);
}

inline
void full_to_block(type_t type, const communicator& comm, const cntx_t* cntx,
                   char* A2, const len_vector&, const stride_vector& stride_A2,
                   const indexed_dpd_marray_view<char>& A)
{
    auto ts = type_size[type];
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();
    auto dense_ndim_A = A.dense_dimension();

    matrix<len_type> off_A{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        len_type off = 0;
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    auto dense_stride_A2 = stride_A2;
    dense_stride_A2.resize(dense_ndim_A);

    scalar factor_A(0.0, type);
    scalar one(1.0, type);

    A[0].for_each_block(
    [&](auto&& local_A, auto&& irreps_A)
    {
        auto& dense_len_A = local_A.lengths();
        auto& dense_stride_A = local_A.strides();

        for (auto i : range(A.num_indices()))
        {
            auto data_A = A.data(0) + (local_A.data()-A.data(0))*ts + (A.data(i) - A.data(0));
            auto idx_A = A.indices(i);

            factor_A.from(A.factors().data() + i*ts);

            auto data_A2 = A2;
            for (auto i : range(dense_ndim_A))
                data_A2 += off_A[i][irreps_A[i]]*stride_A2[i]*ts;
            for (auto i : range(dense_ndim_A,ndim_A))
                data_A2 += (idx_A[i-dense_ndim_A] +
                    off_A[i][A.indexed_irrep(i-dense_ndim_A)])*stride_A2[i]*ts;

            add(type, comm, cntx, {}, {}, dense_len_A,
                factor_A, false, data_A2, {}, dense_stride_A2,
                     one, false,  data_A, {},  dense_stride_A);
        }
    });
}

template <int N> struct dpd_index_group;

template <int I, int N>
void assign_dense_idx_helper(int, dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void assign_dense_idx_helper(int i, dpd_index_group<N>& group,
                             const indexed_dpd_marray_view<T>&,
                             const dim_vector& idx_A, const Args&... args)
{
    group.dense_idx[I].push_back(idx_A[i]);
    assign_dense_idx_helper<I+1>(i, group, args...);
}

template <int N, typename T, typename... Args>
void assign_dense_idx(int i, dpd_index_group<N>& group,
                      const indexed_dpd_marray_view<T>& A,
                      const dim_vector& idx_A, const Args&... args)
{
    assign_dense_idx_helper<0>(i, group, A, idx_A, args...);
}

template <int I, int N>
void assign_mixed_or_batch_idx_helper(int, int,
                                      dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void assign_mixed_or_batch_idx_helper(int i, int pos,
                                      dpd_index_group<N>& group,
                                      const indexed_dpd_marray_view<T>& A,
                                      const dim_vector& idx_A, const Args&... args)
{

    if (idx_A[i] < A.dense_dimension())
    {
        group.mixed_idx[I].push_back(idx_A[i]);
        group.mixed_pos[I].push_back(pos);
    }
    else
    {
        auto idx = idx_A[i] - A.dense_dimension();

        group.batch_idx[I].push_back(idx);
        group.batch_pos[I].push_back(pos);

        TBLIS_ASSERT(group.batch_irrep[pos] == -1 ||
                     group.batch_irrep[pos] == A.indexed_irrep(idx));
        TBLIS_ASSERT(group.batch_len[pos] == -1 ||
                     group.batch_len[pos] == A.indexed_length(idx));
        group.batch_irrep[pos] = A.indexed_irrep(idx);
        group.batch_len[pos] = A.indexed_length(idx);
    }

    assign_mixed_or_batch_idx_helper<I+1>(i, pos, group, args...);
}

template <int N, typename T, typename... Args>
void assign_mixed_or_batch_idx(int i, int pos,
                               dpd_index_group<N>& group,
                               const indexed_dpd_marray_view<T>& A,
                               const dim_vector& idx_A, const Args&... args)
{
    assign_mixed_or_batch_idx_helper<0>(i, pos, group,
                                        A, idx_A, args...);
}

template <int N>
struct dpd_index_group
{
    int dense_ndim = 0;
    int batch_ndim = 0;
    int dense_nblock = 1;
    stride_type dense_size = 0;

    std::array<dim_vector,N> dense_idx;
    std::array<int,N> unit_dim;

    std::array<dim_vector,N> mixed_idx;
    std::array<dim_vector,N> mixed_pos;

    len_vector batch_len;
    stride_vector batch_stride;
    irrep_vector batch_irrep;
    std::array<dim_vector,N> batch_idx;
    std::array<dim_vector,N> batch_pos;

    template <int... I>
    dim_vector sort_by_stride(const std::array<stride_vector,N>& dense_stride,
                              std::integer_sequence<int, I...>)
    {
        return internal::sort_by_stride(dense_stride[I]...);
    }

    template <typename T, typename... Args>
    dpd_index_group(const indexed_dpd_marray_view<T>& A, const dim_vector& idx_A,
                    const Args&... args)
    {
        auto nirrep = A.num_irreps();

        batch_len.resize(idx_A.size(), -1);
        batch_irrep.resize(idx_A.size(), -1);

        for (auto i : range(idx_A.size()))
        {
            if (is_idx_dense(i, A, idx_A, args...))
            {
                assign_dense_idx(i, *this, A, idx_A, args...);
                dense_ndim++;
            }
            else
            {
                assign_mixed_or_batch_idx(i, batch_ndim,
                                          *this, A, idx_A, args...);
                batch_ndim++;
            }
        }

        batch_len.resize(batch_ndim);
        batch_stride.resize(batch_ndim);
        batch_irrep.resize(batch_ndim);

        if (batch_ndim > 0) batch_stride[0] = 1;
        for (auto i : range(1,batch_ndim))
            batch_stride[i] = batch_stride[i-1]*batch_len[i-1];

        std::array<len_vector,N> dense_len;
        std::array<stride_vector,N> dense_stride;
        dense_total_lengths_and_strides(dense_len, dense_stride,
                                        A, idx_A, args...);

        dense_size = 1;
        for (auto i : range(dense_ndim))
        {
            dense_size *= dense_len[0][i];
            dense_nblock *= nirrep;
        }

        if (dense_nblock > 1)
        {
            dense_size = std::max<stride_type>(1, dense_size/nirrep);
            dense_nblock /= nirrep;
        }

        std::array<stride_vector,N> dense_stride_sub;
        for (auto i : range(N))
            dense_stride_sub[i] = stl_ext::select_from(dense_stride[i],
                                                       dense_idx[i]);

        auto reorder = sort_by_stride(dense_stride_sub,
                                      std::make_integer_sequence<int, N>{});

        for (auto i : range(N))
            stl_ext::permute(dense_idx[i], reorder);

        for (auto i : range(N))
        {
            unit_dim[i] = dense_ndim;
            for (auto j : range(dense_ndim))
            if (dense_stride[i][reorder[j]] == 1)
            {
                unit_dim[i] = j;
                break;
            }
        }
    }
};

template <int I, int N>
void assign_irreps_helper(const dpd_index_group<N>&) {}

template <int I, int N, typename... Args>
void assign_irreps_helper(const dpd_index_group<N>& group,
                          irrep_vector& irreps, Args&... args)
{
    for (auto j : range(group.mixed_idx[I].size()))
    {
        irreps[group.mixed_idx[I][j]] = group.batch_irrep[group.mixed_pos[I][j]];
    }

    assign_irreps_helper<I+1>(group, args...);
}

template <int N, typename... Args>
void assign_irreps(const dpd_index_group<N>& group, Args&... args)
{
    assign_irreps_helper<0>(group, args...);
}

template <int I, int N>
void get_local_geometry_helper(const len_vector&,
                               const dpd_index_group<N>&,
                               len_vector&) {}

template <int I, int N, typename T, typename... Args>
void get_local_geometry_helper(const len_vector& idx,
                               const dpd_index_group<N>& group,
                               len_vector& len,  const marray_view<T>& local_A,
                               stride_vector& stride,
                               int, Args&&... args)
{
    if (I == 0)
        len = stl_ext::select_from(local_A.lengths(), group.dense_idx[I]);

    stride = stl_ext::select_from(local_A.strides(), group.dense_idx[I]);

    get_local_geometry_helper<I+1>(idx, group, len, std::forward<Args>(args)...);
}

template <int N, typename... Args>
void get_local_geometry(const len_vector& idx, const dpd_index_group<N>& group,
                        len_vector& len, Args&&... args)
{
    get_local_geometry_helper<0>(idx, group, len, std::forward<Args>(args)...);
}

template <int I, int N>
void get_local_offset_helper(const len_vector&,
                             const dpd_index_group<N>&) {}

template <int I, int N, typename T, typename... Args>
void get_local_offset_helper(const len_vector& idx,
                             const dpd_index_group<N>& group,
                             const T& A, stride_type& off,
                             int i, Args&&... args)
{
    off = 0;
    for (auto j : range(group.mixed_idx[i].size()))
        off += idx[group.mixed_pos[i][j]]*
            A.stride(group.mixed_idx[i][j]);

    get_local_offset_helper<I+1>(idx, group, std::forward<Args>(args)...);
}

template <int N, typename... Args>
void get_local_offset(const len_vector& idx, const dpd_index_group<N>& group,
                      Args&&... args)
{
    get_local_offset_helper<0>(idx, group, std::forward<Args>(args)...);
}

}
}

#endif
