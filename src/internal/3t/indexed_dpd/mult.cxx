#include "mult.hpp"

#include "internal/1t/dense/add.hpp"
#include "internal/1t/indexed_dpd/util.hpp"
#include "internal/1t/indexed_dpd/set.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/3t/dense/mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "matrix/tensor_matrix.hpp"
#include "matrix/dpd_tensor_matrix.hpp"
#include "matrix/scatter_matrix.hpp"
#include "matrix/scatter_tensor_matrix.hpp"

#include "nodes/gemm.hpp"

#include <memory>

namespace MArray
{

template <typename T, size_t N, typename Allocator>
std::ostream& operator<<(std::ostream& os, const short_vector<T, N, Allocator>& v)
{
    return os << std::vector<T>(v.begin(), v.end());
}

}

namespace tblis
{
namespace internal
{

/*
template <typename T, int N>
std::ostream& operator<<(std::ostream& os, const index_set<N>& v)
{
    os << "\n{\n";
    os << "\toffset, factor: " << v.offset << " " << v.factor << '\n';
    for (int i = 0;i < N;i++)
        os << "\tkey, idx: " << v.key[i] << " " << v.idx[i] << '\n';
    return os << '}';
}

template <typename T>
std::enable_if_t<!is_complex<T>::value,T>
make_complex_if(real_type_t<T> r, real_type_t<T> i)
{
    return r;
}

template <typename T>
std::enable_if_t<is_complex<T>::value,T>
make_complex_if(real_type_t<T> r, real_type_t<T> i)
{
    return {r, i};
}
*/

enum { OPT_NFUSE = 256 };

typedef short_vector<stride_type,OPT_NFUSE> stride_vector_fuse;
typedef short_vector<std::tuple<double,double,stride_type,stride_type>,OPT_NFUSE> tuple_vector_fuse;

static thread_local std::aligned_storage_t<sizeof(stride_vector_fuse)> scat_A_AB_;
static thread_local std::aligned_storage_t<sizeof(stride_vector_fuse)> scat_B_AB_;
static thread_local std::aligned_storage_t<sizeof(stride_vector_fuse)> scat_B_BC_;
static thread_local std::aligned_storage_t<sizeof(stride_vector_fuse)> scat_C_BC_;
static thread_local std::aligned_storage_t<sizeof(tuple_vector_fuse)> scat_AB_;
static thread_local std::aligned_storage_t<sizeof(tuple_vector_fuse)> scat_BC_;

double relative_perf(double m, double n, double k)
{
    // e.g. 200GFLOPs (double) and 80GB/s memory BW
    constexpr double flops_per_element = 20;
    return 1/(1 + flops_per_element*(0.5/m + 0.5/n + 0.5/k));
}

template <typename T>
void mult_full(const communicator& comm, const config& cfg,
               T alpha, bool conj_A, const indexed_dpd_marray_view<T>& A,
               const dim_vector& idx_A_AB,
               const dim_vector& idx_A_AC,
               const dim_vector& idx_A_ABC,
                        bool conj_B, const indexed_dpd_marray_view<T>& B,
               const dim_vector& idx_B_AB,
               const dim_vector& idx_B_BC,
               const dim_vector& idx_B_ABC,
                                     const indexed_dpd_marray_view<T>& C,
               const dim_vector& idx_C_AC,
               const dim_vector& idx_C_BC,
               const dim_vector& idx_C_ABC)
{
    marray<T> A2, B2, C2;

    comm.broadcast(
    [&](marray<T>& A2, marray<T>& B2, marray<T>& C2)
    {
        block_to_full(comm, cfg, A, A2);
        block_to_full(comm, cfg, B, B2);
        block_to_full(comm, cfg, C, C2);

        auto len_AB = stl_ext::select_from(A2.lengths(), idx_A_AB);
        auto len_AC = stl_ext::select_from(C2.lengths(), idx_C_AC);
        auto len_BC = stl_ext::select_from(C2.lengths(), idx_C_BC);
        auto len_ABC = stl_ext::select_from(C2.lengths(), idx_C_ABC);
        auto stride_A_AB = stl_ext::select_from(A2.strides(), idx_A_AB);
        auto stride_A_AC = stl_ext::select_from(A2.strides(), idx_A_AC);
        auto stride_B_AB = stl_ext::select_from(B2.strides(), idx_B_AB);
        auto stride_B_BC = stl_ext::select_from(B2.strides(), idx_B_BC);
        auto stride_C_AC = stl_ext::select_from(C2.strides(), idx_C_AC);
        auto stride_C_BC = stl_ext::select_from(C2.strides(), idx_C_BC);
        auto stride_A_ABC = stl_ext::select_from(A2.strides(), idx_A_ABC);
        auto stride_B_ABC = stl_ext::select_from(B2.strides(), idx_B_ABC);
        auto stride_C_ABC = stl_ext::select_from(C2.strides(), idx_C_ABC);

        mult(type_tag<T>::value, comm, cfg, len_AB, len_AC, len_BC, len_ABC,
             alpha, conj_A, reinterpret_cast<char*>(A2.data()), stride_A_AB, stride_A_AC, stride_A_ABC,
                    conj_B, reinterpret_cast<char*>(B2.data()), stride_B_AB, stride_B_BC, stride_B_ABC,
              T(0),  false, reinterpret_cast<char*>(C2.data()), stride_C_AC, stride_C_BC, stride_C_ABC);

        full_to_block(comm, cfg, C2, C);
    },
    A2, B2, C2);
}

void mult_block_fuse_AB(type_t type, const communicator& comm, const config& cfg,
                        const scalar& alpha,
                        bool conj_A, const indexed_dpd_marray_view<char>& A,
                        dim_vector idx_A_AB,
                        dim_vector idx_A_AC,
                        bool conj_B, const indexed_dpd_marray_view<char>& B,
                        dim_vector idx_B_AB,
                        dim_vector idx_B_BC,
                                     const indexed_dpd_marray_view<char>& C,
                        dim_vector idx_C_AC,
                        dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<2> indices_A(type, A, group_AC, 0, group_AB, 0);
    group_indices<2> indices_B(type, B, group_BC, 0, group_AB, 1);
    group_indices<2> indices_C(type, C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    comm.do_tasks_deferred(nirrep*nidx_C*group_AC.dense_nblock*group_BC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size*group_AB.dense_nblock/inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        stride_type idx = 0;
        stride_type idx_A = 0;
        stride_type idx_C = 0;

        for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                   idx_C, nidx_C, indices_C, 0,
        [&](stride_type next_A, stride_type next_C)
        {
            stride_type idx_B = 0;

            for_each_match<true, false>(idx_B, nidx_B, indices_B, 0,
                                        idx_C, next_C, indices_C, 1,
            [&](stride_type next_B)
            {
                if (indices_C[idx_C].factor.is_zero()) return;

                for (auto irrep_AB0 : range(nirrep))
                {
                    auto irrep_AB = irrep_AB0;
                    auto irrep_AC = B.irrep()^C.irrep()^irrep_AB0;
                    auto irrep_BC = A.irrep()^C.irrep()^irrep_AB0;

                    for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
                    for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
                    for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                    if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
                    if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                    if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                    for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                    for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                    {
                        tasks.visit(idx++,
                        [&,idx_A,idx_B,idx_C,next_A,next_B,
                         irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC]
                        (const communicator& subcomm)
                        {
                            auto& scat_A_AB = *(new (&scat_A_AB_) stride_vector_fuse);
                            auto& scat_B_AB = *(new (&scat_B_AB_) stride_vector_fuse);
                            auto& scat_AB = *(new (&scat_AB_) tuple_vector_fuse);

                            auto local_irreps_A = irreps_A;
                            auto local_irreps_B = irreps_B;
                            auto local_irreps_C = irreps_C;

                            assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                          local_irreps_A, group_AC.dense_idx[0],
                                          local_irreps_C, group_AC.dense_idx[1]);

                            assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                          local_irreps_B, group_BC.dense_idx[0],
                                          local_irreps_C, group_BC.dense_idx[1]);

                            if (is_block_empty(dpd_C, local_irreps_C)) return;

                            marray_view<char> local_C = dpd_C(local_irreps_C);

                            for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                            {
                                assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                              local_irreps_A, group_AB.dense_idx[0],
                                              local_irreps_B, group_AB.dense_idx[1]);

                                if (is_block_empty(dpd_A, local_irreps_A)) continue;

                                marray_view<char> local_A = dpd_A(local_irreps_A);
                                marray_view<char> local_B = dpd_B(local_irreps_B);

                                len_vector len_AC;
                                stride_vector stride_A_AC, stride_C_AC;
                                stride_type off_A_AC, off_C_AC;
                                get_local_geometry(indices_A[idx_A].idx[0], group_AC, len_AC,
                                                   local_A, stride_A_AC, 0,
                                                   local_C, stride_C_AC, 1);
                                get_local_offset(indices_A[idx_A].idx[0], group_AC,
                                                 local_A, off_A_AC, 0,
                                                 local_C, off_C_AC, 1);

                                len_vector len_BC;
                                stride_vector stride_B_BC, stride_C_BC;
                                stride_type off_B_BC, off_C_BC;
                                get_local_geometry(indices_B[idx_B].idx[0], group_BC, len_BC,
                                                   local_B, stride_B_BC, 0,
                                                   local_C, stride_C_BC, 1);
                                get_local_offset(indices_B[idx_B].idx[0], group_BC,
                                                 local_B, off_B_BC, 0,
                                                 local_C, off_C_BC, 1);

                                len_vector len_AB;
                                stride_vector stride_A_AB, stride_B_AB;
                                get_local_geometry(indices_A[idx_A].idx[1], group_AB, len_AB,
                                                   local_A, stride_A_AB, 0,
                                                   local_B, stride_B_AB, 1);

                                auto local_idx_A = idx_A;
                                auto local_idx_B = idx_B;

                                scat_AB.clear();

                                for_each_match<false, false>(local_idx_A, next_A, indices_A, 1,
                                                             local_idx_B, next_B, indices_B, 1,
                                [&]
                                {
                                    auto factor = alpha*indices_A[local_idx_A].factor*
                                                        indices_B[local_idx_B].factor*
                                                        indices_C[idx_C].factor;
                                    if (factor.is_zero()) return;

                                    stride_type off_A_AB, off_B_AB;
                                    get_local_offset(indices_A[local_idx_A].idx[1], group_AB,
                                                     local_A, off_A_AB, 0,
                                                     local_B, off_B_AB, 1);

                                    switch (type)
                                    {
                                        case TYPE_FLOAT:
                                            scat_AB.emplace_back(factor.template get<float>(), 0.0,
                                                                 indices_A[local_idx_A].offset + off_A_AB,
                                                                 indices_B[local_idx_B].offset + off_B_AB);
                                            break;
                                        case TYPE_DOUBLE:
                                            scat_AB.emplace_back(factor.template get<double>(), 0.0,
                                                                 indices_A[local_idx_A].offset + off_A_AB,
                                                                 indices_B[local_idx_B].offset + off_B_AB);
                                            break;
                                        case TYPE_SCOMPLEX:
                                            scat_AB.emplace_back(factor.template get<scomplex>().real(),
                                                                 factor.template get<scomplex>().imag(),
                                                                 indices_A[local_idx_A].offset + off_A_AB,
                                                                 indices_B[local_idx_B].offset + off_B_AB);
                                            break;
                                        case TYPE_DCOMPLEX:
                                            scat_AB.emplace_back(factor.template get<dcomplex>().real(),
                                                                 factor.template get<dcomplex>().imag(),
                                                                 indices_A[local_idx_A].offset + off_A_AB,
                                                                 indices_B[local_idx_B].offset + off_B_AB);
                                            break;
                                    }
                                });

                                if (scat_AB.empty()) continue;

                                std::sort(scat_AB.begin(), scat_AB.end());

                                auto data_A = A.data(0) + (local_A.data() - A.data(0) + off_A_AC)*ts;
                                auto data_B = B.data(0) + (local_B.data() - B.data(0) + off_B_BC)*ts;
                                auto data_C = C.data(0) + (local_C.data() - C.data(0) +
                                    indices_C[idx_C].offset + off_C_AC + off_C_BC)*ts;

                                for (auto i : range(scat_AB.size()))
                                {
                                    scat_A_AB.push_back(std::get<2>(scat_AB[i]));
                                    scat_B_AB.push_back(std::get<3>(scat_AB[i]));

                                    if (i == scat_AB.size()-1 ||
                                        std::get<0>(scat_AB[i]) != std::get<0>(scat_AB[i+1]) ||
                                        std::get<1>(scat_AB[i]) != std::get<1>(scat_AB[i+1]))
                                    {
                                        scalar alpha(dcomplex(std::get<0>(scat_AB[i]),
                                                              std::get<1>(scat_AB[i])), type);

                                        scatter_tensor_matrix at(alpha, conj_A, len_AC, len_AB, data_A, stride_A_AC, stride_A_AB,
                                                                    {}, row_view<stride_type>{{scat_A_AB.size()}, scat_A_AB.data()});
                                        scatter_tensor_matrix bt(  one, conj_B, len_AB, len_BC, data_B, stride_B_AB, stride_B_BC,
                                                                    row_view<stride_type>{{scat_B_AB.size()}, scat_B_AB.data()}, {});
                                                tensor_matrix ct(  one,  false, len_AC, len_BC, data_C, stride_C_AC, stride_C_BC);

                                        GotoGEMM{}(subcomm, cfg, at, bt, ct);

                                        scat_A_AB.clear();
                                        scat_B_AB.clear();
                                    }
                                }
                            }

                            scat_A_AB.~short_vector();
                            scat_B_AB.~short_vector();
                            scat_AB.~short_vector();
                        });
                    }
                }
            });
        });
    });
}

//if(comm.thread_num() == 0) comm.thread_master
//print the information is given
void mult_block_fuse_BC(type_t type, const communicator& comm, const config& cfg,
                        const scalar& alpha,
                        bool conj_A, const indexed_dpd_marray_view<char>& A,
                        dim_vector idx_A_AB,
                        dim_vector idx_A_AC,
                        bool conj_B, const indexed_dpd_marray_view<char>& B,
                        dim_vector idx_B_AB,
                        dim_vector idx_B_BC,
                                     const indexed_dpd_marray_view<char>& C,
                        dim_vector idx_C_AC,
                        dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<2> indices_A(type, A, group_AC, 0, group_AB, 0);
    group_indices<2> indices_B(type, B, group_AB, 1, group_BC, 0);
    group_indices<2> indices_C(type, C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    //print task list
    comm.do_tasks_deferred(nirrep*nidx_C*group_AC.dense_nblock*group_BC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size*group_AB.dense_nblock/inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        stride_type idx = 0;
        stride_type idx_A = 0;
        stride_type idx_C = 0;

        for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                   idx_C, nidx_C, indices_C, 0,
        [&](stride_type next_A, stride_type next_C)
        {
            for (auto irrep_AB0 : range(nirrep))
            {
                auto irrep_AB = irrep_AB0;
                auto irrep_AC = B.irrep()^C.irrep()^irrep_AB0;
                auto irrep_BC = A.irrep()^C.irrep()^irrep_AB0;

                for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
                for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
                for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
                if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                {
                    //print task lists
                    tasks.visit(idx++,
                    [&,idx_A,idx_C,next_A,next_C,
                     irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC]
                    (const communicator& subcomm)
                    {
                        auto& scat_B_BC = *(new (&scat_B_BC_) stride_vector_fuse);
                        auto& scat_C_BC = *(new (&scat_C_BC_) stride_vector_fuse);
                        auto& scat_BC = *(new (&scat_BC_) tuple_vector_fuse);

                        auto local_irreps_A = irreps_A;
                        auto local_irreps_B = irreps_B;
                        auto local_irreps_C = irreps_C;

                        assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                      local_irreps_A, group_AC.dense_idx[0],
                                      local_irreps_C, group_AC.dense_idx[1]);

                        assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                      local_irreps_B, group_BC.dense_idx[0],
                                      local_irreps_C, group_BC.dense_idx[1]);

                        if (is_block_empty(dpd_C, local_irreps_C)) return;

                        marray_view<char> local_C = dpd_C(local_irreps_C);

                        for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                        {
                            assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                          local_irreps_A, group_AB.dense_idx[0],
                                          local_irreps_B, group_AB.dense_idx[1]);

                            if (is_block_empty(dpd_A, local_irreps_A)) continue;

                            marray_view<char> local_A = dpd_A(local_irreps_A);
                            marray_view<char> local_B = dpd_B(local_irreps_B);

                            len_type local_idx_A = idx_A;
                            len_type local_idx_B = 0;

                            len_vector len_AC;
                            stride_vector stride_A_AC, stride_C_AC;
                            stride_type off_A_AC, off_C_AC;
                            get_local_geometry(indices_A[local_idx_A].idx[0], group_AC, len_AC,
                                               local_A, stride_A_AC, 0,
                                               local_C, stride_C_AC, 1);
                            get_local_offset(indices_A[local_idx_A].idx[0], group_AC,
                                             local_A, off_A_AC, 0,
                                             local_C, off_C_AC, 1);

                            len_vector len_BC;
                            stride_vector stride_B_BC, stride_C_BC;
                            get_local_geometry(indices_B[local_idx_B].idx[1], group_BC, len_BC,
                                               local_B, stride_B_BC, 0,
                                               local_C, stride_C_BC, 1);

                            len_vector len_AB;
                            stride_vector stride_A_AB, stride_B_AB;
                            get_local_geometry(indices_A[local_idx_A].idx[1], group_AB, len_AB,
                                               local_A, stride_A_AB, 0,
                                               local_B, stride_B_AB, 1);

                            for_each_match<false, true>(local_idx_A, next_A, indices_A, 1,
                                                        local_idx_B, nidx_B, indices_B, 0,
                            [&](stride_type next_B)
                            {
                                if (indices_A[local_idx_A].factor.is_zero()) return;

                                stride_type off_A_AB, off_B_AB;
                                get_local_offset(indices_A[local_idx_A].idx[1], group_AB,
                                                 local_A, off_A_AB, 0,
                                                 local_B, off_B_AB, 1);

                                scat_BC.clear();

                                len_type local_idx_C = idx_C;

                                for_each_match<false, false>(local_idx_B, next_B, indices_B, 1,
                                                             local_idx_C, next_C, indices_C, 1,
                                [&]
                                {
                                    auto factor = alpha*indices_A[local_idx_A].factor*
                                                        indices_B[local_idx_B].factor*
                                                        indices_C[local_idx_C].factor;
                                    if (factor.is_zero()) return;

                                    stride_type off_B_BC, off_C_BC;
                                    get_local_offset(indices_C[local_idx_C].idx[1], group_BC,
                                                     local_B, off_B_BC, 0,
                                                     local_C, off_C_BC, 1);

                                    switch (type)
                                    {
                                        case TYPE_FLOAT:
                                            scat_BC.emplace_back(factor.template get<float>(), 0.0,
                                                                 indices_B[local_idx_B].offset + off_B_BC,
                                                                 indices_C[local_idx_C].offset + off_C_BC);
                                            break;
                                        case TYPE_DOUBLE:
                                            scat_BC.emplace_back(factor.template get<double>(), 0.0,
                                                                 indices_B[local_idx_B].offset + off_B_BC,
                                                                 indices_C[local_idx_C].offset + off_C_BC);
                                            break;
                                        case TYPE_SCOMPLEX:
                                            scat_BC.emplace_back(factor.template get<scomplex>().real(),
                                                                 factor.template get<scomplex>().imag(),
                                                                 indices_B[local_idx_B].offset + off_B_BC,
                                                                 indices_C[local_idx_C].offset + off_C_BC);
                                            break;
                                        case TYPE_DCOMPLEX:
                                            scat_BC.emplace_back(factor.template get<dcomplex>().real(),
                                                                 factor.template get<dcomplex>().imag(),
                                                                 indices_B[local_idx_B].offset + off_B_BC,
                                                                 indices_C[local_idx_C].offset + off_C_BC);
                                            break;
                                    }
                                });

                                if (scat_BC.empty()) return;

                                std::sort(scat_BC.begin(), scat_BC.end());

                                auto data_A = A.data(0) + (local_A.data() - A.data(0) +
                                    indices_A[local_idx_A].offset + off_A_AB + off_A_AC)*ts;
                                auto data_B = B.data(0) + (local_B.data() - B.data(0) + off_B_AB)*ts;
                                auto data_C = C.data(0) + (local_C.data() - C.data(0) + off_C_AC)*ts;

                                for (auto i : range(scat_BC.size()))
                                {
                                    scat_B_BC.push_back(std::get<2>(scat_BC[i]));
                                    scat_C_BC.push_back(std::get<3>(scat_BC[i]));

                                    if (i == scat_BC.size()-1 ||
                                        std::get<0>(scat_BC[i]) != std::get<0>(scat_BC[i+1]) ||
                                        std::get<1>(scat_BC[i]) != std::get<1>(scat_BC[i+1]))
                                    {
                                        scalar alpha(dcomplex(std::get<0>(scat_BC[i]),
                                                              std::get<1>(scat_BC[i])), type);

                                                tensor_matrix at(alpha, conj_A, len_AC, len_AB, data_A, stride_A_AC, stride_A_AB);
                                        scatter_tensor_matrix bt(  one, conj_B, len_AB, len_BC, data_B, stride_B_AB, stride_B_BC,
                                                                 {}, row_view<stride_type>{{scat_B_BC.size()}, scat_B_BC.data()});
                                        scatter_tensor_matrix ct(  one, conj_B, len_AC, len_BC, data_C, stride_C_AC, stride_C_BC,
                                                                 {}, row_view<stride_type>{{scat_C_BC.size()}, scat_C_BC.data()});

                                        GotoGEMM{}(subcomm, cfg, at, bt, ct);

                                        scat_B_BC.clear();
                                        scat_C_BC.clear();
                                    }
                                }
                            });
                        }

                        scat_B_BC.~short_vector();
                        scat_C_BC.~short_vector();
                        scat_BC.~short_vector();
                    });
                }
            }
        });
    });
}

void mult_block_fuse_AB_BC(type_t type, const communicator& comm, const config& cfg,
                           const scalar& alpha,
                           bool conj_A, const indexed_dpd_marray_view<char>& A,
                           dim_vector idx_A_AB,
                           dim_vector idx_A_AC,
                           bool conj_B, const indexed_dpd_marray_view<char>& B,
                           dim_vector idx_B_AB,
                           dim_vector idx_B_BC,
                                        const indexed_dpd_marray_view<char>& C,
                           dim_vector idx_C_AC,
                           dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    for (auto& len : group_AB.batch_len) if (len == 0) return;
    for (auto& len : group_AC.batch_len) if (len == 0) return;
    for (auto& len : group_BC.batch_len) if (len == 0) return;

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<2> indices_A(type, A, group_AC, 0, group_AB, 0);
    TBLIS_ASSERT(B.indexed_dimension() == 0);
    group_indices<2> indices_C(type, C, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);
    scalar factor_B(0.0, type);
    factor_B.from(B.factors().data());

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    comm.do_tasks_deferred(nirrep*nidx_C*group_AC.dense_nblock*group_BC.dense_nblock,
                           group_AC.dense_size*group_BC.dense_size*group_AB.dense_size*group_AB.dense_nblock/inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        stride_type idx = 0;
        stride_type idx_A = 0;
        stride_type idx_C = 0;

        for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                   idx_C, nidx_C, indices_C, 0,
        [&](stride_type next_A, stride_type next_C)
        {
            for (auto irrep_AB0 : range(nirrep))
            {
                auto irrep_AB = irrep_AB0;
                auto irrep_AC = A.irrep()^irrep_AB0;
                auto irrep_BC = B.irrep()^irrep_AB0;

                for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
                for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
                for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
                if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                {
                    tasks.visit(idx++,
                    [&,idx_A,idx_C,next_A,next_C,
                     irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC]
                    (const communicator& subcomm)
                    {
                        auto& scat_A_AB = *(new (&scat_A_AB_) stride_vector_fuse);
                        auto& scat_B_AB = *(new (&scat_B_AB_) stride_vector_fuse);
                        auto& scat_B_BC = *(new (&scat_B_BC_) stride_vector_fuse);
                        auto& scat_C_BC = *(new (&scat_C_BC_) stride_vector_fuse);
                        auto& scat_AB = *(new (&scat_AB_) tuple_vector_fuse);
                        auto& scat_BC = *(new (&scat_BC_) tuple_vector_fuse);

                        auto local_irreps_A = irreps_A;
                        auto local_irreps_B = irreps_B;
                        auto local_irreps_C = irreps_C;

                        assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                      local_irreps_A, group_AC.dense_idx[0],
                                      local_irreps_C, group_AC.dense_idx[1]);
                        assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                      local_irreps_B, group_BC.dense_idx[0],
                                      local_irreps_C, group_BC.dense_idx[1]);

                        if (is_block_empty(dpd_C, local_irreps_C)) return;

                        marray_view<char> local_C = dpd_C(local_irreps_C);

                        for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                        {
                            assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                          local_irreps_A, group_AB.dense_idx[0],
                                          local_irreps_B, group_AB.dense_idx[1]);

                            if (is_block_empty(dpd_A, local_irreps_A)) continue;

                            marray_view<char> local_A = dpd_A(local_irreps_A);
                            marray_view<char> local_B = dpd_B(local_irreps_B);

                            len_vector len_AB;
                            stride_vector stride_A_AB, stride_B_AB;
                            get_local_geometry(indices_A[idx_A].idx[1], group_AB, len_AB,
                                               local_A, stride_A_AB, 0,
                                               local_B, stride_B_AB, 1);

                            len_vector len_BC;
                            stride_vector stride_B_BC, stride_C_BC;
                            get_local_geometry(indices_C[idx_C].idx[1], group_BC, len_BC,
                                               local_B, stride_B_BC, 0,
                                               local_C, stride_C_BC, 1);

                            len_vector len_AC;
                            stride_vector stride_A_AC, stride_C_AC;
                            stride_type off_A_AC, off_C_AC;
                            get_local_geometry(indices_A[idx_A].idx[0], group_AC, len_AC,
                                               local_A, stride_A_AC, 0,
                                               local_C, stride_C_AC, 1);
                            get_local_offset(indices_A[idx_A].idx[0], group_AC,
                                             local_A, off_A_AC, 0,
                                             local_C, off_C_AC, 1);

                            scat_AB.clear();
                            scat_BC.clear();

                            for (auto local_idx_A = idx_A;local_idx_A < next_A;local_idx_A++)
                            {
                                auto factor = alpha*indices_A[local_idx_A].factor*factor_B;
                                if (factor.is_zero()) continue;

                                stride_type off_A_AB, off_B_AB;
                                get_local_offset(indices_A[local_idx_A].idx[1], group_AB,
                                                 local_A, off_A_AB, 0,
                                                 local_B, off_B_AB, 1);

                                switch (type)
                                {
                                    case TYPE_FLOAT:
                                        scat_AB.emplace_back(factor.template get<float>(), 0.0,
                                                             off_A_AB + indices_A[local_idx_A].offset,
                                                             off_B_AB);
                                        break;
                                    case TYPE_DOUBLE:
                                        scat_AB.emplace_back(factor.template get<double>(), 0.0,
                                                             off_A_AB + indices_A[local_idx_A].offset,
                                                             off_B_AB);
                                        break;
                                    case TYPE_SCOMPLEX:
                                        scat_AB.emplace_back(factor.template get<scomplex>().real(),
                                                             factor.template get<scomplex>().imag(),
                                                             off_A_AB + indices_A[local_idx_A].offset,
                                                             off_B_AB);
                                        break;
                                    case TYPE_DCOMPLEX:
                                        scat_AB.emplace_back(factor.template get<dcomplex>().real(),
                                                             factor.template get<dcomplex>().imag(),
                                                             off_A_AB + indices_A[local_idx_A].offset,
                                                             off_B_AB);
                                        break;
                                }
                            }

                            if (scat_AB.empty()) return;

                            for (auto local_idx_C = idx_C;local_idx_C < next_C;local_idx_C++)
                            {
                                auto factor = indices_C[local_idx_C].factor;
                                if (factor.is_zero()) continue;

                                stride_type off_B_BC, off_C_BC;
                                get_local_offset(indices_C[local_idx_C].idx[1], group_BC,
                                                 local_B, off_B_BC, 0,
                                                 local_C, off_C_BC, 1);

                                switch (type)
                                {
                                    case TYPE_FLOAT:
                                        scat_BC.emplace_back(factor.get<float>(), 0.0,
                                                             off_B_BC,
                                                             off_C_BC + indices_C[local_idx_C].offset);
                                        break;
                                    case TYPE_DOUBLE:
                                        scat_BC.emplace_back(factor.get<double>(), 0.0,
                                                             off_B_BC,
                                                             off_C_BC + indices_C[local_idx_C].offset);
                                        break;
                                    case TYPE_SCOMPLEX:
                                        scat_BC.emplace_back(factor.get<scomplex>().real(),
                                                             factor.get<scomplex>().imag(),
                                                             off_B_BC,
                                                             off_C_BC + indices_C[local_idx_C].offset);
                                        break;
                                    case TYPE_DCOMPLEX:
                                        scat_BC.emplace_back(factor.get<dcomplex>().real(),
                                                             factor.get<dcomplex>().imag(),
                                                             off_B_BC,
                                                             off_C_BC + indices_C[local_idx_C].offset);
                                        break;
                                }
                            }

                            if (scat_BC.empty()) return;

                            auto data_A = A.data(0) + (local_A.data() - A.data(0) + off_A_AC)*ts;
                            auto data_B = B.data(0) + (local_B.data() - B.data(0))*ts;
                            auto data_C = C.data(0) + (local_C.data() - C.data(0) + off_C_AC)*ts;

                            for (auto i : range(scat_AB.size()))
                            {
                                scat_A_AB.push_back(std::get<2>(scat_AB[i]));
                                scat_B_AB.push_back(std::get<3>(scat_AB[i]));

                                if (i == scat_AB.size()-1 ||
                                    std::get<0>(scat_AB[i]) != std::get<0>(scat_AB[i+1]) ||
                                    std::get<1>(scat_AB[i]) != std::get<1>(scat_AB[i+1]))
                                {
                                    scalar alpha_(dcomplex(std::get<0>(scat_AB[i]),
                                                           std::get<1>(scat_AB[i])), type);

                                    for (auto j : range(scat_BC.size()))
                                    {
                                        scat_B_BC.push_back(std::get<2>(scat_BC[j]));
                                        scat_C_BC.push_back(std::get<3>(scat_BC[j]));

                                        if (j == scat_BC.size()-1 ||
                                            std::get<0>(scat_BC[j]) != std::get<0>(scat_BC[j+1]) ||
                                            std::get<1>(scat_BC[j]) != std::get<1>(scat_BC[j+1]))
                                        {
                                            scalar alpha(dcomplex(std::get<0>(scat_BC[j]),
                                                                  std::get<1>(scat_BC[j])), type);
                                            alpha *= alpha_;

                                            if (len_AB.empty() && len_BC.empty())
                                            {
                                                scatter_tensor_matrix at(alpha, conj_A, len_AC, len_AB, data_A, stride_A_AC, stride_A_AB, {},
                                                                         row_view<stride_type>{{scat_A_AB.size()}, scat_A_AB.data()});
                                                       scatter_matrix bt(  one, conj_B, data_B,
                                                                         row_view<stride_type>{{scat_B_AB.size()}, scat_B_AB.data()},
                                                                         row_view<stride_type>{{scat_B_BC.size()}, scat_B_BC.data()});
                                                scatter_tensor_matrix ct(  one,  false, len_AC, len_BC, data_C, stride_C_AC, stride_C_BC, {},
                                                                         row_view<stride_type>{{scat_C_BC.size()}, scat_C_BC.data()});

                                                GotoGEMM{}(subcomm, cfg, at, bt, ct);
                                            }
                                            else
                                            {
                                                scatter_tensor_matrix at(alpha, conj_A, len_AC, len_AB, data_A, stride_A_AC, stride_A_AB, {},
                                                                         row_view<stride_type>{{scat_A_AB.size()}, scat_A_AB.data()});
                                                scatter_tensor_matrix bt(  one, conj_B, len_AB, len_BC, data_B, stride_B_AB, stride_B_BC,
                                                                         row_view<stride_type>{{scat_B_AB.size()}, scat_B_AB.data()},
                                                                         row_view<stride_type>{{scat_B_BC.size()}, scat_B_BC.data()});
                                                scatter_tensor_matrix ct(  one,  false, len_AC, len_BC, data_C, stride_C_AC, stride_C_BC, {},
                                                                         row_view<stride_type>{{scat_C_BC.size()}, scat_C_BC.data()});

                                                GotoGEMM{}(subcomm, cfg, at, bt, ct);
                                            }

                                            scat_B_BC.clear();
                                            scat_C_BC.clear();
                                        }
                                    }

                                    scat_A_AB.clear();
                                    scat_B_AB.clear();
                                }
                            }
                        }

                        scat_A_AB.~short_vector();
                        scat_B_AB.~short_vector();
                        scat_B_BC.~short_vector();
                        scat_C_BC.~short_vector();
                        scat_AB.~short_vector();
                        scat_BC.~short_vector();
                    });
                }
            }
        });
    });
}

void mult_block(type_t type, const communicator& comm, const config& cfg,
                const scalar& alpha,
                bool conj_A, const indexed_dpd_marray_view<char>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                bool conj_B, const indexed_dpd_marray_view<char>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                             const indexed_dpd_marray_view<char>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC)
{
    const len_type ts = type_size[type];
    const auto nirrep = A.num_irreps();

    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(C, idx_C_AC, A, idx_A_AC);
    dpd_index_group<2> group_BC(C, idx_C_BC, B, idx_B_BC);

    for (auto& len : group_AB.batch_len) if (len == 0) return;
    for (auto& len : group_AC.batch_len) if (len == 0) return;
    for (auto& len : group_BC.batch_len) if (len == 0) return;

#if 1

    len_type dense_AB = group_AB.dense_size;
    len_type dense_AC = group_AC.dense_size;
    len_type dense_BC = group_BC.dense_size;
    double frac_A = double(A.num_indices())/stl_ext::prod(A.indexed_lengths());
    double frac_B = double(B.num_indices())/stl_ext::prod(B.indexed_lengths());
    double frac_C = double(C.num_indices())/stl_ext::prod(C.indexed_lengths());
    double idx_AB = stl_ext::prod(group_AB.batch_len)*sqrt(frac_A*frac_B);
    double idx_AC = stl_ext::prod(group_AC.batch_len)*sqrt(frac_A*frac_C);
    double idx_BC = stl_ext::prod(group_BC.batch_len)*sqrt(frac_B*frac_C);

    enum {FUSE_NONE, FUSE_AB_AC, FUSE_AB_BC, FUSE_AC, FUSE_BC, FUSE_AB};

    std::vector<std::pair<double,int>> fuse;

    //double baseline = relative_perf(dense_AC, dense_BC, dense_AB);
    //printf("\nmnk: %ld %ld %ld\n", dense_AC, dense_BC, dense_AB);

    fuse.emplace_back(relative_perf(dense_AC, dense_BC, dense_AB*idx_AB), FUSE_AB);
    //printf("AB: %g %g %g\n", (double)stl_ext::prod(group_AB.batch_len), sqrt(frac_A*frac_B), fuse.back().first/baseline);

    fuse.emplace_back(relative_perf(dense_AC*idx_AC, dense_BC, dense_AB), FUSE_AC);
    //printf("AC: %g %g %g\n", (double)stl_ext::prod(group_AC.batch_len), sqrt(frac_A*frac_C), fuse.back().first/baseline);

    fuse.emplace_back(relative_perf(dense_AC, dense_BC*idx_BC, dense_AB), FUSE_BC);
    //printf("BC: %g %g %g\n", (double)stl_ext::prod(group_BC.batch_len), sqrt(frac_B*frac_C), fuse.back().first/baseline);

    if (!A.indexed_dimension() && B.indexed_dimension() && C.indexed_dimension())
    {
        fuse.emplace_back(relative_perf(dense_AC*idx_AC, dense_BC, dense_AB*idx_AB), FUSE_AB_AC);
        //printf("AB_AC: %g\n", fuse.back().first/baseline);
    }

    if (A.indexed_dimension() && !B.indexed_dimension() && C.indexed_dimension())
    {
        fuse.emplace_back(relative_perf(dense_AC, dense_BC*idx_BC, dense_AB*idx_AB), FUSE_AB_BC);
        //printf("AB_BC: %g\n", fuse.back().first/baseline);
    }

    std::sort(fuse.begin(), fuse.end());

    switch (fuse.back().second)
    {
        case FUSE_AB_AC:
        {
            //printf("fuse AB AC\n");
            mult_block_fuse_AB_BC(type, comm, cfg, alpha,
                                  conj_B, B, idx_B_AB, idx_B_BC,
                                  conj_A, A, idx_A_AB, idx_A_AC,
                                          C, idx_C_BC, idx_C_AC);
        }
        break;
        case FUSE_AB_BC:
        {
            //printf("fuse AB BC\n");
            mult_block_fuse_AB_BC(type, comm, cfg, alpha,
                                  conj_A, A, idx_A_AB, idx_A_AC,
                                  conj_B, B, idx_B_AB, idx_B_BC,
                                          C, idx_C_AC, idx_C_BC);
        }
        break;
        case FUSE_AC:
        {
            //printf("fuse AC\n");
            mult_block_fuse_BC(type, comm, cfg, alpha,
                               conj_B, B, idx_B_AB, idx_B_BC,
                               conj_A, A, idx_A_AB, idx_A_AC,
                                       C, idx_C_BC, idx_C_AC);
        }
        break;
        case FUSE_BC:
        {
            //printf("fuse BC\n");
            mult_block_fuse_BC(type, comm, cfg, alpha,
                               conj_A, A, idx_A_AB, idx_A_AC,
                               conj_B, B, idx_B_AB, idx_B_BC,
                                       C, idx_C_AC, idx_C_BC);
        }
        break;
        case FUSE_AB:
        {
            //printf("fuse AB\n");
            mult_block_fuse_AB(type, comm, cfg, alpha,
                               conj_A, A, idx_A_AB, idx_A_AC,
                               conj_B, B, idx_B_AB, idx_B_BC,
                                       C, idx_C_AC, idx_C_BC);
        }
        break;
    }

    return;

#endif

    group_indices<2> indices_A(type, A, group_AC, 1, group_AB, 0);
    group_indices<2> indices_B(type, B, group_BC, 1, group_AB, 1);
    group_indices<2> indices_C(type, C, group_AC, 0, group_BC, 0);

    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    auto mixed_dim_A = stl_ext::appended(group_AC.mixed_idx[1],
                                         group_AB.mixed_idx[0]);
    auto mixed_dim_B = stl_ext::appended(group_AB.mixed_idx[1],
                                         group_BC.mixed_idx[1]);
    auto mixed_dim_C = stl_ext::appended(group_AC.mixed_idx[0],
                                         group_BC.mixed_idx[0]);

    auto mixed_irrep_A = stl_ext::appended(stl_ext::select_from(group_AC.batch_irrep, group_AC.mixed_pos[1]),
                                           stl_ext::select_from(group_AB.batch_irrep, group_AB.mixed_pos[0]));
    auto mixed_irrep_B = stl_ext::appended(stl_ext::select_from(group_AB.batch_irrep, group_AB.mixed_pos[1]),
                                           stl_ext::select_from(group_BC.batch_irrep, group_BC.mixed_pos[1]));
    auto mixed_irrep_C = stl_ext::appended(stl_ext::select_from(group_AC.batch_irrep, group_AC.mixed_pos[0]),
                                           stl_ext::select_from(group_BC.batch_irrep, group_BC.mixed_pos[0]));

    for (auto irrep_AB0 : range(nirrep))
    {
        auto irrep_AB = irrep_AB0;
        auto irrep_AC = A.irrep()^irrep_AB0;
        auto irrep_BC = B.irrep()^irrep_AB0;

        for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
        for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
        for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

        if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
        if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
        if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

        scalar one(1.0, type);

        stride_type idx = 0;
        stride_type idx_A = 0;
        stride_type idx_C = 0;

        comm.do_tasks_deferred(nidx_C,
                               group_AB.dense_size*group_AC.dense_size*group_BC.dense_size/inout_ratio,
        [&](communicator::deferred_task_set& tasks)
        {
            for_each_match<true, true>(idx_A, nidx_A, indices_A, 0,
                                       idx_C, nidx_C, indices_C, 0,
            [&](stride_type next_A, stride_type next_C)
            {
                stride_type idx_B = 0;

                for_each_match<true, false>(idx_B, nidx_B, indices_B, 0,
                                            idx_C, next_C, indices_C, 1,
                [&](stride_type next_B)
                {
                    if (indices_C[idx_C].factor.is_zero()) return;

                    tasks.visit(idx++,
                    [&,idx_A,idx_B,idx_C,next_A,next_B]
                    (const communicator& subcomm)
                    {
                        auto local_idx_A = idx_A;
                        auto local_idx_B = idx_B;

                        auto dpd_A = A[0];
                        auto dpd_B = B[0];
                        auto dpd_C = C[0];

                        for_each_match<false, false>(local_idx_A, next_A, indices_A, 1,
                                                     local_idx_B, next_B, indices_B, 1,
                        [&]
                        {
                            auto factor = alpha*indices_A[local_idx_A].factor*
                                                indices_B[local_idx_B].factor*
                                                indices_C[      idx_C].factor;
                            if (factor.is_zero()) return;

                            dpd_A.data(A.data(0) + indices_A[local_idx_A].offset*ts);
                            dpd_B.data(B.data(0) + indices_B[local_idx_B].offset*ts);
                            dpd_C.data(C.data(0) + indices_C[      idx_C].offset*ts);

                            auto mixed_idx_A = stl_ext::appended(stl_ext::select_from(indices_A[local_idx_A].idx[0], group_AC.mixed_pos[1]),
                                                                 stl_ext::select_from(indices_A[local_idx_A].idx[1], group_AB.mixed_pos[0]));
                            auto mixed_idx_B = stl_ext::appended(stl_ext::select_from(indices_B[local_idx_B].idx[1], group_AB.mixed_pos[1]),
                                                                 stl_ext::select_from(indices_B[local_idx_B].idx[0], group_BC.mixed_pos[1]));
                            auto mixed_idx_C = stl_ext::appended(stl_ext::select_from(indices_C[      idx_C].idx[0], group_AC.mixed_pos[0]),
                                                                 stl_ext::select_from(indices_C[      idx_C].idx[1], group_BC.mixed_pos[0]));

                            dpd_tensor_matrix at(factor, conj_A, dpd_A, group_AC.dense_idx[1], group_AB.dense_idx[0], irrep_AB,
                                                 mixed_dim_A, mixed_irrep_A, mixed_idx_A, group_AC.pack_3d, group_AB.pack_3d);
                            dpd_tensor_matrix bt(   one, conj_B, dpd_B, group_AB.dense_idx[1], group_BC.dense_idx[1], irrep_BC,
                                                 mixed_dim_B, mixed_irrep_B, mixed_idx_B, group_AB.pack_3d, group_BC.pack_3d);
                            dpd_tensor_matrix ct(   one,  false, dpd_C, group_AC.dense_idx[0], group_BC.dense_idx[0], irrep_BC,
                                                 mixed_dim_C, mixed_irrep_C, mixed_idx_C, group_AC.pack_3d, group_BC.pack_3d);

                            GotoGEMM{}(subcomm, cfg, at, bt, ct);
                        });
                    });
                });
            });
        });
    }
}

void mult_block(type_t type, const communicator& comm, const config& cfg,
                const scalar& alpha,
                bool conj_A, const indexed_dpd_marray_view<char>& A,
                dim_vector idx_A_AB,
                dim_vector idx_A_AC,
                dim_vector idx_A_ABC,
                bool conj_B, const indexed_dpd_marray_view<char>& B,
                dim_vector idx_B_AB,
                dim_vector idx_B_BC,
                dim_vector idx_B_ABC,
                             const indexed_dpd_marray_view<char>& C,
                dim_vector idx_C_AC,
                dim_vector idx_C_BC,
                dim_vector idx_C_ABC)
{
    const len_type ts = type_size[type];

    const auto nirrep = A.num_irreps();

    dpd_index_group<3> group_ABC(A, idx_A_ABC, B, idx_B_ABC, C, idx_C_ABC);
    dpd_index_group<2> group_AB(A, idx_A_AB, B, idx_B_AB);
    dpd_index_group<2> group_AC(A, idx_A_AC, C, idx_C_AC);
    dpd_index_group<2> group_BC(B, idx_B_BC, C, idx_C_BC);

    irrep_vector irreps_A(A.dense_dimension());
    irrep_vector irreps_B(B.dense_dimension());
    irrep_vector irreps_C(C.dense_dimension());
    assign_irreps(group_ABC, irreps_A, irreps_B, irreps_C);
    assign_irreps(group_AB, irreps_A, irreps_B);
    assign_irreps(group_AC, irreps_A, irreps_C);
    assign_irreps(group_BC, irreps_B, irreps_C);

    group_indices<3> indices_A(type, A, group_ABC, 0, group_AC, 0, group_AB, 0);
    group_indices<3> indices_B(type, B, group_ABC, 1, group_BC, 0, group_AB, 1);
    group_indices<3> indices_C(type, C, group_ABC, 2, group_AC, 1, group_BC, 1);
    auto nidx_A = indices_A.size();
    auto nidx_B = indices_B.size();
    auto nidx_C = indices_C.size();

    scalar one(1.0, type);

    auto dpd_A = A[0];
    auto dpd_B = B[0];
    auto dpd_C = C[0];

    stride_type idx = 0;
    stride_type idx_A = 0;
    stride_type idx_B0 = 0;
    stride_type idx_C = 0;

    auto irrep_ABC = A.irrep()^B.irrep()^C.irrep();
    for (auto irrep : group_ABC.batch_irrep) irrep_ABC ^= irrep;

    if (group_ABC.dense_ndim == 0 && irrep_ABC != 0) return;

    comm.do_tasks_deferred(nirrep*nidx_C*group_AC.dense_nblock*group_BC.dense_nblock*group_ABC.dense_nblock,
                           group_AB.dense_size*group_AC.dense_size*group_BC.dense_size*group_ABC.dense_size*inout_ratio,
    [&](communicator::deferred_task_set& tasks)
    {
        for_each_match<true, true, true>(idx_A,  nidx_A, indices_A, 0,
                                        idx_B0, nidx_B, indices_B, 0,
                                        idx_C,  nidx_C, indices_C, 0,
        [&](stride_type next_A_ABC, stride_type next_B_ABC, stride_type next_C_ABC)
        {
            for_each_match<true, true>(idx_A, next_A_ABC, indices_A, 1,
                                      idx_C, next_C_ABC, indices_C, 1,
            [&](stride_type next_A_AB, stride_type next_C_AC)
            {
                stride_type idx_B = idx_B0;

                for_each_match<true, false>(idx_B, next_B_ABC, indices_B, 1,
                                           idx_C,  next_C_AC, indices_C, 2,
                [&](stride_type next_B_AB)
                {
                    if (indices_C[idx_C].factor.is_zero()) return;

                    for (auto irrep_AB0 : range(nirrep))
                    {
                        auto irrep_AB = irrep_AB0;
                        auto irrep_AC = B.irrep()^C.irrep()^irrep_AB0;
                        auto irrep_BC = A.irrep()^C.irrep()^irrep_AB0;

                        for (auto irrep : group_AB.batch_irrep) irrep_AB ^= irrep;
                        for (auto irrep : group_AC.batch_irrep) irrep_AC ^= irrep;
                        for (auto irrep : group_BC.batch_irrep) irrep_BC ^= irrep;

                        if (group_AB.dense_ndim == 0 && irrep_AB != 0) continue;
                        if (group_AC.dense_ndim == 0 && irrep_AC != 0) continue;
                        if (group_BC.dense_ndim == 0 && irrep_BC != 0) continue;

                        for (stride_type block_ABC = 0;block_ABC < group_ABC.dense_nblock;block_ABC++)
                        for (stride_type block_AC = 0;block_AC < group_AC.dense_nblock;block_AC++)
                        for (stride_type block_BC = 0;block_BC < group_BC.dense_nblock;block_BC++)
                        {
                            tasks.visit(idx++,
                            [&,idx_A,idx_B,idx_C,next_A_AB,next_B_AB,
                             irrep_AB,irrep_AC,irrep_BC,block_AC,block_BC,block_ABC]
                            (const communicator& subcomm)
                            {
                                auto local_irreps_A = irreps_A;
                                auto local_irreps_B = irreps_B;
                                auto local_irreps_C = irreps_C;

                                assign_irreps(group_ABC.dense_ndim, irrep_ABC, nirrep, block_ABC,
                                              local_irreps_A, group_ABC.dense_idx[0],
                                              local_irreps_B, group_ABC.dense_idx[1],
                                              local_irreps_C, group_ABC.dense_idx[2]);

                                assign_irreps(group_AC.dense_ndim, irrep_AC, nirrep, block_AC,
                                              local_irreps_A, group_AC.dense_idx[0],
                                              local_irreps_C, group_AC.dense_idx[1]);

                                assign_irreps(group_BC.dense_ndim, irrep_BC, nirrep, block_BC,
                                              local_irreps_B, group_BC.dense_idx[0],
                                              local_irreps_C, group_BC.dense_idx[1]);

                                if (is_block_empty(dpd_C, local_irreps_C)) return;

                                marray_view<char> local_C = dpd_C(local_irreps_C);

                                for (stride_type block_AB = 0;block_AB < group_AB.dense_nblock;block_AB++)
                                {
                                    assign_irreps(group_AB.dense_ndim, irrep_AB, nirrep, block_AB,
                                                  local_irreps_A, group_AB.dense_idx[0],
                                                  local_irreps_B, group_AB.dense_idx[1]);

                                    if (is_block_empty(dpd_A, local_irreps_A)) continue;

                                    auto local_idx_A = idx_A;
                                    auto local_idx_B = idx_B;

                                    marray_view<char> local_A = dpd_A(local_irreps_A);
                                    marray_view<char> local_B = dpd_B(local_irreps_B);

                                    len_vector len_ABC;
                                    stride_vector stride_A_ABC, stride_B_ABC, stride_C_ABC;
                                    stride_type off_A_ABC, off_B_ABC, off_C_ABC;
                                    get_local_geometry(indices_A[local_idx_A].idx[0], group_ABC, len_ABC,
                                                       local_A, stride_A_ABC, 0,
                                                       local_B, stride_B_ABC, 1,
                                                       local_C, stride_C_ABC, 2);
                                    get_local_offset(indices_A[local_idx_A].idx[0], group_ABC,
                                                     local_A, off_A_ABC, 0,
                                                     local_B, off_B_ABC, 1,
                                                     local_C, off_C_ABC, 2);

                                    len_vector len_AC;
                                    stride_vector stride_A_AC, stride_C_AC;
                                    stride_type off_A_AC, off_C_AC;
                                    get_local_geometry(indices_A[local_idx_A].idx[1], group_AC, len_AC,
                                                       local_A, stride_A_AC, 0,
                                                       local_C, stride_C_AC, 1);
                                    get_local_offset(indices_A[local_idx_A].idx[1], group_AC,
                                                     local_A, off_A_AC, 0,
                                                     local_C, off_C_AC, 1);

                                    len_vector len_BC;
                                    stride_vector stride_B_BC, stride_C_BC;
                                    stride_type off_B_BC, off_C_BC;
                                    get_local_geometry(indices_B[local_idx_B].idx[1], group_BC, len_BC,
                                                       local_B, stride_B_BC, 0,
                                                       local_C, stride_C_BC, 1);
                                    get_local_offset(indices_B[local_idx_B].idx[1], group_BC,
                                                     local_B, off_B_BC, 0,
                                                     local_C, off_C_BC, 1);

                                    len_vector len_AB;
                                    stride_vector stride_A_AB, stride_B_AB;
                                    get_local_geometry(indices_A[local_idx_A].idx[2], group_AB, len_AB,
                                                       local_A, stride_A_AB, 0,
                                                       local_B, stride_B_AB, 1);

                                    auto data_C = C.data(0) + (local_C.data() - C.data(0) +
                                        indices_C[idx_C].offset + off_C_AC + off_C_BC + off_C_ABC)*ts;

                                    for_each_match<false, false>(local_idx_A, next_A_AB, indices_A, 2,
                                                                 local_idx_B, next_B_AB, indices_B, 2,
                                    [&]
                                    {
                                        auto factor = alpha*indices_A[local_idx_A].factor*
                                                            indices_B[local_idx_B].factor*
                                                            indices_C[idx_C].factor;
                                        if (factor.is_zero()) return;

                                        stride_type off_A_AB, off_B_AB;
                                        get_local_offset(indices_A[local_idx_A].idx[2], group_AB,
                                                         local_A, off_A_AB, 0,
                                                         local_B, off_B_AB, 1);

                                        auto data_A = A.data(0) + (local_A.data() - A.data(0) +
                                            indices_A[local_idx_A].offset + off_A_AB + off_A_AC + off_A_ABC)*ts;
                                        auto data_B = B.data(0) + (local_B.data() - B.data(0) +
                                            indices_B[local_idx_B].offset + off_B_AB + off_B_BC + off_B_ABC)*ts;

                                        mult(type, subcomm, cfg,
                                             len_AB, len_AC, len_BC, len_ABC,
                                             factor, conj_A, data_A, stride_A_AB, stride_A_AC, stride_A_ABC,
                                                     conj_B, data_B, stride_B_AB, stride_B_BC, stride_B_ABC,
                                                one,  false, data_C, stride_C_AC, stride_C_BC, stride_C_ABC);
                                    });
                                }
                            });
                        }
                    }
                });
            });
        });
    });
}

void mult(type_t type, const communicator& comm, const config& cfg,
          const scalar& alpha, bool conj_A, const indexed_dpd_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
                               bool conj_B, const indexed_dpd_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta, bool conj_C, const indexed_dpd_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC)
{
    if (beta.is_zero())
    {
        set(type, comm, cfg, beta, C, range(C.dimension()));
    }
    else if (!beta.is_one() || (beta.is_complex() && conj_C))
    {
        scale(type, comm, cfg, beta, conj_C, C, range(C.dimension()));
    }

    for (auto i : range(idx_A_AB.size()))
    {
        if (idx_A_AB[i] >= A.dense_dimension() &&
            idx_B_AB[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AB[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_AB[i] - B.dense_dimension())) return;
        }
    }

    for (auto i : range(idx_A_AC.size()))
    {
        if (idx_A_AC[i] >= A.dense_dimension() &&
            idx_C_AC[i] >= C.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_AC[i] - A.dense_dimension()) !=
                C.indexed_irrep(idx_C_AC[i] - C.dense_dimension())) return;
        }
    }

    for (auto i : range(idx_B_BC.size()))
    {
        if (idx_B_BC[i] >= B.dense_dimension() &&
            idx_C_BC[i] >= C.dense_dimension())
        {
            if (B.indexed_irrep(idx_B_BC[i] - B.dense_dimension()) !=
                C.indexed_irrep(idx_C_BC[i] - C.dense_dimension())) return;
        }
    }

    for (auto i : range(idx_A_ABC.size()))
    {
        if (idx_A_ABC[i] >= A.dense_dimension() &&
            idx_B_ABC[i] >= B.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_ABC[i] - A.dense_dimension()) !=
                B.indexed_irrep(idx_B_ABC[i] - B.dense_dimension())) return;
        }

        if (idx_A_ABC[i] >= A.dense_dimension() &&
            idx_C_ABC[i] >= C.dense_dimension())
        {
            if (A.indexed_irrep(idx_A_ABC[i] - A.dense_dimension()) !=
                C.indexed_irrep(idx_C_ABC[i] - C.dense_dimension())) return;
        }

        if (idx_B_ABC[i] >= B.dense_dimension() &&
            idx_C_ABC[i] >= C.dense_dimension())
        {
            if (B.indexed_irrep(idx_B_ABC[i] - B.dense_dimension()) !=
                C.indexed_irrep(idx_C_ABC[i] - C.dense_dimension())) return;
        }
    }

    if (dpd_impl == FULL)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                mult_full(comm, cfg, alpha.get<float>(),
                          conj_A, reinterpret_cast<const indexed_dpd_marray_view<float>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, reinterpret_cast<const indexed_dpd_marray_view<float>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                                  reinterpret_cast<const indexed_dpd_marray_view<float>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_DOUBLE:
                mult_full(comm, cfg, alpha.get<double>(),
                          conj_A, reinterpret_cast<const indexed_dpd_marray_view<double>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, reinterpret_cast<const indexed_dpd_marray_view<double>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                                  reinterpret_cast<const indexed_dpd_marray_view<double>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_SCOMPLEX:
                mult_full(comm, cfg, alpha.get<scomplex>(),
                          conj_A, reinterpret_cast<const indexed_dpd_marray_view<scomplex>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, reinterpret_cast<const indexed_dpd_marray_view<scomplex>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                                  reinterpret_cast<const indexed_dpd_marray_view<scomplex>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
            case TYPE_DCOMPLEX:
                mult_full(comm, cfg, alpha.get<dcomplex>(),
                          conj_A, reinterpret_cast<const indexed_dpd_marray_view<dcomplex>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, reinterpret_cast<const indexed_dpd_marray_view<dcomplex>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                                  reinterpret_cast<const indexed_dpd_marray_view<dcomplex>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
                break;
        }
    }
    else if (!idx_C_ABC.empty())
    {
        mult_block(type, comm, cfg,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC, idx_A_ABC,
                          conj_B, B, idx_B_AB, idx_B_BC, idx_B_ABC,
                                  C, idx_C_AC, idx_C_BC, idx_C_ABC);
    }
    else //this is what it is called in (t)
    {
        mult_block(type, comm, cfg,
                   alpha, conj_A, A, idx_A_AB, idx_A_AC,
                          conj_B, B, idx_B_AB, idx_B_BC,
                                  C, idx_C_AC, idx_C_BC);
    }

    comm.barrier();
}

}
}
