#include "mult.h"

#include "plugin/bli_plugin_tblis.h"

#include "frame/base/tensor.hpp"
#include "frame/1t/dense/scale.hpp"
#include "frame/1t/dense/set.hpp"
#include "frame/3t/dense/mult.hpp"
#include "frame/1t/dpd/scale.hpp"
#include "frame/1t/dpd/set.hpp"
#include "frame/3t/dpd/mult.hpp"
#include "frame/1t/indexed/scale.hpp"
#include "frame/1t/indexed/set.hpp"
#include "frame/3t/indexed/mult.hpp"
#include "frame/1t/indexed_dpd/scale.hpp"
#include "frame/1t/indexed_dpd/set.hpp"
#include "frame/3t/indexed_dpd/mult.hpp"

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_mult(const tblis_comm* comm,
                       const tblis_config* cntx,
                       const tblis_tensor* A,
                       const label_type* idx_A_,
                       const tblis_tensor* B,
                       const label_type* idx_B_,
                             tblis_tensor* C,
                       const label_type* idx_C_)
{
    internal::initialize_once();

    TBLIS_ASSERT(A->type == B->type);
    TBLIS_ASSERT(A->type == C->type);

    auto ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    auto ndim_B = B->ndim;
    len_vector len_B;
    stride_vector stride_B;
    label_vector idx_B;
    diagonal(ndim_B, B->len, B->stride, idx_B_, len_B, stride_B, idx_B);

    auto ndim_C = C->ndim;
    len_vector len_C;
    stride_vector stride_C;
    label_vector idx_C;
    diagonal(ndim_C, C->len, C->stride, idx_C_, len_C, stride_C, idx_C);

    /*
    auto ndim_ABC = stl_ext::intersection(idx_A, idx_B, idx_C).size();

    if (idx_A.size() == ndim_ABC ||
        idx_B.size() == ndim_ABC ||
        idx_C.size() == ndim_ABC)
    {
        len_A.push_back(1);
        len_B.push_back(1);
        len_C.push_back(1);
        len_C.push_back(1);
        stride_A.push_back(0);
        stride_B.push_back(0);
        stride_C.push_back(0);
        stride_C.push_back(0);
        label_type idx = internal::free_idx(idx_A, idx_B, idx_C);
        idx_A.push_back(idx);
        idx_C.push_back(idx);
        idx = internal::free_idx(idx_A, idx_B, idx_C);
        idx_B.push_back(idx);
        idx_C.push_back(idx);
    }
    */

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto len_ABC = stl_ext::select_from(len_A, idx_A, idx_ABC);
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_B, idx_B, idx_ABC));
    TBLIS_ASSERT(len_ABC == stl_ext::select_from(len_C, idx_C, idx_ABC));
    auto stride_A_ABC = stl_ext::select_from(stride_A, idx_A, idx_ABC);
    auto stride_B_ABC = stl_ext::select_from(stride_B, idx_B, idx_ABC);
    auto stride_C_ABC = stl_ext::select_from(stride_C, idx_C, idx_ABC);

    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto len_AC = stl_ext::select_from(len_A, idx_A, idx_AC);
    TBLIS_ASSERT(len_AC == stl_ext::select_from(len_C, idx_C, idx_AC));
    auto stride_A_AC = stl_ext::select_from(stride_A, idx_A, idx_AC);
    auto stride_C_AC = stl_ext::select_from(stride_C, idx_C, idx_AC);

    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto len_BC = stl_ext::select_from(len_B, idx_B, idx_BC);
    TBLIS_ASSERT(len_BC == stl_ext::select_from(len_C, idx_C, idx_BC));
    auto stride_B_BC = stl_ext::select_from(stride_B, idx_B, idx_BC);
    auto stride_C_BC = stl_ext::select_from(stride_C, idx_C, idx_BC);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    fold(len_ABC, idx_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_AC, idx_AC, stride_A_AC, stride_C_AC);
    fold(len_BC, idx_BC, stride_B_BC, stride_C_BC);

    len_vector nolen;
    stride_vector nostride;

    auto alpha = A->scalar*B->scalar;
    auto beta = C->scalar;

    auto data_A = reinterpret_cast<char*>(A->data);
    auto data_B = reinterpret_cast<char*>(B->data);
    auto data_C = reinterpret_cast<char*>(C->data);

    parallelize_if(
    [&](const communicator& comm)
    {
        if (alpha.is_zero())
        {
            if (beta.is_zero())
            {
                internal::set(A->type, comm, bli_gks_query_cntx(),
                              len_AC+len_BC+len_ABC, beta, data_C,
                              stride_C_AC+stride_C_BC+stride_C_ABC);
            }
            else if (!beta.is_one() || (beta.is_complex() && C->conj))
            {
                internal::scale(A->type, comm, bli_gks_query_cntx(),
                                len_AC+len_BC+len_ABC,
                                beta, C->conj, data_C,
                                stride_C_AC+stride_C_BC+stride_C_ABC);
            }
        }
        else
        {
            internal::mult(A->type, comm, bli_gks_query_cntx(),
                           len_AB, len_AC, len_BC, len_ABC,
                           alpha, A->conj, data_A,
                           stride_A_AB, stride_A_AC, stride_A_ABC,
                                  B->conj, data_B,
                           stride_B_AB, stride_B_BC, stride_B_ABC,
                            beta, C->conj, data_C,
                           stride_C_AC, stride_C_BC, stride_C_ABC);
        }
    }, comm);

    C->scalar = 1;
    C->conj = false;
}

template <typename T>
void mult(const communicator& comm,
          T alpha, const dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const dpd_marray_view<      T>& C, const label_vector& idx_C)
{
    internal::initialize_once();

    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();
    auto ndim_C = C.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     B.length(idx_B_ABC[i], irrep));
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     C.length(idx_C_ABC[i], irrep));
    }

    for (auto i : range(idx_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    for (auto i : range(idx_AC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                     C.length(idx_C_AC[i], irrep));
    }

    for (auto i : range(idx_BC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                     C.length(idx_C_BC[i], irrep));
    }

    if (alpha == T(0) || (idx_ABC.empty() && ((A.irrep()^B.irrep()) != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, bli_gks_query_cntx(),
                          beta, reinterpret_cast<const dpd_marray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, bli_gks_query_cntx(),
                            beta, false, reinterpret_cast<const dpd_marray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, bli_gks_query_cntx(),
                       alpha, false, reinterpret_cast<const dpd_marray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const dpd_marray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const dpd_marray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const dpd_marray_view<const T>& A, const label_vector& idx_A, \
                            const dpd_marray_view<const T>& B, const label_vector& idx_B, \
                   T  beta, const dpd_marray_view<      T>& C, const label_vector& idx_C);
DO_FOREACH_TYPE

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_marray_view<      T>& C, const label_vector& idx_C)
{
    internal::initialize_once();

    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();
    auto ndim_C = C.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     B.length(idx_B_ABC[i]));
        TBLIS_ASSERT(A.length(idx_A_ABC[i]) ==
                     C.length(idx_C_ABC[i]));
    }

    for (auto i : range(idx_AB.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    for (auto i : range(idx_AC.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i]) ==
                     C.length(idx_C_AC[i]));
    }

    for (auto i : range(idx_BC.size()))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i]) ==
                     C.length(idx_C_BC[i]));
    }

    if (alpha == T(0))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, bli_gks_query_cntx(),
                          beta, reinterpret_cast<const indexed_marray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, bli_gks_query_cntx(),
                            beta, false, reinterpret_cast<const indexed_marray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, bli_gks_query_cntx(),
                       alpha, false, reinterpret_cast<const indexed_marray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const indexed_marray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const indexed_marray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const indexed_marray_view<const T>& A, const label_vector& idx_A, \
                            const indexed_marray_view<const T>& B, const label_vector& idx_B, \
                   T  beta, const indexed_marray_view<      T>& C, const label_vector& idx_C);
DO_FOREACH_TYPE

template <typename T>
void mult(const communicator& comm,
          T alpha, const indexed_dpd_marray_view<const T>& A, const label_vector& idx_A,
                   const indexed_dpd_marray_view<const T>& B, const label_vector& idx_B,
          T  beta, const indexed_dpd_marray_view<      T>& C, const label_vector& idx_C)
{
    internal::initialize_once();

    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);
    TBLIS_ASSERT(C.num_irreps() == nirrep);

    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();
    auto ndim_C = C.dimension();

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    for (auto i : range(1,ndim_C))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_C[i] != idx_C[j]);

    auto idx_ABC = stl_ext::intersection(idx_A, idx_B, idx_C);
    auto idx_AB = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_B), idx_ABC);
    auto idx_AC = stl_ext::exclusion(stl_ext::intersection(idx_A, idx_C), idx_ABC);
    auto idx_BC = stl_ext::exclusion(stl_ext::intersection(idx_B, idx_C), idx_ABC);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB, idx_AC, idx_ABC);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB, idx_BC, idx_ABC);
    auto idx_C_only = stl_ext::exclusion(idx_C, idx_AC, idx_BC, idx_ABC);

    TBLIS_ASSERT(idx_A_only.empty());
    TBLIS_ASSERT(idx_B_only.empty());
    TBLIS_ASSERT(idx_C_only.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_AC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AB, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_BC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_AC, idx_ABC).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_BC, idx_ABC).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);
    dim_vector range_C = range(ndim_C);

    auto idx_A_ABC = stl_ext::select_from(range_A, idx_A, idx_ABC);
    auto idx_B_ABC = stl_ext::select_from(range_B, idx_B, idx_ABC);
    auto idx_C_ABC = stl_ext::select_from(range_C, idx_C, idx_ABC);
    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_AC = stl_ext::select_from(range_A, idx_A, idx_AC);
    auto idx_C_AC = stl_ext::select_from(range_C, idx_C, idx_AC);
    auto idx_B_BC = stl_ext::select_from(range_B, idx_B, idx_BC);
    auto idx_C_BC = stl_ext::select_from(range_C, idx_C, idx_BC);

    for (auto i : range(idx_ABC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     B.length(idx_B_ABC[i], irrep));
        TBLIS_ASSERT(A.length(idx_A_ABC[i], irrep) ==
                     C.length(idx_C_ABC[i], irrep));
    }

    for (auto i : range(idx_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    for (auto i : range(idx_AC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AC[i], irrep) ==
                     C.length(idx_C_AC[i], irrep));
    }

    for (auto i : range(idx_BC.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(B.length(idx_B_BC[i], irrep) ==
                     C.length(idx_C_BC[i], irrep));
    }

    if (alpha == T(0) || (idx_ABC.empty() && ((A.irrep()^B.irrep()) != C.irrep())))
    {
        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, bli_gks_query_cntx(),
                          beta, reinterpret_cast<const indexed_dpd_marray_view<char>&>(C), range_C);
        }
        else if (beta != T(1))
        {
            internal::scale(type_tag<T>::value, comm, bli_gks_query_cntx(),
                            beta, false, reinterpret_cast<const indexed_dpd_marray_view<char>&>(C), range_C);
        }
    }
    else
    {
        internal::mult(type_tag<T>::value, comm, bli_gks_query_cntx(),
                       alpha, false, reinterpret_cast<const indexed_dpd_marray_view<char>&>(A), idx_A_AB, idx_A_AC, idx_A_ABC,
                              false, reinterpret_cast<const indexed_dpd_marray_view<char>&>(B), idx_B_AB, idx_B_BC, idx_B_ABC,
                        beta, false, reinterpret_cast<const indexed_dpd_marray_view<char>&>(C), idx_C_AC, idx_C_BC, idx_C_ABC);
    }
}

#undef FOREACH_TYPE
#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, \
                   T alpha, const indexed_dpd_marray_view<const T>& A, const label_vector& idx_A, \
                            const indexed_dpd_marray_view<const T>& B, const label_vector& idx_B, \
                   T  beta, const indexed_dpd_marray_view<      T>& C, const label_vector& idx_C);
DO_FOREACH_TYPE

}
