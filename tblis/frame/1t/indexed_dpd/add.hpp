#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_ADD_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_ADD_HPP_

#include "util.hpp"

namespace tblis
{
namespace internal
{

void add(type_t type, const communicator& comm, const cntx_t* cntx,
         const scalar& alpha, bool conj_A, const indexed_dpd_marray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const indexed_dpd_marray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB);

}
}

#endif
