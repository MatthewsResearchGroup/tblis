#ifndef _TBLIS_INTERNAL_3T_INDEXED_DPD_MULT_HPP_
#define _TBLIS_INTERNAL_3T_INDEXED_DPD_MULT_HPP_

#include "frame/base/thread.h"
#include "frame/base/basic_types.h"

#include "external/marray/marray/indexed_dpd/indexed_dpd_marray_view.hpp"

namespace tblis
{
namespace internal
{

using MArray::indexed_dpd_marray_view;
using MArray::dim_vector;

void mult(type_t type, const communicator& comm, const cntx_t* cntx,
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
          const dim_vector& idx_C_ABC);

}
}

#endif
