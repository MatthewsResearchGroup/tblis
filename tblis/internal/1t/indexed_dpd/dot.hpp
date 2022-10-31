#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_DOT_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void dot(type_t type, const communicator& comm, const config& cfg,
         bool conj_A, const indexed_dpd_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const indexed_dpd_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result);

}
}

#endif
