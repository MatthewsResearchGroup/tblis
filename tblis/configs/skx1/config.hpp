#ifndef _TBLIS_CONFIGS_SKX1_CONFIG_HPP_
#define _TBLIS_CONFIGS_SKX1_CONFIG_HPP_

#include "configs/config_builder.hpp"


EXTERN_BLIS_GEMM_UKR(bli_sgemm_haswell_asm_6x16);
EXTERN_BLIS_GEMM_UKR(bli_dgemm_haswell_asm_6x8);
EXTERN_BLIS_GEMM_UKR(bli_cgemm_haswell_asm_3x8);
EXTERN_BLIS_GEMM_UKR(bli_zgemm_haswell_asm_3x4);

namespace tblis
{

extern int skx1_check();

TBLIS_BEGIN_CONFIG(skx1)

    TBLIS_CONFIG_GEMM_MR(   6,    6,    3,    3)
    TBLIS_CONFIG_GEMM_NR(  16,    8,    8,    4)
    TBLIS_CONFIG_GEMM_KR(   8,    4,    4,    4)
    TBLIS_CONFIG_GEMM_MC( 336,  144,  150,  384)
    TBLIS_CONFIG_GEMM_NC(4080, 4080, 4080, 4080)
    TBLIS_CONFIG_GEMM_KC( 256,  256,  256,  256)

    TBLIS_CONFIG_GEMM_WRAP_UKR(bli_sgemm_haswell_asm_6x16,
                               bli_dgemm_haswell_asm_6x8,
                               bli_cgemm_haswell_asm_3x8,
                               bli_zgemm_haswell_asm_3x4)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, true, true)

    TBLIS_CONFIG_CHECK(skx1_check)

TBLIS_END_CONFIG

}

#endif
