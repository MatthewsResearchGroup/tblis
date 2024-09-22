#include "../bli_plugin_tblis.h"
#include "../kernel.hpp"

namespace tblis
{

template <typename T>
void TBLIS_REF_KERNEL(shift)
    (
            len_type n,
      const void*    alpha_,
      const void*    beta_,
            bool     conj_A, void* A_, stride_type inc_A
    )
{
    T alpha = *static_cast<const T*>(alpha_);
    T beta  = *static_cast<const T*>(beta_ );

    T* TBLIS_RESTRICT A = static_cast<T*>(A_);

    if (beta == T(0))
    {
        if (inc_A == 1)
        {
            for (len_type i = 0;i < n;i++) A[i] = alpha;
        }
        else
        {
            for (len_type i = 0;i < n;i++) A[i*inc_A] = alpha;
        }
    }
    else
    {
        if (is_complex_v<T> && conj_A)
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha + beta*conj(A[i]);
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha + beta*conj(A[i*inc_A]);
            }
        }
        else
        {
            if (inc_A == 1)
            {
                #pragma omp simd
                for (len_type i = 0;i < n;i++)
                    A[i] = alpha + beta*A[i];
            }
            else
            {
                for (len_type i = 0;i < n;i++)
                    A[i*inc_A] = alpha + beta*A[i*inc_A];
            }
        }
    }
}

TBLIS_INIT_REF_KERNEL(shift)

}
