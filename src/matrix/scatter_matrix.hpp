#ifndef _TBLIS_SCATTER_MATRIX_HPP_
#define _TBLIS_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "abstract_matrix.hpp"

namespace tblis
{

struct scatter_matrix_impl
{
    char* data_ = nullptr;
    std::array<const stride_type*, 2> scatter_ = {};

    scatter_matrix_impl(char* ptr,
                        const row_view<const stride_type>& rscat,
                        const row_view<const stride_type>& cscat)
    : data_(ptr), scatter_{rscat.data(), cscat.data()} {}
};

class scatter_matrix : public abstract_matrix_adapter<scatter_matrix,scatter_matrix_impl>
{
    protected:
        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto& A = static_cast<scatter_matrix&>(A_);
            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);

            const len_type m = A.length( trans);
            const len_type k = A.length(!trans);
            const stride_type* rscat = A.scatter( trans) + A.offset( trans);
            const stride_type* cscat = A.scatter(!trans) + A.offset(!trans);

            const len_type m_p = ceil_div(m, MR)*ME;
            const len_type k_p = round_up(k, KR);
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL;

            packed_matrix P(type, !trans ? m : k, !trans ? k : m,
                            A.get_buffer(comm, nelem, pool), k_p*ME);

            tensor_matrix::pack(comm, cfg, A.scale(), A.conj(), trans, m, k,
                                A.data(), rscat, cscat, nullptr, nullptr,
                                P.data(), P.panel_stride());

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool&, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto& A = static_cast<const packed_matrix&>(A_);
            auto& B = static_cast<const packed_matrix&>(B_);
            auto& C = static_cast<     scatter_matrix&>(C_);

            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const stride_type* rscat = C.scatter(0);
            const stride_type* cscat = C.scatter(1);

            tensor_matrix::gemm(comm, cfg, C.scale(), C.conj(), m, n, k,
                                A.data(), A.panel_stride(),
                                B.data(), B.panel_stride(),
                                C.data(), rscat, cscat, nullptr, nullptr);
        }

    public:
        scatter_matrix(const scalar& alpha, bool conj, char* A,
                       const row_view<const stride_type>& rscat,
                       const row_view<const stride_type>& cscat)
        : abstract_matrix_adapter(alpha, conj, false,
                                  rscat.length(), cscat.length(),
                                  A, rscat, cscat)
        {
            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }

        char* data() const
        {
            return impl().data_;
        }

        const stride_type* scatter(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().scatter_[dim^transposed()];
        }
};

}

#endif
