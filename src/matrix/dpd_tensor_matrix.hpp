#ifndef _TBLIS_DPD_TENSOR_MATRIX_HPP_
#define _TBLIS_DPD_TENSOR_MATRIX_HPP_

#include "internal/1t/dpd/util.hpp"

#include "tensor_matrix.hpp"

namespace tblis
{

struct dpd_tensor_matrix_impl
{
    const dpd_marray_view<char>& tensor_;
    std::array<dim_vector, 2> dims_ = {};
    dim_vector extra_dims_ = {};
    irrep_vector extra_irreps_ = {};
    len_vector extra_idx_ = {};
    std::array<int, 2> irrep_ = {};
    std::array<len_vector, 2> block_size_ = {};
    std::array<len_vector, 2> block_idx_ = {};
    std::array<bool, 2> pack_3d_ = {};

    dpd_tensor_matrix_impl(const dpd_marray_view<char>& other,
                           const dim_vector& row_inds,
                           const dim_vector& col_inds,
                           int col_irrep,
                           const dim_vector& extra_inds,
                           const irrep_vector& extra_irreps,
                           const len_vector& extra_idx,
                           bool pack_m_3d, bool pack_n_3d)
    : tensor_(other), dims_{row_inds, col_inds}, extra_dims_(extra_inds),
      extra_irreps_(extra_irreps), extra_idx_(extra_idx),
      irrep_{col_irrep^other.irrep(), col_irrep},
      pack_3d_{pack_m_3d, pack_n_3d}
    {
        for (auto irrep : extra_irreps_)
            irrep_[0] ^= irrep;

        TBLIS_ASSERT((int)(dims_[0].size() + dims_[1].size() + extra_dims_.size()) == other.dimension());
        TBLIS_ASSERT(extra_dims_.size() == extra_irreps_.size());
        TBLIS_ASSERT(extra_dims_.size() == extra_idx_.size());

        const auto nirrep = other.num_irreps();

        for (auto& i : dims_[0])
        for (auto& j : dims_[1])
        {
            (void)i; (void)j;
            TBLIS_ASSERT(i != j);
        }

        for (auto dim : {0,1})
        {
            for (auto& i : dims_[dim])
            for (auto& j : extra_dims_)
            {
                (void)i; (void)j;
                TBLIS_ASSERT(i != j);
            }

            if (dims_[dim].empty())
            {
                if (irrep_[dim] == 0)
                {
                    block_size_[dim].push_back(1);
                    block_idx_[dim].push_back(0);
                }
            }
            else
            {
                MArray::irrep_iterator it(irrep_[dim], nirrep, dims_[dim].size());
                block_size_[dim].reserve(it.nblock());
                block_idx_[dim].reserve(it.nblock());
                for (int idx = 0;it.next();idx++)
                {
                    stride_type size = 1;
                    for (auto i : range(dims_[dim].size()))
                        size *= other.length(dims_[dim][i], it.irrep(i));

                    if (size == 0) continue;

                    block_size_[dim].push_back(size);
                    block_idx_[dim].push_back(idx);
                }
            }
        }
    }
};

template <typename T>
struct is_dpd_tensor_helper : std::false_type {};

template <typename T, typename Allocator>
struct is_dpd_tensor_helper<dpd_marray<T,Allocator>> : std::true_type {};

template <typename T>
struct is_dpd_tensor_helper<dpd_marray_view<T>> : std::true_type {};

template <typename T>
struct is_dpd_tensor : is_dpd_tensor_helper<typename std::decay<T>::type> {};

class dpd_tensor_matrix : public abstract_matrix_adapter<dpd_tensor_matrix,dpd_tensor_matrix_impl>
{
    protected:
        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto& A = static_cast<dpd_tensor_matrix&>(A_);
            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);
            const len_type KE = cfg.gemm_kr.extent(type);

            std::array<int,2> first_patch = {};
            std::array<len_type,2> off_patch = {};
            const stride_type ts = type_size[type];
            const len_type m = A.length(trans);
            const len_type m_p = A.get_patches( trans, first_patch[ trans], off_patch[ trans], MR, ME);
            const len_type k_p = A.get_patches(!trans, first_patch[!trans], off_patch[!trans], KE, KE);
            const stride_type ps_ap = k_p*ME;
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL +
                                      2*size_as_type<stride_type>(MC+KC, type);

            packed_matrix P(type, !trans ? m : k_p, !trans ? k_p : m,
                            A.get_buffer(comm, nelem, pool), ps_ap);

            stride_type* rscat = convert_and_align<stride_type>(P.data() + m_p*k_p*ts);
            stride_type* cscat = rscat + m_p;
            stride_type* rbs = cscat + k_p;
            stride_type* cbs = rbs + m_p;

            char* p_ap0 = P.data();

            A.for_each_patch(trans, first_patch[trans], off_patch[trans],
            [&](int m_patch, len_type m_patch_size, len_type m_off_patch)
            {
                char* p_ap1 = p_ap0;

                A.for_each_patch(!trans, first_patch[!trans], off_patch[!trans],
                [&](int k_patch, len_type k_patch_size, len_type k_off_patch)
                {
                    char* p_a0 = !trans ?
                        A.fill_block_scatter(comm, MR, KR,
                                             m_patch, m_off_patch, m_patch_size,
                                             k_patch, k_off_patch, k_patch_size,
                                             rscat, cscat, rbs, cbs) :
                        A.fill_block_scatter(comm, KR, MR,
                                             k_patch, k_off_patch, k_patch_size,
                                             m_patch, m_off_patch, m_patch_size,
                                             cscat, rscat, cbs, rbs);

                    TBLIS_ASSERT(p_ap1 >= P.data());
                    TBLIS_ASSERT(p_ap1+k_patch_size*ME*ts <= P.data()+m_p*k_p*ts);

                    tensor_matrix::pack(comm, cfg, A.scale(), A.conj(),
                                        trans, m_patch_size, k_patch_size,
                                        p_a0, rscat, cscat, rbs, cbs,
                                        p_ap1, ps_ap);

                    comm.barrier();

                    p_ap1 += round_up(k_patch_size,KE)*ME*ts;
                });

                p_ap0 += ceil_div(m_patch_size,MR)*ps_ap*ts;
            });

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool& pool, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto& A = static_cast<const packed_matrix&>(A_);
            auto& B = static_cast<const packed_matrix&>(B_);
            auto& C = static_cast<  dpd_tensor_matrix&>(C_);

            const type_t type = C.type();
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type NR = cfg.gemm_nr.def(type);

            std::array<int,2> first_patch;
            std::array<len_type,2> off_patch;
            C.get_patches(0, first_patch[0], off_patch[0]);
            C.get_patches(1, first_patch[1], off_patch[1]);

            const stride_type ts = type_size[type];
            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const stride_type ps_a = A.panel_stride();
            const stride_type ps_b = B.panel_stride();
            const len_type MC = std::max(cfg.gemm_mc.max(type), m);
            const len_type NC = std::max(cfg.gemm_nc.max(type), n);
            const stride_type nelem = 2*size_as_type<stride_type>(MC+NC, type);

            stride_type* rscat = convert_and_align<stride_type>(C.get_buffer(comm, nelem, pool));
            stride_type* cscat = rscat + m;
            stride_type* rbs = cscat + n;
            stride_type* cbs = rbs + m;

            char* p_b0 = B.data();

            C.for_each_patch(1, first_patch[1], off_patch[1],
            [&](int n_patch, len_type n_patch_size, len_type n_off_patch)
            {
                char* p_a0 = A.data();

                C.for_each_patch(0, first_patch[0], off_patch[0],
                [&](int m_patch, len_type m_patch_size, len_type m_off_patch)
                {
                    auto p_c0 =
                        C.fill_block_scatter(comm, MR, NR,
                                             m_patch, m_off_patch, m_patch_size,
                                             n_patch, n_off_patch, n_patch_size,
                                             rscat, cscat, rbs, cbs);

                    tensor_matrix::gemm(comm, cfg, C.scale(), C.conj(),
                                        m_patch_size, n_patch_size, k,
                                        p_a0, ps_a, p_b0, ps_b,
                                        p_c0, rscat, cscat, rbs,  cbs);

                    p_a0 += ceil_div(m_patch_size,MR)*ps_a*ts;
                });

                p_b0 += ceil_div(n_patch_size,NR)*ps_b*ts;
            });
        }

    public:
        dpd_tensor_matrix(const tblis_scalar& alpha, bool conj,
                          const dpd_marray_view<char>& other,
                          const dim_vector& row_inds,
                          const dim_vector& col_inds,
                          int col_irrep,
                          const dim_vector& extra_inds,
                          const irrep_vector& extra_irreps,
                          const len_vector& extra_idx,
                          bool pack_row_3d = false,
                          bool pack_col_3d = false)
        : abstract_matrix_adapter(alpha, conj, false, 0, 0,
                                  other, row_inds, col_inds, col_irrep,
                                  extra_inds, extra_irreps, extra_idx,
                                  pack_row_3d, pack_col_3d)
        {
            std::array<len_vector,1> len;
            std::array<stride_vector,1> stride;
            internal::dense_total_lengths_and_strides(len, stride, other, dims(1));

            len_type m = stl_ext::sum(block_size(0));
            len_type n = stl_ext::sum(block_size(1));
            bool row_major = !stride[0].empty() &&
                             !dims(1).empty() &&
                             stride[0][dims(1)[0]] == 1 &&
                             n > 1;

            reset(row_major, m, n);

            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }

        template <typename Func>
        void for_each_patch(int dim, int patch, len_type off, Func&& func)
        {
            len_type left = length(dim);
            while (left > 0)
            {
                len_type size = std::min(block_size(dim)[patch]-off, left);

                func(patch, size, off);

                off = 0;
                left -= size;
                patch++;
            }
        }

        len_type get_patches(int dim, int& first, len_type& offset,
                             len_type MR=1, len_type ME=1)
        {
            auto& patch_size = block_size(dim);

            len_type off = this->offset(dim);
            len_type idx = 0;
            while (off >= patch_size[idx]) off -= patch_size[idx++];

            first = idx;
            offset = off;

            len_type len_round = 0;
            for_each_patch(dim, first, offset,
            [&](auto, auto size, auto)
            {
                len_round += ceil_div(size,MR)*ME;
            });

            return len_round;
        }

        char* fill_block_scatter(const communicator& comm,
                                 len_type MR, len_type NR,
                                 len_type m_patch, len_type m_off_patch, len_type m_patch_size,
                                 len_type n_patch, len_type n_off_patch, len_type n_patch_size,
                                 stride_type* rscat, stride_type* cscat,
                                 stride_type* rbs, stride_type* cbs)
        {
            const type_t type = this->type();
            const stride_type ts = type_size[type];

            const auto nirrep = tensor().num_irreps();
            const int irrep_mask = nirrep - 1;
            const int irrep_bits = __builtin_popcount(irrep_mask);

            irrep_vector irreps(tensor().dimension());

            for (auto i : range(extra_dims().size()))
                irreps[extra_dims()[i]] = extra_irreps()[i];

            for (auto dim : {0,1})
            {
                auto& dims = this->dims(dim);

                if (dims.empty()) continue;

                auto idx = block_idx(dim)[dim == 0 ? m_patch : n_patch];
                TBLIS_ASSERT(idx < (1u << irrep_bits*(std::max<int>(dims.size(),1)-1)));
                irreps[dims[0]] = irrep(dim);
                for (auto i : range(1,dims.size()))
                {
                    irreps[dims[0]] ^=
                        irreps[dims[i]] = idx & irrep_mask;
                    idx >>= irrep_bits;
                }
            }

            auto& A = tensor();
            marray_view<char> A2 = A(irreps);

            auto len_m = stl_ext::select_from(A2.lengths(), dims(0));
            auto len_n = stl_ext::select_from(A2.lengths(), dims(1));
            auto stride_m = stl_ext::select_from(A2.strides(), dims(0));
            auto stride_n = stl_ext::select_from(A2.strides(), dims(1));

            auto p_a = A.data() + (A2.data()-A.data())*ts;

            for (auto i : range(extra_dims().size()))
                p_a += A2.stride(extra_dims()[i])*extra_idx()[i]*ts;

            comm.barrier();

            if (comm.master())
            {
                tensor_matrix::fill_block_scatter(type, len_m, stride_m, MR,
                                                  m_off_patch, m_patch_size,
                                                  rscat, rbs,
                                                  pack_3d(0));

                tensor_matrix::fill_block_scatter(type, len_n, stride_n, NR,
                                                  n_off_patch, n_patch_size,
                                                  cscat, cbs,
                                                  pack_3d(1));
            }

            comm.barrier();

            return p_a;
        }

        const len_vector& block_idx(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().block_idx_[dim^transposed()];
        }

        const len_vector& block_size(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().block_size_[dim^transposed()];
        }

        const dim_vector& dims(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().dims_[dim^transposed()];
        }

        const dim_vector& extra_dims() const
        {
            return impl().extra_dims_;
        }

        const irrep_vector& extra_irreps() const
        {
            return impl().extra_irreps_;
        }

        const len_vector& extra_idx() const
        {
            return impl().extra_idx_;
        }

        const dpd_marray_view<char>& tensor() const
        {
            return impl().tensor_;
        }

        bool pack_3d(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().pack_3d_[dim^transposed()];
        }

        int irrep(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().irrep_[dim^transposed()];
        }
};

}

#endif
