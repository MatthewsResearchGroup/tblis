#ifndef MARRAY_VECTOR_AVX512_HPP
#define MARRAY_VECTOR_AVX512_HPP

#if defined(__AVX512F__)

#include <x86intrin.h>

#include "vector.hpp"

namespace MArray
{

template <>
struct vector_traits<float>
{
    static constexpr int vector_width = 16;
    static constexpr size_t alignment = 64;
    typedef __m512 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512 v)
    {
        return _mm512_cvtps_pd(_mm512_castps512_ps256(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512 v)
    {
        // (15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m512 tmp1 = _mm512_shuffle_f32x4(v, v, _MM_PERM_BBAA);
        // ( 7, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0)
        __m512 tmp2 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(tmp1), _MM_PERM_BBAA));
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_unpacklo_ps(tmp2, _mm512_setzero_ps());
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512 v)
    {
        // (15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m512 tmp1 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(v), _MM_PERM_BBAA));
        // (11,10,11,10, 9, 8, 9, 8, 3, 2, 3, 2, 1, 0, 1, 0)
        __m512d tmp2 = _mm512_cvtps_pd(_mm512_castps512_ps256(_mm512_permute_ps(tmp1, _MM_PERM_BBAA)));
        // ( 3, 3, 2, 2, 1, 1, 0, 0)
        return _mm512_unpacklo_pd(tmp2, _mm512_setzero_pd());
        // ( -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m512i>
    convert(__m512 v)
    {
        __m512i i32 = _mm512_cvtps_epi32(v);
        __m256i i32lo = _mm512_extracti64x4_epi64(i32, 0);
        __m256i i32hi = _mm512_extracti64x4_epi64(i32, 1);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32lo, i32hi), _MM_PERM_DBCA);
        __m256i i8 = _mm256_permute4x64_epi64(_mm256_packs_epi16(i16, i16), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512 v)
    {
        __m512i i32 = _mm512_cvtps_epi32(v);
        __m256i i32lo = _mm512_extracti64x4_epi64(i32, 0);
        __m256i i32hi = _mm512_extracti64x4_epi64(i32, 1);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32lo, i32hi), _MM_PERM_DBCA);
        __m256i i8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(i16, i16), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m512i>
    convert(__m512 v)
    {
        __m512i i32 = _mm512_cvtps_epi32(v);
        __m256i i32lo = _mm512_extracti64x4_epi64(i32, 0);
        __m256i i32hi = _mm512_extracti64x4_epi64(i32, 1);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32lo, i32hi), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512 v)
    {
        __m512i i32 = _mm512_cvtps_epi32(v);
        __m256i i32lo = _mm512_extracti64x4_epi64(i32, 0);
        __m256i i32hi = _mm512_extracti64x4_epi64(i32, 1);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32lo, i32hi), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_cvtps_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_setr_epi32((uint32_t)v[ 0], (uint32_t)v[ 1],
                                 (uint32_t)v[ 2], (uint32_t)v[ 3],
                                 (uint32_t)v[ 4], (uint32_t)v[ 5],
                                 (uint32_t)v[ 6], (uint32_t)v[ 7],
                                 (uint32_t)v[ 8], (uint32_t)v[ 9],
                                 (uint32_t)v[10], (uint32_t)v[11],
                                 (uint32_t)v[12], (uint32_t)v[13],
                                 (uint32_t)v[14], (uint32_t)v[15]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_setr_epi64((T)v[ 0], (T)v[ 1],
                                 (T)v[ 2], (T)v[ 3],
                                 (T)v[ 4], (T)v[ 5],
                                 (T)v[ 6], (T)v[ 7]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_loadu_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_load_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_load_pd((double*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_broadcast_f32x4(_mm_loadu_ps(ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512>
    load(const float* ptr)
    {
        return _mm512_broadcast_f32x4(_mm_load_ps(ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m512>
    load(const float* ptr)
    {
        return _mm512_castpd_ps(_mm512_set1_pd(*(double*)ptr));
    }

    static __m512 load1(const float* ptr)
    {
        return _mm512_set1_ps(*ptr);
    }

    static __m512 set1(float val)
    {
        return _mm512_set1_ps(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m512 v, float* ptr)
    {
        _mm512_storeu_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m512 v, float* ptr)
    {
        _mm512_store_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512 v, float* ptr)
    {
        _mm256_storeu_ps(ptr, _mm512_castps512_ps256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512 v, float* ptr)
    {
        _mm256_store_ps(ptr, _mm512_castps512_ps256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512 v, float* ptr)
    {
        _mm_storeu_ps(ptr, _mm512_castps512_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512 v, float* ptr)
    {
        _mm_store_ps(ptr, _mm512_castps512_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m512 v, float* ptr)
    {
        _mm_store_sd((double*)ptr, _mm_castps_pd(_mm512_castps512_ps128(v)));
    }

    static __m512 add(__m512 a, __m512 b)
    {
        return _mm512_add_ps(a, b);
    }

    static __m512 sub(__m512 a, __m512 b)
    {
        return _mm512_sub_ps(a, b);
    }

    static __m512 mul(__m512 a, __m512 b)
    {
        return _mm512_mul_ps(a, b);
    }

    static __m512 div(__m512 a, __m512 b)
    {
        return _mm512_div_ps(a, b);
    }

    static __m512 pow(__m512 a, __m512 b)
    {
        return _mm512_setr_ps(std::pow((float)a[ 0], (float)b[ 0]),
                              std::pow((float)a[ 1], (float)b[ 1]),
                              std::pow((float)a[ 2], (float)b[ 2]),
                              std::pow((float)a[ 3], (float)b[ 3]),
                              std::pow((float)a[ 4], (float)b[ 4]),
                              std::pow((float)a[ 5], (float)b[ 5]),
                              std::pow((float)a[ 6], (float)b[ 6]),
                              std::pow((float)a[ 7], (float)b[ 7]),
                              std::pow((float)a[ 8], (float)b[ 8]),
                              std::pow((float)a[ 9], (float)b[ 9]),
                              std::pow((float)a[10], (float)b[10]),
                              std::pow((float)a[11], (float)b[11]),
                              std::pow((float)a[12], (float)b[12]),
                              std::pow((float)a[13], (float)b[13]),
                              std::pow((float)a[14], (float)b[14]),
                              std::pow((float)a[15], (float)b[15]));
    }

    static __m256 abs(__m256 a)
    {
        return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    }

    static __m512 negate(__m512 a)
    {
        return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a),
                                                    _mm512_castps_si512(_mm512_set1_ps(-0.0f))));
    }

    static __m512 exp(__m512 a)
    {
        return _mm512_setr_ps(std::exp((float)a[ 0]),
                              std::exp((float)a[ 1]),
                              std::exp((float)a[ 2]),
                              std::exp((float)a[ 3]),
                              std::exp((float)a[ 4]),
                              std::exp((float)a[ 5]),
                              std::exp((float)a[ 6]),
                              std::exp((float)a[ 7]),
                              std::exp((float)a[ 8]),
                              std::exp((float)a[ 9]),
                              std::exp((float)a[10]),
                              std::exp((float)a[11]),
                              std::exp((float)a[12]),
                              std::exp((float)a[13]),
                              std::exp((float)a[14]),
                              std::exp((float)a[15]));
    }

    static __m512 log(__m512 a)
    {
        return _mm512_setr_ps(std::log((float)a[ 0]),
                              std::log((float)a[ 1]),
                              std::log((float)a[ 2]),
                              std::log((float)a[ 3]),
                              std::log((float)a[ 4]),
                              std::log((float)a[ 5]),
                              std::log((float)a[ 6]),
                              std::log((float)a[ 7]),
                              std::log((float)a[ 8]),
                              std::log((float)a[ 9]),
                              std::log((float)a[10]),
                              std::log((float)a[11]),
                              std::log((float)a[12]),
                              std::log((float)a[13]),
                              std::log((float)a[14]),
                              std::log((float)a[15]));
    }

    static __m512 abs(__m512 a)
    {
        return _mm512_and_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    }

    static __m512 sqrt(__m512 a)
    {
        return _mm512_sqrt_ps(a);
    }
};

template <>
struct vector_traits<double>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 64;
    typedef __m512d vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512d v)
    {
        __m512 lo = _mm512_castps256_ps512(_mm512_cvtpd_ps(v));
        return _mm512_shuffle_f32x4(lo, lo, _MM_PERM_BABA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512d v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512d v)
    {
        // ( 7, 6, 5, 4, 3, 2, 1, 0)
        __m512 sp = _mm512_castps256_ps512(_mm512_cvtpd_ps(v));
        __m512 tmp1 = _mm512_shuffle_f32x4(sp, sp, _MM_PERM_BBAA);
        // ( 7, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0)
        __m512 tmp2 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(tmp1), _MM_PERM_BBAA));
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_unpacklo_ps(tmp2, _mm512_setzero_ps());
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512d v)
    {
        // ( 7, 6, 5, 4, 3, 2, 1, 0)
        __m512d tmp1 = _mm512_shuffle_f64x2(v, v, _MM_PERM_BBAA);
        // ( 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_shuffle_pd(tmp1, _mm512_setzero_pd(), 0xcc);
        // ( -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m512i>
    convert(__m512d v)
    {
        __m256i i32 = _mm512_cvtpd_epi32(v);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_PERM_DBCA);
        __m256i i8 = _mm256_packs_epi16(i16, i16);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512d v)
    {
        __m256i i32 = _mm512_cvtpd_epi32(v);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_PERM_DBCA);
        __m256i i8 = _mm256_packus_epi16(i16, i16);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m512i>
    convert(__m512d v)
    {
        __m256i i32 = _mm512_cvtpd_epi32(v);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512d v)
    {
        __m256i i32 = _mm512_cvtpd_epi32(v);
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m512i>
    convert(__m512d v)
    {
        __m256i i32 = _mm512_cvtpd_epi32(v);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i32), i32, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512d v)
    {
        return _mm512_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
                                 (uint32_t)v[2], (uint32_t)v[3],
                                 (uint32_t)v[4], (uint32_t)v[5],
                                 (uint32_t)v[6], (uint32_t)v[7],
                                 (uint32_t)v[0], (uint32_t)v[1],
                                 (uint32_t)v[2], (uint32_t)v[3],
                                 (uint32_t)v[4], (uint32_t)v[5],
                                 (uint32_t)v[6], (uint32_t)v[7]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512d v)
    {
        return _mm512_setr_epi64((T)v[0], (T)v[1],
                                 (T)v[2], (T)v[3],
                                 (T)v[4], (T)v[5],
                                 (T)v[6], (T)v[7]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_loadu_pd(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_load_pd(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_broadcast_f64x4(_mm256_loadu_pd(ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_broadcast_f64x4(_mm256_load_pd(ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_broadcast_f64x2(_mm_loadu_pd(ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m512d>
    load(const double* ptr)
    {
        return _mm512_broadcast_f64x2(_mm_load_pd(ptr));
    }

    static __m512d load1(const double* ptr)
    {
        return _mm512_set1_pd(*ptr);
    }

    static __m512d set1(double val)
    {
        return _mm512_set1_pd(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512d v, double* ptr)
    {
        _mm512_storeu_pd(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512d v, double* ptr)
    {
        _mm512_store_pd(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512d v, double* ptr)
    {
        _mm256_storeu_pd(ptr, _mm512_castpd512_pd256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512d v, double* ptr)
    {
        _mm256_store_pd(ptr, _mm512_castpd512_pd256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m512d v, double* ptr)
    {
        _mm_storeu_pd(ptr, _mm512_castpd512_pd128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m512d v, double* ptr)
    {
        _mm_store_pd(ptr, _mm512_castpd512_pd128(v));
    }

    static __m512d add(__m512d a, __m512d b)
    {
        return _mm512_add_pd(a, b);
    }

    static __m512d sub(__m512d a, __m512d b)
    {
        return _mm512_sub_pd(a, b);
    }

    static __m512d mul(__m512d a, __m512d b)
    {
        return _mm512_mul_pd(a, b);
    }

    static __m512d div(__m512d a, __m512d b)
    {
        return _mm512_div_pd(a, b);
    }

    static __m512d pow(__m512d a, __m512d b)
    {
        return _mm512_setr_pd(std::pow((double)a[0], (double)b[0]),
                              std::pow((double)a[1], (double)b[1]),
                              std::pow((double)a[2], (double)b[2]),
                              std::pow((double)a[3], (double)b[3]),
                              std::pow((double)a[4], (double)b[4]),
                              std::pow((double)a[5], (double)b[5]),
                              std::pow((double)a[6], (double)b[6]),
                              std::pow((double)a[7], (double)b[7]));
    }

    static __m512d negate(__m512d a)
    {
        return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a),
                                                    _mm512_castpd_si512(_mm512_set1_pd(-0.0))));
    }

    static __m512d exp(__m512d a)
    {
        return _mm512_setr_pd(std::exp((double)a[0]),
                              std::exp((double)a[1]),
                              std::exp((double)a[2]),
                              std::exp((double)a[3]),
                              std::exp((double)a[4]),
                              std::exp((double)a[5]),
                              std::exp((double)a[6]),
                              std::exp((double)a[7]));
    }

    static __m512d log(__m512d a)
    {
        return _mm512_setr_pd(std::log((double)a[0]),
                              std::log((double)a[1]),
                              std::log((double)a[2]),
                              std::log((double)a[3]),
                              std::log((double)a[4]),
                              std::log((double)a[5]),
                              std::log((double)a[6]),
                              std::log((double)a[7]));
    }

    static __m512d abs(__m512d a)
    {
        return _mm512_and_pd(a, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFull)));
    }

    static __m512d sqrt(__m512d a)
    {
        return _mm512_sqrt_pd(a);
    }
};

template <>
struct vector_traits<std::complex<float>>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 64;
    typedef __m512 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512 v)
    {
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
        __m512 tmp1 = _mm512_permute_ps(v, _MM_PERM_DBCA);
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        __m512 tmp2 = _mm512_castpd_ps(_mm512_permutex_pd(_mm512_castps_pd(tmp1), _MM_PERM_DBCA));
        // ( 7, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0)
        return _mm512_shuffle_f32x4(tmp2, tmp2, _MM_PERM_DBCA);
        // ( 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512 v)
    {
        return _mm512_cvtps_pd(_mm512_castps512_ps256(convert<float>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512 v)
    {
        return _mm512_cvtps_pd(_mm512_castps512_ps256(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m512i>
    convert(__m512 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(_mm512_castps512_ps256(convert<float>(v)));
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_PERM_DBCA);
        __m256i i8 = _mm256_packs_epi16(i16, i16);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(_mm512_castps512_ps256(convert<float>(v)));
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_PERM_DBCA);
        __m256i i8 = _mm256_packus_epi16(i16, i16);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m512i>
    convert(__m512 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(_mm512_castps512_ps256(convert<float>(v)));
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(_mm512_castps512_ps256(convert<float>(v)));
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_cvtps_epi32(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_setr_epi32((uint32_t)v[ 0], (uint32_t)v[ 2],
                                 (uint32_t)v[ 4], (uint32_t)v[ 6],
                                 (uint32_t)v[ 8], (uint32_t)v[10],
                                 (uint32_t)v[12], (uint32_t)v[14],
                                 (uint32_t)v[ 0], (uint32_t)v[ 2],
                                 (uint32_t)v[ 4], (uint32_t)v[ 6],
                                 (uint32_t)v[ 8], (uint32_t)v[10],
                                 (uint32_t)v[12], (uint32_t)v[14]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512 v)
    {
        return _mm512_setr_epi64((T)v[ 0], (T)v[ 2],
                                 (T)v[ 4], (T)v[ 6],
                                 (T)v[ 8], (T)v[10],
                                 (T)v[12], (T)v[14]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_loadu_ps((float*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_load_ps((float*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_loadu_pd((double*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_load_pd((double*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x2(_mm_loadu_pd((double*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m512>
    load(const std::complex<float>* ptr)
    {
        return _mm512_castpd_ps(_mm512_broadcast_f64x2(_mm_load_pd((double*)ptr)));
    }

    static __m512 load1(const std::complex<float>* ptr)
    {
        return set1(*ptr);
    }

    static __m512 set1(std::complex<float> val)
    {
        double dval;
        memcpy(&dval, &val, 8);
        return _mm512_castpd_ps(_mm512_set1_pd(dval));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm512_storeu_ps((float*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm512_store_ps((float*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm256_storeu_ps((float*)ptr, _mm512_castps512_ps256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm256_store_ps((float*)ptr, _mm512_castps512_ps256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm_storeu_ps((float*)ptr, _mm512_castps512_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m512 v, std::complex<float>* ptr)
    {
        _mm_store_ps((float*)ptr, _mm512_castps512_ps128(v));
    }

    static __m512 add(__m512 a, __m512 b)
    {
        return _mm512_add_ps(a, b);
    }

    static __m512 sub(__m512 a, __m512 b)
    {
        return _mm512_sub_ps(a, b);
    }

    static __m512 mul(__m512 a, __m512 b)
    {
        __m512 ashuf = _mm512_permute_ps(a, _MM_PERM_CDAB);
        __m512 breal = _mm512_moveldup_ps(b);
        __m512 bimag = _mm512_movehdup_ps(b);
        __m512 tmp = _mm512_mul_ps(ashuf, bimag); // tmp = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm512_fmaddsub_ps(a, breal, tmp); //       (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
    }

    static __m512 div(__m512 a, __m512 b)
    {
        std::complex<float> a0((float)a[ 0], (float)a[ 1]);
        std::complex<float> a1((float)a[ 2], (float)a[ 3]);
        std::complex<float> a2((float)a[ 4], (float)a[ 5]);
        std::complex<float> a3((float)a[ 6], (float)a[ 7]);
        std::complex<float> a4((float)a[ 8], (float)a[ 9]);
        std::complex<float> a5((float)a[10], (float)a[11]);
        std::complex<float> a6((float)a[12], (float)a[13]);
        std::complex<float> a7((float)a[14], (float)a[15]);
        std::complex<float> b0((float)b[ 0], (float)b[ 1]);
        std::complex<float> b1((float)b[ 2], (float)b[ 3]);
        std::complex<float> b2((float)b[ 4], (float)b[ 5]);
        std::complex<float> b3((float)b[ 6], (float)b[ 7]);
        std::complex<float> b4((float)b[ 8], (float)b[ 9]);
        std::complex<float> b5((float)b[10], (float)b[11]);
        std::complex<float> b6((float)b[12], (float)b[13]);
        std::complex<float> b7((float)b[14], (float)b[15]);
        std::complex<float> c0 = a0 / b0;
        std::complex<float> c1 = a1 / b1;
        std::complex<float> c2 = a2 / b2;
        std::complex<float> c3 = a3 / b3;
        std::complex<float> c4 = a4 / b4;
        std::complex<float> c5 = a5 / b5;
        std::complex<float> c6 = a6 / b6;
        std::complex<float> c7 = a7 / b7;
        return _mm512_setr_ps(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag(),
                              c4.real(), c4.imag(),
                              c5.real(), c5.imag(),
                              c6.real(), c6.imag(),
                              c7.real(), c7.imag());
    }

    static __m512 pow(__m512 a, __m512 b)
    {
        std::complex<float> a0((float)a[ 0], (float)a[ 1]);
        std::complex<float> a1((float)a[ 2], (float)a[ 3]);
        std::complex<float> a2((float)a[ 4], (float)a[ 5]);
        std::complex<float> a3((float)a[ 6], (float)a[ 7]);
        std::complex<float> a4((float)a[ 8], (float)a[ 9]);
        std::complex<float> a5((float)a[10], (float)a[11]);
        std::complex<float> a6((float)a[12], (float)a[13]);
        std::complex<float> a7((float)a[14], (float)a[15]);
        std::complex<float> b0((float)b[ 0], (float)b[ 1]);
        std::complex<float> b1((float)b[ 2], (float)b[ 3]);
        std::complex<float> b2((float)b[ 4], (float)b[ 5]);
        std::complex<float> b3((float)b[ 6], (float)b[ 7]);
        std::complex<float> b4((float)b[ 8], (float)b[ 9]);
        std::complex<float> b5((float)b[10], (float)b[11]);
        std::complex<float> b6((float)b[12], (float)b[13]);
        std::complex<float> b7((float)b[14], (float)b[15]);
        std::complex<float> c0 = std::pow(a0, b0);
        std::complex<float> c1 = std::pow(a1, b1);
        std::complex<float> c2 = std::pow(a2, b2);
        std::complex<float> c3 = std::pow(a3, b3);
        std::complex<float> c4 = std::pow(a4, b4);
        std::complex<float> c5 = std::pow(a5, b5);
        std::complex<float> c6 = std::pow(a6, b6);
        std::complex<float> c7 = std::pow(a7, b7);
        return _mm512_setr_ps(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag(),
                              c4.real(), c4.imag(),
                              c5.real(), c5.imag(),
                              c6.real(), c6.imag(),
                              c7.real(), c7.imag());
    }

    static __m512 negate(__m512 a)
    {
        return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a),
                                                    _mm512_castps_si512(_mm512_set1_ps(-0.0f))));
    }

    static __m512 exp(__m512 a)
    {
        std::complex<float> a0((float)a[ 0], (float)a[ 1]);
        std::complex<float> a1((float)a[ 2], (float)a[ 3]);
        std::complex<float> a2((float)a[ 4], (float)a[ 5]);
        std::complex<float> a3((float)a[ 6], (float)a[ 7]);
        std::complex<float> a4((float)a[ 8], (float)a[ 9]);
        std::complex<float> a5((float)a[10], (float)a[11]);
        std::complex<float> a6((float)a[12], (float)a[13]);
        std::complex<float> a7((float)a[14], (float)a[15]);
        std::complex<float> b0 = std::exp(a0);
        std::complex<float> b1 = std::exp(a1);
        std::complex<float> b2 = std::exp(a2);
        std::complex<float> b3 = std::exp(a3);
        std::complex<float> b4 = std::exp(a4);
        std::complex<float> b5 = std::exp(a5);
        std::complex<float> b6 = std::exp(a6);
        std::complex<float> b7 = std::exp(a7);
        return _mm512_setr_ps(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag(),
                              b4.real(), b4.imag(),
                              b5.real(), b5.imag(),
                              b6.real(), b6.imag(),
                              b7.real(), b7.imag());
    }

    static __m512 sqrt(__m512 a)
    {
        std::complex<float> a0((float)a[ 0], (float)a[ 1]);
        std::complex<float> a1((float)a[ 2], (float)a[ 3]);
        std::complex<float> a2((float)a[ 4], (float)a[ 5]);
        std::complex<float> a3((float)a[ 6], (float)a[ 7]);
        std::complex<float> a4((float)a[ 8], (float)a[ 9]);
        std::complex<float> a5((float)a[10], (float)a[11]);
        std::complex<float> a6((float)a[12], (float)a[13]);
        std::complex<float> a7((float)a[14], (float)a[15]);
        std::complex<float> b0 = std::sqrt(a0);
        std::complex<float> b1 = std::sqrt(a1);
        std::complex<float> b2 = std::sqrt(a2);
        std::complex<float> b3 = std::sqrt(a3);
        std::complex<float> b4 = std::sqrt(a4);
        std::complex<float> b5 = std::sqrt(a5);
        std::complex<float> b6 = std::sqrt(a6);
        std::complex<float> b7 = std::sqrt(a7);
        return _mm512_setr_ps(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag(),
                              b4.real(), b4.imag(),
                              b5.real(), b5.imag(),
                              b6.real(), b6.imag(),
                              b7.real(), b7.imag());
    }
};

template <>
struct vector_traits<std::complex<double>>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 64;
    typedef __m512d vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512d v)
    {
        // ( -, 3, -, 2, -, 1, -, 0)
        __m512 lo = _mm512_castps256_ps512(_mm512_cvtpd_ps(convert<double>(v)));
        // ( -, -, -, -, -, -, -, -, 3, 2, 1, 0, 3, 2, 1, 0)
        return _mm512_shuffle_f32x4(lo, lo, _MM_PERM_AAAA);
        // ( 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512d v)
    {
        // ( -, 3, -, 2, -, 1, -, 0)
        __m512d tmp = _mm512_permutex_pd(v, _MM_PERM_DBCA);
        // ( 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_shuffle_f64x2(tmp, tmp, _MM_PERM_DBCA);
        // ( 3, 2, 1, 0, 3, 2, 1, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512d v)
    {
        // ( -, 3, -, 2, -, 1, -, 0)
        __m512 lo = _mm512_castps256_ps512(_mm512_cvtpd_ps(v));
        // ( -, -, -, -, -, -, -, -, -, 3, -, 2, -, 1, -, 0)
        return _mm512_shuffle_f32x4(lo, lo, _MM_PERM_BABA);
        // ( -, 3, -, 2, -, 1, -, 0, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512d v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m512i>
    convert(__m512d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(_mm512_castpd512_pd256(convert<double>(v)));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        __m512i i8 = _mm512_castsi128_si512(_mm_packs_epi16(i16, i16));
        return _mm512_shuffle_i32x4(i8, i8, _MM_PERM_AAAA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(_mm512_castpd512_pd256(convert<double>(v)));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        __m512i i8 = _mm512_castsi128_si512(_mm_packus_epi16(i16, i16));
        return _mm512_shuffle_i32x4(i8, i8, _MM_PERM_AAAA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m512i>
    convert(__m512d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(_mm512_castpd512_pd256(convert<double>(v)));
        __m512i i16 = _mm512_castsi128_si512(_mm_packs_epi32(i32, i32));
        return _mm512_shuffle_i32x4(i16, i16, _MM_PERM_AAAA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(_mm512_castpd512_pd256(convert<double>(v)));
        __m512i i16 = _mm512_castsi128_si512(_mm_packus_epi32(i32, i32));
        return _mm512_shuffle_i32x4(i16, i16, _MM_PERM_AAAA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m512i>
    convert(__m512d v)
    {
        __m512i lo = _mm512_castsi256_si512(_mm512_cvtpd_epi32(convert<double>(v)));
        return _mm512_shuffle_i32x4(lo, lo, _MM_PERM_AAAA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512d v)
    {
        return _mm512_setr_epi32((uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512d v)
    {
        return _mm512_setr_epi64((T)v[0], (T)v[2],
                                 (T)v[4], (T)v[6],
                                 (T)v[0], (T)v[2],
                                 (T)v[4], (T)v[6]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512d>
    load(const std::complex<double>* ptr)
    {
        return _mm512_loadu_pd((double*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512d>
    load(const std::complex<double>* ptr)
    {
        return _mm512_load_pd((double*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m512d>
    load(const std::complex<double>* ptr)
    {
        return _mm512_broadcast_f64x4(_mm256_loadu_pd((double*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m512d>
    load(const std::complex<double>* ptr)
    {
        return _mm512_broadcast_f64x4(_mm256_load_pd((double*)ptr));
    }

    static __m512d load1(const std::complex<double>* ptr)
    {
        return _mm512_broadcast_f64x2(_mm_loadu_pd((double*)ptr));
    }

    static __m512d set1(std::complex<double> val)
    {
        return _mm512_setr_pd(val.real(), val.imag(), val.real(), val.imag(),
                              val.real(), val.imag(), val.real(), val.imag());
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512d v, std::complex<double>* ptr)
    {
        _mm512_storeu_pd((double*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512d v, std::complex<double>* ptr)
    {
        _mm512_store_pd((double*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m512d v, std::complex<double>* ptr)
    {
        _mm256_storeu_pd((double*)ptr, _mm512_castpd512_pd256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m512d v, std::complex<double>* ptr)
    {
        _mm256_store_pd((double*)ptr, _mm512_castpd512_pd256(v));
    }

    static __m512d add(__m512d a, __m512d b)
    {
        return _mm512_add_pd(a, b);
    }

    static __m512d sub(__m512d a, __m512d b)
    {
        return _mm512_sub_pd(a, b);
    }

    static __m512d mul(__m512d a, __m512d b)
    {
        __m512d ashuf = _mm512_shuffle_pd(a, a, 0x55);
        __m512d breal = _mm512_shuffle_pd(b, b, 0x00);
        __m512d bimag = _mm512_shuffle_pd(b, b, 0xff);
        __m512d tmp = _mm512_mul_pd(ashuf, bimag); // tmp = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm512_fmaddsub_pd(a, breal, tmp);  //       (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
    }

    static __m512d div(__m512d a, __m512d b)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> a2((double)a[4], (double)a[5]);
        std::complex<double> a3((double)a[6], (double)a[7]);
        std::complex<double> b0((double)b[0], (double)b[1]);
        std::complex<double> b1((double)b[2], (double)b[3]);
        std::complex<double> b2((double)b[4], (double)b[5]);
        std::complex<double> b3((double)b[6], (double)b[7]);
        std::complex<double> c0 = a0 / b0;
        std::complex<double> c1 = a1 / b1;
        std::complex<double> c2 = a2 / b2;
        std::complex<double> c3 = a3 / b3;
        return _mm512_setr_pd(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag());
    }

    static __m512d pow(__m512d a, __m512d b)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> a2((double)a[4], (double)a[5]);
        std::complex<double> a3((double)a[6], (double)a[7]);
        std::complex<double> b0((double)b[0], (double)b[1]);
        std::complex<double> b1((double)b[2], (double)b[3]);
        std::complex<double> b2((double)b[4], (double)b[5]);
        std::complex<double> b3((double)b[6], (double)b[7]);
        std::complex<double> c0 = std::pow(a0, b0);
        std::complex<double> c1 = std::pow(a1, b1);
        std::complex<double> c2 = std::pow(a2, b2);
        std::complex<double> c3 = std::pow(a3, b3);
        return _mm512_setr_pd(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag());
    }

    static __m512d negate(__m512d a)
    {
        return _mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(a),
                                                    _mm512_castpd_si512(_mm512_set1_pd(-0.0))));
    }

    static __m512d exp(__m512d a)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> a2((double)a[4], (double)a[5]);
        std::complex<double> a3((double)a[6], (double)a[7]);
        std::complex<double> b0 = std::exp(a0);
        std::complex<double> b1 = std::exp(a1);
        std::complex<double> b2 = std::exp(a2);
        std::complex<double> b3 = std::exp(a3);
        return _mm512_setr_pd(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag());
    }

    static __m512d sqrt(__m512d a)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> a2((double)a[4], (double)a[5]);
        std::complex<double> a3((double)a[6], (double)a[7]);
        std::complex<double> b0 = std::sqrt(a0);
        std::complex<double> b1 = std::sqrt(a1);
        std::complex<double> b2 = std::sqrt(a2);
        std::complex<double> b3 = std::sqrt(a3);
        return _mm512_setr_pd(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag());
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int8_t>::value ||
                                            std::is_same<U,uint8_t>::value>>
{
    static constexpr int vector_width = 64;
    static constexpr size_t alignment = 64;
    typedef __m512i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_pd(_mm512_castsi512_si256(convert<int32_t>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512i v)
    {
        // (...,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m128i mask = _mm_set_epi8(7,6,7,6,5,4,5,4,3,2,3,2,1,0,1,0);
        __m128i dup = _mm_shuffle_epi8(_mm512_castsi512_si128(v), mask);
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_unpacklo_ps(convert<float>(_mm512_castsi128_si512(dup)), _mm512_setzero_ps());
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512i v)
    {
        // (...,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m128i mask = _mm_set_epi8(3,3,2,2,1,1,0,0,3,3,2,2,1,1,0,0);
        __m128i dup = _mm_shuffle_epi8(_mm512_castsi512_si128(v), mask);
        // ( 3, 3, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 1, 1, 0, 0)
        return _mm512_unpacklo_pd(convert<double>(_mm512_castsi128_si512(dup)), _mm512_setzero_pd());
        // ( -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512i v)
    {
        __m128i lo8 = _mm512_extracti32x4_epi32(v, 0);
        __m128i hi8 = _mm512_extracti32x4_epi32(v, 1);
        __m256i lo16 = std::is_signed<U>::value ? _mm256_cvtepi8_epi16(lo8)
                                                : _mm256_cvtepu8_epi16(lo8);
        __m256i hi16 = std::is_signed<U>::value ? _mm256_cvtepi8_epi16(hi8)
                                                : _mm256_cvtepu8_epi16(hi8);
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo16), hi16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512i v)
    {
        return std::is_signed<U>::value ? _mm512_cvtepi8_epi32(_mm512_castsi512_si128(v))
                                        : _mm512_cvtepu8_epi32(_mm512_castsi512_si128(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512i v)
    {
        return std::is_signed<U>::value ? _mm512_cvtepi8_epi64(_mm512_castsi512_si128(v))
                                        : _mm512_cvtepu8_epi64(_mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 64 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_loadu_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 64 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_load_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_castps_si512(_mm512_broadcast_f32x4(_mm_loadu_ps((float*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_castps_si512(_mm512_broadcast_f32x4(_mm_load_ps((float*)ptr)));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi64(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi32(*(int32_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi16(*(int16_t*)ptr);
    }

    static __m512i load1(const U* ptr)
    {
        return _mm512_set1_epi8(*ptr);
    }

    static __m512i set1(U val)
    {
        return _mm512_set1_epi8(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 64 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_storeu_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 64 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_store_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8>
    store(__m512i v, U* ptr)
    {
        *(int64_t*)ptr = _mm_extract_epi64(_mm512_castsi512_si128(v), 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m512i v, U* ptr)
    {
        *(int32_t*)ptr = _mm_extract_epi32(_mm512_castsi512_si128(v), 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m512i v, U* ptr)
    {
        *(int16_t*)ptr = _mm_extract_epi16(_mm512_castsi512_si128(v), 0);
    }

    static __m512i add(__m512i a, __m512i b)
    {
        __m256i alo = _mm512_extracti64x4_epi64(a, 0);
        __m256i ahi = _mm512_extracti64x4_epi64(a, 1);
        __m256i blo = _mm512_extracti64x4_epi64(b, 0);
        __m256i bhi = _mm512_extracti64x4_epi64(b, 1);
        __m256i clo = _mm256_add_epi8(alo, blo);
        __m256i chi = _mm256_add_epi8(ahi, bhi);
        return _mm512_inserti64x4(_mm512_castsi256_si512(clo), chi, 1);
    }

    static __m512i sub(__m512i a, __m512i b)
    {
        __m256i alo = _mm512_extracti64x4_epi64(a, 0);
        __m256i ahi = _mm512_extracti64x4_epi64(a, 1);
        __m256i blo = _mm512_extracti64x4_epi64(b, 0);
        __m256i bhi = _mm512_extracti64x4_epi64(b, 1);
        __m256i clo = _mm256_sub_epi8(alo, blo);
        __m256i chi = _mm256_sub_epi8(ahi, bhi);
        return _mm512_inserti64x4(_mm512_castsi256_si512(clo), chi, 1);
    }

    static __m512i mul(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i lo0 = _mm256_and_si256(_mm256_mullo_epi16(a0, b0), _mm256_set1_epi16(0xff));
        __m256i lo1 = _mm256_and_si256(_mm256_mullo_epi16(a1, b1), _mm256_set1_epi16(0xff));
        __m256i hi0 = _mm256_mullo_epi16(_mm256_srli_epi16(a0, 8), _mm256_srli_epi16(b0, 8));
        __m256i hi1 = _mm256_mullo_epi16(_mm256_srli_epi16(a1, 8), _mm256_srli_epi16(b1, 8));
        __m256i ab0 = _mm256_or_si256(_mm256_slli_epi16(hi0, 8), lo0);
        __m256i ab1 = _mm256_or_si256(_mm256_slli_epi16(hi1, 8), lo1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(ab0), ab1, 1);
    }

    static __m512i div(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi8((U)_mm256_extract_epi8(a0, 0) / (U)_mm256_extract_epi8(b0, 0),
                                      (U)_mm256_extract_epi8(a0, 1) / (U)_mm256_extract_epi8(b0, 1),
                                      (U)_mm256_extract_epi8(a0, 2) / (U)_mm256_extract_epi8(b0, 2),
                                      (U)_mm256_extract_epi8(a0, 3) / (U)_mm256_extract_epi8(b0, 3),
                                      (U)_mm256_extract_epi8(a0, 4) / (U)_mm256_extract_epi8(b0, 4),
                                      (U)_mm256_extract_epi8(a0, 5) / (U)_mm256_extract_epi8(b0, 5),
                                      (U)_mm256_extract_epi8(a0, 6) / (U)_mm256_extract_epi8(b0, 6),
                                      (U)_mm256_extract_epi8(a0, 7) / (U)_mm256_extract_epi8(b0, 7),
                                      (U)_mm256_extract_epi8(a0, 8) / (U)_mm256_extract_epi8(b0, 8),
                                      (U)_mm256_extract_epi8(a0, 9) / (U)_mm256_extract_epi8(b0, 9),
                                      (U)_mm256_extract_epi8(a0,10) / (U)_mm256_extract_epi8(b0,10),
                                      (U)_mm256_extract_epi8(a0,11) / (U)_mm256_extract_epi8(b0,11),
                                      (U)_mm256_extract_epi8(a0,12) / (U)_mm256_extract_epi8(b0,12),
                                      (U)_mm256_extract_epi8(a0,13) / (U)_mm256_extract_epi8(b0,13),
                                      (U)_mm256_extract_epi8(a0,14) / (U)_mm256_extract_epi8(b0,14),
                                      (U)_mm256_extract_epi8(a0,15) / (U)_mm256_extract_epi8(b0,15),
                                      (U)_mm256_extract_epi8(a0,16) / (U)_mm256_extract_epi8(b0,16),
                                      (U)_mm256_extract_epi8(a0,17) / (U)_mm256_extract_epi8(b0,17),
                                      (U)_mm256_extract_epi8(a0,18) / (U)_mm256_extract_epi8(b0,18),
                                      (U)_mm256_extract_epi8(a0,19) / (U)_mm256_extract_epi8(b0,19),
                                      (U)_mm256_extract_epi8(a0,20) / (U)_mm256_extract_epi8(b0,20),
                                      (U)_mm256_extract_epi8(a0,21) / (U)_mm256_extract_epi8(b0,21),
                                      (U)_mm256_extract_epi8(a0,22) / (U)_mm256_extract_epi8(b0,22),
                                      (U)_mm256_extract_epi8(a0,23) / (U)_mm256_extract_epi8(b0,23),
                                      (U)_mm256_extract_epi8(a0,24) / (U)_mm256_extract_epi8(b0,24),
                                      (U)_mm256_extract_epi8(a0,25) / (U)_mm256_extract_epi8(b0,25),
                                      (U)_mm256_extract_epi8(a0,26) / (U)_mm256_extract_epi8(b0,26),
                                      (U)_mm256_extract_epi8(a0,27) / (U)_mm256_extract_epi8(b0,27),
                                      (U)_mm256_extract_epi8(a0,28) / (U)_mm256_extract_epi8(b0,28),
                                      (U)_mm256_extract_epi8(a0,29) / (U)_mm256_extract_epi8(b0,29),
                                      (U)_mm256_extract_epi8(a0,30) / (U)_mm256_extract_epi8(b0,30),
                                      (U)_mm256_extract_epi8(a0,31) / (U)_mm256_extract_epi8(b0,31));
        __m256i c1 = _mm256_setr_epi8((U)_mm256_extract_epi8(a1, 0) / (U)_mm256_extract_epi8(b1, 0),
                                      (U)_mm256_extract_epi8(a1, 1) / (U)_mm256_extract_epi8(b1, 1),
                                      (U)_mm256_extract_epi8(a1, 2) / (U)_mm256_extract_epi8(b1, 2),
                                      (U)_mm256_extract_epi8(a1, 3) / (U)_mm256_extract_epi8(b1, 3),
                                      (U)_mm256_extract_epi8(a1, 4) / (U)_mm256_extract_epi8(b1, 4),
                                      (U)_mm256_extract_epi8(a1, 5) / (U)_mm256_extract_epi8(b1, 5),
                                      (U)_mm256_extract_epi8(a1, 6) / (U)_mm256_extract_epi8(b1, 6),
                                      (U)_mm256_extract_epi8(a1, 7) / (U)_mm256_extract_epi8(b1, 7),
                                      (U)_mm256_extract_epi8(a1, 8) / (U)_mm256_extract_epi8(b1, 8),
                                      (U)_mm256_extract_epi8(a1, 9) / (U)_mm256_extract_epi8(b1, 9),
                                      (U)_mm256_extract_epi8(a1,10) / (U)_mm256_extract_epi8(b1,10),
                                      (U)_mm256_extract_epi8(a1,11) / (U)_mm256_extract_epi8(b1,11),
                                      (U)_mm256_extract_epi8(a1,12) / (U)_mm256_extract_epi8(b1,12),
                                      (U)_mm256_extract_epi8(a1,13) / (U)_mm256_extract_epi8(b1,13),
                                      (U)_mm256_extract_epi8(a1,14) / (U)_mm256_extract_epi8(b1,14),
                                      (U)_mm256_extract_epi8(a1,15) / (U)_mm256_extract_epi8(b1,15),
                                      (U)_mm256_extract_epi8(a1,16) / (U)_mm256_extract_epi8(b1,16),
                                      (U)_mm256_extract_epi8(a1,17) / (U)_mm256_extract_epi8(b1,17),
                                      (U)_mm256_extract_epi8(a1,18) / (U)_mm256_extract_epi8(b1,18),
                                      (U)_mm256_extract_epi8(a1,19) / (U)_mm256_extract_epi8(b1,19),
                                      (U)_mm256_extract_epi8(a1,20) / (U)_mm256_extract_epi8(b1,20),
                                      (U)_mm256_extract_epi8(a1,21) / (U)_mm256_extract_epi8(b1,21),
                                      (U)_mm256_extract_epi8(a1,22) / (U)_mm256_extract_epi8(b1,22),
                                      (U)_mm256_extract_epi8(a1,23) / (U)_mm256_extract_epi8(b1,23),
                                      (U)_mm256_extract_epi8(a1,24) / (U)_mm256_extract_epi8(b1,24),
                                      (U)_mm256_extract_epi8(a1,25) / (U)_mm256_extract_epi8(b1,25),
                                      (U)_mm256_extract_epi8(a1,26) / (U)_mm256_extract_epi8(b1,26),
                                      (U)_mm256_extract_epi8(a1,27) / (U)_mm256_extract_epi8(b1,27),
                                      (U)_mm256_extract_epi8(a1,28) / (U)_mm256_extract_epi8(b1,28),
                                      (U)_mm256_extract_epi8(a1,29) / (U)_mm256_extract_epi8(b1,29),
                                      (U)_mm256_extract_epi8(a1,30) / (U)_mm256_extract_epi8(b1,30),
                                      (U)_mm256_extract_epi8(a1,31) / (U)_mm256_extract_epi8(b1,31));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i pow(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi8((U)std::pow((U)_mm256_extract_epi8(a0, 0), (U)_mm256_extract_epi8(b0, 0)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 1), (U)_mm256_extract_epi8(b0, 1)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 2), (U)_mm256_extract_epi8(b0, 2)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 3), (U)_mm256_extract_epi8(b0, 3)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 4), (U)_mm256_extract_epi8(b0, 4)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 5), (U)_mm256_extract_epi8(b0, 5)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 6), (U)_mm256_extract_epi8(b0, 6)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 7), (U)_mm256_extract_epi8(b0, 7)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 8), (U)_mm256_extract_epi8(b0, 8)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0, 9), (U)_mm256_extract_epi8(b0, 9)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,10), (U)_mm256_extract_epi8(b0,10)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,11), (U)_mm256_extract_epi8(b0,11)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,12), (U)_mm256_extract_epi8(b0,12)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,13), (U)_mm256_extract_epi8(b0,13)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,14), (U)_mm256_extract_epi8(b0,14)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,15), (U)_mm256_extract_epi8(b0,15)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,16), (U)_mm256_extract_epi8(b0,16)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,17), (U)_mm256_extract_epi8(b0,17)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,18), (U)_mm256_extract_epi8(b0,18)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,19), (U)_mm256_extract_epi8(b0,19)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,20), (U)_mm256_extract_epi8(b0,20)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,21), (U)_mm256_extract_epi8(b0,21)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,22), (U)_mm256_extract_epi8(b0,22)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,23), (U)_mm256_extract_epi8(b0,23)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,24), (U)_mm256_extract_epi8(b0,24)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,25), (U)_mm256_extract_epi8(b0,25)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,26), (U)_mm256_extract_epi8(b0,26)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,27), (U)_mm256_extract_epi8(b0,27)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,28), (U)_mm256_extract_epi8(b0,28)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,29), (U)_mm256_extract_epi8(b0,29)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,30), (U)_mm256_extract_epi8(b0,30)),
                                      (U)std::pow((U)_mm256_extract_epi8(a0,31), (U)_mm256_extract_epi8(b0,31)));
        __m256i c1 = _mm256_setr_epi8((U)std::pow((U)_mm256_extract_epi8(a1, 0), (U)_mm256_extract_epi8(b1, 0)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 1), (U)_mm256_extract_epi8(b1, 1)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 2), (U)_mm256_extract_epi8(b1, 2)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 3), (U)_mm256_extract_epi8(b1, 3)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 4), (U)_mm256_extract_epi8(b1, 4)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 5), (U)_mm256_extract_epi8(b1, 5)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 6), (U)_mm256_extract_epi8(b1, 6)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 7), (U)_mm256_extract_epi8(b1, 7)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 8), (U)_mm256_extract_epi8(b1, 8)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1, 9), (U)_mm256_extract_epi8(b1, 9)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,10), (U)_mm256_extract_epi8(b1,10)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,11), (U)_mm256_extract_epi8(b1,11)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,12), (U)_mm256_extract_epi8(b1,12)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,13), (U)_mm256_extract_epi8(b1,13)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,14), (U)_mm256_extract_epi8(b1,14)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,15), (U)_mm256_extract_epi8(b1,15)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,16), (U)_mm256_extract_epi8(b1,16)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,17), (U)_mm256_extract_epi8(b1,17)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,18), (U)_mm256_extract_epi8(b1,18)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,19), (U)_mm256_extract_epi8(b1,19)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,20), (U)_mm256_extract_epi8(b1,20)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,21), (U)_mm256_extract_epi8(b1,21)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,22), (U)_mm256_extract_epi8(b1,22)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,23), (U)_mm256_extract_epi8(b1,23)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,24), (U)_mm256_extract_epi8(b1,24)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,25), (U)_mm256_extract_epi8(b1,25)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,26), (U)_mm256_extract_epi8(b1,26)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,27), (U)_mm256_extract_epi8(b1,27)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,28), (U)_mm256_extract_epi8(b1,28)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,29), (U)_mm256_extract_epi8(b1,29)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,30), (U)_mm256_extract_epi8(b1,30)),
                                      (U)std::pow((U)_mm256_extract_epi8(a1,31), (U)_mm256_extract_epi8(b1,31)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i negate(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i na0 = _mm256_sub_epi8(_mm256_setzero_si256(), a0);
        __m256i na1 = _mm256_sub_epi8(_mm256_setzero_si256(), a1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(na0), na1, 1);
    }

    static __m512i exp(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i c0 = _mm256_setr_epi8((U)std::exp((U)_mm256_extract_epi8(a0, 0)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 1)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 2)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 3)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 4)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 5)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 6)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 7)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 8)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0, 9)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,10)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,11)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,12)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,13)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,14)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,15)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,16)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,17)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,18)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,19)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,20)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,21)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,22)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,23)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,24)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,25)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,26)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,27)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,28)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,29)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,30)),
                                      (U)std::exp((U)_mm256_extract_epi8(a0,31)));
        __m256i c1 = _mm256_setr_epi8((U)std::exp((U)_mm256_extract_epi8(a1, 0)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 1)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 2)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 3)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 4)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 5)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 6)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 7)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 8)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1, 9)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,10)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,11)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,12)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,13)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,14)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,15)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,16)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,17)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,18)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,19)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,20)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,21)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,22)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,23)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,24)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,25)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,26)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,27)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,28)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,29)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,30)),
                                      (U)std::exp((U)_mm256_extract_epi8(a1,31)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i sqrt(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i c0 = _mm256_setr_epi8((U)std::sqrt((U)_mm256_extract_epi8(a0, 0)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 1)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 2)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 3)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 4)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 5)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 6)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 7)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 8)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0, 9)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,10)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,11)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,12)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,13)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,14)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,15)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,16)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,17)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,18)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,19)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,20)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,21)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,22)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,23)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,24)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,25)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,26)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,27)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,28)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,29)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,30)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a0,31)));
        __m256i c1 = _mm256_setr_epi8((U)std::sqrt((U)_mm256_extract_epi8(a1, 0)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 1)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 2)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 3)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 4)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 5)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 6)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 7)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 8)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1, 9)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,10)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,11)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,12)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,13)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,14)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,15)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,16)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,17)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,18)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,19)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,20)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,21)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,22)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,23)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,24)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,25)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,26)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,27)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,28)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,29)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,30)),
                                      (U)std::sqrt((U)_mm256_extract_epi8(a1,31)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int16_t>::value ||
                                            std::is_same<U,uint16_t>::value>>
{
    static constexpr int vector_width = 32;
    static constexpr size_t alignment = 64;
    typedef __m512i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_pd(_mm512_castsi512_si256(convert<int32_t>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512i v)
    {
        // (...,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m256i tmp1 = _mm256_permute4x64_epi64(_mm512_castsi512_si256(v), _MM_PERM_BBAA);
        // ( 7, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0)
        __m256i tmp2 = _mm256_shuffle_epi32(tmp1, _MM_PERM_BBAA);
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_unpacklo_ps(convert<float>(_mm512_castsi256_si512(tmp2)), _mm512_setzero_ps());
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512i v)
    {
        // (..., 7, 6, 5, 4, 3, 2, 1, 0)
        __m128i tmp1 = _mm_shuffle_epi32(_mm512_castsi512_si128(v), _MM_PERM_BBAA);
        // ( 3, 2, 3, 2, 1, 0, 1, 0)
        __m256i tmp2 = _mm256_shuffle_epi32(_mm256_cvtepi16_epi32(tmp1), _MM_PERM_BBAA);
        // ( 3, 3, 2, 2, 1, 1, 0, 0)
        return _mm512_unpacklo_pd(_mm512_cvtepi32_pd(tmp2), _mm512_setzero_pd());
        // ( -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo16 = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi16 = _mm512_extracti64x4_epi64(v, 1);
        __m256i tmp = std::is_signed<U>::value ? _mm256_packs_epi16(lo16, hi16)
                                              : _mm256_packus_epi16(lo16, hi16);
        __m512i lo = _mm512_castsi256_si512(_mm256_permute4x64_epi64(tmp, _MM_PERM_DBCA));
        return _mm512_shuffle_i64x2(lo, lo, _MM_PERM_BABA);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512i v)
    {
        return std::is_signed<U>::value ? _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v))
                                        : _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512i v)
    {
        return std::is_signed<U>::value ? _mm512_cvtepi16_epi64(_mm512_castsi512_si128(v))
                                        : _mm512_cvtepu16_epi64(_mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_loadu_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_load_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_load_si128((__m128i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi64(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi32(*(int32_t*)ptr);
    }

    static __m512i load1(const U* ptr)
    {
        return _mm512_set1_epi16(*ptr);
    }

    static __m512i set1(U val)
    {
        return _mm512_set1_epi16(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_storeu_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_store_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m512i v, U* ptr)
    {
        *(int64_t*)ptr = _mm_extract_epi64(_mm512_castsi512_si128(v), 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m512i v, U* ptr)
    {
        *(int32_t*)ptr = _mm_extract_epi32(_mm512_castsi512_si128(v), 0);
    }

    static __m512i add(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i ab0 = _mm256_add_epi16(a0, b0);
        __m256i ab1 = _mm256_add_epi16(a1, b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(ab0), ab1, 1);
    }

    static __m512i sub(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i ab0 = _mm256_sub_epi16(a0, b0);
        __m256i ab1 = _mm256_sub_epi16(a1, b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(ab0), ab1, 1);
    }

    static __m512i mul(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i ab0 = _mm256_mullo_epi16(a0, b0);
        __m256i ab1 = _mm256_mullo_epi16(a1, b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(ab0), ab1, 1);
    }

    static __m512i div(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi16((U)_mm256_extract_epi16(a0, 0) / (U)_mm256_extract_epi16(b0, 0),
                                       (U)_mm256_extract_epi16(a0, 1) / (U)_mm256_extract_epi16(b0, 1),
                                       (U)_mm256_extract_epi16(a0, 2) / (U)_mm256_extract_epi16(b0, 2),
                                       (U)_mm256_extract_epi16(a0, 3) / (U)_mm256_extract_epi16(b0, 3),
                                       (U)_mm256_extract_epi16(a0, 4) / (U)_mm256_extract_epi16(b0, 4),
                                       (U)_mm256_extract_epi16(a0, 5) / (U)_mm256_extract_epi16(b0, 5),
                                       (U)_mm256_extract_epi16(a0, 6) / (U)_mm256_extract_epi16(b0, 6),
                                       (U)_mm256_extract_epi16(a0, 7) / (U)_mm256_extract_epi16(b0, 7),
                                       (U)_mm256_extract_epi16(a0, 8) / (U)_mm256_extract_epi16(b0, 8),
                                       (U)_mm256_extract_epi16(a0, 9) / (U)_mm256_extract_epi16(b0, 9),
                                       (U)_mm256_extract_epi16(a0,10) / (U)_mm256_extract_epi16(b0,10),
                                       (U)_mm256_extract_epi16(a0,11) / (U)_mm256_extract_epi16(b0,11),
                                       (U)_mm256_extract_epi16(a0,12) / (U)_mm256_extract_epi16(b0,12),
                                       (U)_mm256_extract_epi16(a0,13) / (U)_mm256_extract_epi16(b0,13),
                                       (U)_mm256_extract_epi16(a0,14) / (U)_mm256_extract_epi16(b0,14),
                                       (U)_mm256_extract_epi16(a0,15) / (U)_mm256_extract_epi16(b0,15));
        __m256i c1 = _mm256_setr_epi16((U)_mm256_extract_epi16(a1, 0) / (U)_mm256_extract_epi16(b1, 0),
                                       (U)_mm256_extract_epi16(a1, 1) / (U)_mm256_extract_epi16(b1, 1),
                                       (U)_mm256_extract_epi16(a1, 2) / (U)_mm256_extract_epi16(b1, 2),
                                       (U)_mm256_extract_epi16(a1, 3) / (U)_mm256_extract_epi16(b1, 3),
                                       (U)_mm256_extract_epi16(a1, 4) / (U)_mm256_extract_epi16(b1, 4),
                                       (U)_mm256_extract_epi16(a1, 5) / (U)_mm256_extract_epi16(b1, 5),
                                       (U)_mm256_extract_epi16(a1, 6) / (U)_mm256_extract_epi16(b1, 6),
                                       (U)_mm256_extract_epi16(a1, 7) / (U)_mm256_extract_epi16(b1, 7),
                                       (U)_mm256_extract_epi16(a1, 8) / (U)_mm256_extract_epi16(b1, 8),
                                       (U)_mm256_extract_epi16(a1, 9) / (U)_mm256_extract_epi16(b1, 9),
                                       (U)_mm256_extract_epi16(a1,10) / (U)_mm256_extract_epi16(b1,10),
                                       (U)_mm256_extract_epi16(a1,11) / (U)_mm256_extract_epi16(b1,11),
                                       (U)_mm256_extract_epi16(a1,12) / (U)_mm256_extract_epi16(b1,12),
                                       (U)_mm256_extract_epi16(a1,13) / (U)_mm256_extract_epi16(b1,13),
                                       (U)_mm256_extract_epi16(a1,14) / (U)_mm256_extract_epi16(b1,14),
                                       (U)_mm256_extract_epi16(a1,15) / (U)_mm256_extract_epi16(b1,15));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i pow(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi16((U)std::pow((U)_mm256_extract_epi16(a0, 0), (U)_mm256_extract_epi16(b0, 0)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 1), (U)_mm256_extract_epi16(b0, 1)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 2), (U)_mm256_extract_epi16(b0, 2)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 3), (U)_mm256_extract_epi16(b0, 3)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 4), (U)_mm256_extract_epi16(b0, 4)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 5), (U)_mm256_extract_epi16(b0, 5)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 6), (U)_mm256_extract_epi16(b0, 6)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 7), (U)_mm256_extract_epi16(b0, 7)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 8), (U)_mm256_extract_epi16(b0, 8)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0, 9), (U)_mm256_extract_epi16(b0, 9)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,10), (U)_mm256_extract_epi16(b0,10)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,11), (U)_mm256_extract_epi16(b0,11)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,12), (U)_mm256_extract_epi16(b0,12)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,13), (U)_mm256_extract_epi16(b0,13)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,14), (U)_mm256_extract_epi16(b0,14)),
                                       (U)std::pow((U)_mm256_extract_epi16(a0,15), (U)_mm256_extract_epi16(b0,15)));
        __m256i c1 = _mm256_setr_epi16((U)std::pow((U)_mm256_extract_epi16(a1, 0), (U)_mm256_extract_epi16(b1, 0)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 1), (U)_mm256_extract_epi16(b1, 1)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 2), (U)_mm256_extract_epi16(b1, 2)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 3), (U)_mm256_extract_epi16(b1, 3)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 4), (U)_mm256_extract_epi16(b1, 4)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 5), (U)_mm256_extract_epi16(b1, 5)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 6), (U)_mm256_extract_epi16(b1, 6)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 7), (U)_mm256_extract_epi16(b1, 7)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 8), (U)_mm256_extract_epi16(b1, 8)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1, 9), (U)_mm256_extract_epi16(b1, 9)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,10), (U)_mm256_extract_epi16(b1,10)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,11), (U)_mm256_extract_epi16(b1,11)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,12), (U)_mm256_extract_epi16(b1,12)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,13), (U)_mm256_extract_epi16(b1,13)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,14), (U)_mm256_extract_epi16(b1,14)),
                                       (U)std::pow((U)_mm256_extract_epi16(a1,15), (U)_mm256_extract_epi16(b1,15)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i negate(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i na0 = _mm256_sub_epi16(_mm256_setzero_si256(), a0);
        __m256i na1 = _mm256_sub_epi16(_mm256_setzero_si256(), a1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(na0), na1, 1);
    }

    static __m512i exp(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm256_setr_epi16((U)std::exp((U)_mm256_extract_epi16(a0, 0)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 1)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 2)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 3)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 4)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 5)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 6)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 7)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 8)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0, 9)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,10)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,11)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,12)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,13)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,14)),
                                       (U)std::exp((U)_mm256_extract_epi16(a0,15)));
        __m256i b1 = _mm256_setr_epi16((U)std::exp((U)_mm256_extract_epi16(a1, 0)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 1)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 2)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 3)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 4)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 5)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 6)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 7)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 8)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1, 9)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,10)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,11)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,12)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,13)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,14)),
                                       (U)std::exp((U)_mm256_extract_epi16(a1,15)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1);
    }

    static __m512i sqrt(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm256_setr_epi16((U)std::sqrt((U)_mm256_extract_epi16(a0, 0)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 1)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 2)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 3)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 4)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 5)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 6)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 7)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 8)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0, 9)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,10)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,11)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,12)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,13)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,14)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a0,15)));
        __m256i b1 = _mm256_setr_epi16((U)std::sqrt((U)_mm256_extract_epi16(a1, 0)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 1)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 2)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 3)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 4)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 5)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 6)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 7)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 8)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1, 9)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,10)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,11)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,12)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,13)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,14)),
                                       (U)std::sqrt((U)_mm256_extract_epi16(a1,15)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1);
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int32_t>::value ||
                                            std::is_same<U,uint32_t>::value>>
{
    static constexpr int vector_width = 16;
    static constexpr size_t alignment = 64;
    typedef __m512i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_ps(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512i v)
    {
        return _mm512_cvtepi32_pd(_mm512_castsi512_si256(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512i v)
    {
        // (15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        __m512i tmp1 = _mm512_shuffle_i32x4(v, v, _MM_PERM_BBAA);
        // ( 7, 6, 5, 4, 7, 6, 5, 4, 3, 2, 1, 0, 3, 2, 1, 0)
        __m512i tmp2 = _mm512_permutex_epi64(tmp1, _MM_PERM_BBAA);
        // ( 7, 6, 7, 6, 5, 4, 5, 4, 3, 2, 3, 2, 1, 0, 1, 0)
        return _mm512_unpacklo_ps(convert<float>(tmp2), _mm512_setzero_ps());
        // ( -, 7, -, 6, -, 5, -, 4, -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512i v)
    {
        // (..., 7, 6, 5, 4, 3, 2, 1, 0)
        __m512i tmp1 = _mm512_permutex_epi64(v, _MM_PERM_BBAA);
        // ( 3, 2, 3, 2, 1, 0, 1, 0)
        __m512i tmp2 = _mm512_shuffle_epi32(tmp1, _MM_PERM_BBAA);
        // ( 3, 3, 2, 2, 1, 1, 0, 0)
        return _mm512_unpacklo_pd(convert<double>(tmp2), _mm512_setzero_pd());
        // ( -, 3, -, 2, -, 1, -, 0)
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo32 = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi32 = _mm512_extracti64x4_epi64(v, 1);
        __m256i i16 = std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi32(lo32, hi32), _MM_PERM_DBCA)
                                               : _mm256_permute4x64_epi64(_mm256_packus_epi32(lo32, hi32), _MM_PERM_DBCA);
        __m256i lo = std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi16(i16, i16), _MM_PERM_DBCA)
                                              : _mm256_permute4x64_epi64(_mm256_packus_epi16(i16, i16), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), lo, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo32 = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi32 = _mm512_extracti64x4_epi64(v, 1);
        __m256i i16 = std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi32(lo32, hi32), _MM_PERM_DBCA)
                                               : _mm256_permute4x64_epi64(_mm256_packus_epi32(lo32, hi32), _MM_PERM_DBCA);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512i v)
    {
        return std::is_signed<U>::value ? _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v))
                                        : _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_loadu_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_load_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_load_si128((__m128i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m512i>
    load(const U* ptr)
    {
        return _mm512_set1_epi64(*(int64_t*)ptr);
    }

    static __m512i load1(const U* ptr)
    {
        return _mm512_set1_epi32(*ptr);
    }

    static __m512i set1(U val)
    {
        return _mm512_set1_epi32(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_storeu_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_store_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m512i v, U* ptr)
    {
        *(int64_t*)ptr = _mm_extract_epi64(_mm512_castsi512_si128(v), 0);
    }

    static __m512i add(__m512i a, __m512i b)
    {
        return _mm512_add_epi32(a, b);
    }

    static __m512i sub(__m512i a, __m512i b)
    {
        return _mm512_sub_epi32(a, b);
    }

    static __m512i mul(__m512i a, __m512i b)
    {
        return _mm512_mullo_epi32(a, b);
    }

    static __m512i div(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi32((U)_mm256_extract_epi32(a0, 0) / (U)_mm256_extract_epi32(b0, 0),
                                       (U)_mm256_extract_epi32(a0, 1) / (U)_mm256_extract_epi32(b0, 1),
                                       (U)_mm256_extract_epi32(a0, 2) / (U)_mm256_extract_epi32(b0, 2),
                                       (U)_mm256_extract_epi32(a0, 3) / (U)_mm256_extract_epi32(b0, 3),
                                       (U)_mm256_extract_epi32(a0, 4) / (U)_mm256_extract_epi32(b0, 4),
                                       (U)_mm256_extract_epi32(a0, 5) / (U)_mm256_extract_epi32(b0, 5),
                                       (U)_mm256_extract_epi32(a0, 6) / (U)_mm256_extract_epi32(b0, 6),
                                       (U)_mm256_extract_epi32(a0, 7) / (U)_mm256_extract_epi32(b0, 7));
        __m256i c1 = _mm256_setr_epi32((U)_mm256_extract_epi32(a1, 0) / (U)_mm256_extract_epi32(b1, 0),
                                       (U)_mm256_extract_epi32(a1, 1) / (U)_mm256_extract_epi32(b1, 1),
                                       (U)_mm256_extract_epi32(a1, 2) / (U)_mm256_extract_epi32(b1, 2),
                                       (U)_mm256_extract_epi32(a1, 3) / (U)_mm256_extract_epi32(b1, 3),
                                       (U)_mm256_extract_epi32(a1, 4) / (U)_mm256_extract_epi32(b1, 4),
                                       (U)_mm256_extract_epi32(a1, 5) / (U)_mm256_extract_epi32(b1, 5),
                                       (U)_mm256_extract_epi32(a1, 6) / (U)_mm256_extract_epi32(b1, 6),
                                       (U)_mm256_extract_epi32(a1, 7) / (U)_mm256_extract_epi32(b1, 7));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i pow(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i c0 = _mm256_setr_epi32((U)std::pow((U)_mm256_extract_epi32(a0, 0), (U)_mm256_extract_epi32(b0, 0)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 1), (U)_mm256_extract_epi32(b0, 1)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 2), (U)_mm256_extract_epi32(b0, 2)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 3), (U)_mm256_extract_epi32(b0, 3)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 4), (U)_mm256_extract_epi32(b0, 4)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 5), (U)_mm256_extract_epi32(b0, 5)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 6), (U)_mm256_extract_epi32(b0, 6)),
                                       (U)std::pow((U)_mm256_extract_epi32(a0, 7), (U)_mm256_extract_epi32(b0, 7)));
        __m256i c1 = _mm256_setr_epi32((U)std::pow((U)_mm256_extract_epi32(a1, 0), (U)_mm256_extract_epi32(b1, 0)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 1), (U)_mm256_extract_epi32(b1, 1)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 2), (U)_mm256_extract_epi32(b1, 2)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 3), (U)_mm256_extract_epi32(b1, 3)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 4), (U)_mm256_extract_epi32(b1, 4)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 5), (U)_mm256_extract_epi32(b1, 5)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 6), (U)_mm256_extract_epi32(b1, 6)),
                                       (U)std::pow((U)_mm256_extract_epi32(a1, 7), (U)_mm256_extract_epi32(b1, 7)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(c0), c1, 1);
    }

    static __m512i negate(__m512i a)
    {
        return _mm512_sub_epi32(_mm512_setzero_si512(), a);
    }

    static __m512i exp(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm256_setr_epi32((U)std::exp((U)_mm256_extract_epi32(a0, 0)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 1)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 2)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 3)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 4)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 5)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 6)),
                                       (U)std::exp((U)_mm256_extract_epi32(a0, 7)));
        __m256i b1 = _mm256_setr_epi32((U)std::exp((U)_mm256_extract_epi32(a1, 0)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 1)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 2)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 3)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 4)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 5)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 6)),
                                       (U)std::exp((U)_mm256_extract_epi32(a1, 7)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1);
    }

    static __m512i sqrt(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm256_setr_epi32((U)std::sqrt((U)_mm256_extract_epi32(a0, 0)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 1)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 2)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 3)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 4)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 5)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 6)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a0, 7)));
        __m256i b1 = _mm256_setr_epi32((U)std::sqrt((U)_mm256_extract_epi32(a1, 0)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 1)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 2)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 3)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 4)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 5)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 6)),
                                       (U)std::sqrt((U)_mm256_extract_epi32(a1, 7)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(b0), b1, 1);
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int64_t>::value ||
                                            std::is_same<U,uint64_t>::value>>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 64;
    typedef __m512i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m512>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        float a = (U)_mm256_extract_epi64(lo, 0);
        float b = (U)_mm256_extract_epi64(lo, 1);
        float c = (U)_mm256_extract_epi64(lo, 2);
        float d = (U)_mm256_extract_epi64(lo, 3);
        float e = (U)_mm256_extract_epi64(hi, 0);
        float f = (U)_mm256_extract_epi64(hi, 1);
        float g = (U)_mm256_extract_epi64(hi, 2);
        float h = (U)_mm256_extract_epi64(hi, 3);
        return _mm512_setr_ps(a, b, c, d, e, f, g, h,
                              a, b, c, d, e, f, g, h);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m512d>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        double a = (U)_mm256_extract_epi64(lo, 0);
        double b = (U)_mm256_extract_epi64(lo, 1);
        double c = (U)_mm256_extract_epi64(lo, 2);
        double d = (U)_mm256_extract_epi64(lo, 3);
        double e = (U)_mm256_extract_epi64(hi, 0);
        double f = (U)_mm256_extract_epi64(hi, 1);
        double g = (U)_mm256_extract_epi64(hi, 2);
        double h = (U)_mm256_extract_epi64(hi, 3);
        return _mm512_setr_pd(a, b, c, d, e, f, g, h);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m512>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        float a = (U)_mm256_extract_epi64(lo, 0);
        float b = (U)_mm256_extract_epi64(lo, 1);
        float c = (U)_mm256_extract_epi64(lo, 2);
        float d = (U)_mm256_extract_epi64(lo, 3);
        float e = (U)_mm256_extract_epi64(hi, 0);
        float f = (U)_mm256_extract_epi64(hi, 1);
        float g = (U)_mm256_extract_epi64(hi, 2);
        float h = (U)_mm256_extract_epi64(hi, 3);
        return _mm512_setr_ps(a, 0, b, 0, c, 0, d, 0,
                              e, 0, f, 0, g, 0, h, 0);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m512d>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        double a = (U)_mm256_extract_epi64(lo, 0);
        double b = (U)_mm256_extract_epi64(lo, 1);
        double c = (U)_mm256_extract_epi64(lo, 2);
        double d = (U)_mm256_extract_epi64(lo, 3);
        return _mm512_setr_pd(a, 0, b, 0, c, 0, d, 0);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        T a = (U)_mm256_extract_epi64(lo, 0);
        T b = (U)_mm256_extract_epi64(lo, 1);
        T c = (U)_mm256_extract_epi64(lo, 2);
        T d = (U)_mm256_extract_epi64(lo, 3);
        T e = (U)_mm256_extract_epi64(hi, 0);
        T f = (U)_mm256_extract_epi64(hi, 1);
        T g = (U)_mm256_extract_epi64(hi, 2);
        T h = (U)_mm256_extract_epi64(hi, 3);
        __m256i i8 = _mm256_setr_epi8(a, b, c, d, e, f, g, h,
                                      a, b, c, d, e, f, g, h,
                                      a, b, c, d, e, f, g, h,
                                      a, b, c, d, e, f, g, h);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        T a = (U)_mm256_extract_epi64(lo, 0);
        T b = (U)_mm256_extract_epi64(lo, 1);
        T c = (U)_mm256_extract_epi64(lo, 2);
        T d = (U)_mm256_extract_epi64(lo, 3);
        T e = (U)_mm256_extract_epi64(hi, 0);
        T f = (U)_mm256_extract_epi64(hi, 1);
        T g = (U)_mm256_extract_epi64(hi, 2);
        T h = (U)_mm256_extract_epi64(hi, 3);
        __m256i i16 = _mm256_setr_epi16(a, b, c, d, e, f, g, h,
                                        a, b, c, d, e, f, g, h);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m512i>
    convert(__m512i v)
    {
        __m256i lo = _mm512_extracti64x4_epi64(v, 0);
        __m256i hi = _mm512_extracti64x4_epi64(v, 1);
        T a = (U)_mm256_extract_epi64(lo, 0);
        T b = (U)_mm256_extract_epi64(lo, 1);
        T c = (U)_mm256_extract_epi64(lo, 2);
        T d = (U)_mm256_extract_epi64(lo, 3);
        T e = (U)_mm256_extract_epi64(hi, 0);
        T f = (U)_mm256_extract_epi64(hi, 1);
        T g = (U)_mm256_extract_epi64(hi, 2);
        T h = (U)_mm256_extract_epi64(hi, 3);
        __m256i i32 = _mm256_setr_epi32(a, b, c, d, e, f, g, h);
        return _mm512_inserti64x4(_mm512_castsi256_si512(i32), i32, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m512i>
    convert(__m512i v)
    {
        return v;
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_loadu_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_load_si512((__m512i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_loadu_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i64x4(_mm256_load_si256((__m256i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i*)ptr));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m512i>
    load(const U* ptr)
    {
        return _mm512_broadcast_i32x4(_mm_load_si128((__m128i*)ptr));
    }

    static __m512i load1(const U* ptr)
    {
        return _mm512_set1_epi64(*ptr);
    }

    static __m512i set1(U val)
    {
        return _mm512_set1_epi64(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_storeu_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm512_store_si512((__m512i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, _mm512_castsi512_si256(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m512i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm512_castsi512_si128(v));
    }

    static __m512i add(__m512i a, __m512i b)
    {
        return _mm512_add_epi64(a, b);
    }

    static __m512i sub(__m512i a, __m512i b)
    {
        return _mm512_sub_epi64(a, b);
    }

    static __m512i mul(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i lo = _mm256_setr_epi64x((U)_mm256_extract_epi64(a0, 0) * (U)_mm256_extract_epi64(b0, 0),
                                        (U)_mm256_extract_epi64(a0, 1) * (U)_mm256_extract_epi64(b0, 1),
                                        (U)_mm256_extract_epi64(a0, 2) * (U)_mm256_extract_epi64(b0, 2),
                                        (U)_mm256_extract_epi64(a0, 3) * (U)_mm256_extract_epi64(b0, 3));
        __m256i hi = _mm256_setr_epi64x((U)_mm256_extract_epi64(a1, 0) * (U)_mm256_extract_epi64(b1, 0),
                                        (U)_mm256_extract_epi64(a1, 1) * (U)_mm256_extract_epi64(b1, 1),
                                        (U)_mm256_extract_epi64(a1, 2) * (U)_mm256_extract_epi64(b1, 2),
                                        (U)_mm256_extract_epi64(a1, 3) * (U)_mm256_extract_epi64(b1, 3));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    }

    static __m512i div(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i lo = _mm256_setr_epi64x((U)_mm256_extract_epi64(a0, 0) / (U)_mm256_extract_epi64(b0, 0),
                                        (U)_mm256_extract_epi64(a0, 1) / (U)_mm256_extract_epi64(b0, 1),
                                        (U)_mm256_extract_epi64(a0, 2) / (U)_mm256_extract_epi64(b0, 2),
                                        (U)_mm256_extract_epi64(a0, 3) / (U)_mm256_extract_epi64(b0, 3));
        __m256i hi = _mm256_setr_epi64x((U)_mm256_extract_epi64(a1, 0) / (U)_mm256_extract_epi64(b1, 0),
                                        (U)_mm256_extract_epi64(a1, 1) / (U)_mm256_extract_epi64(b1, 1),
                                        (U)_mm256_extract_epi64(a1, 2) / (U)_mm256_extract_epi64(b1, 2),
                                        (U)_mm256_extract_epi64(a1, 3) / (U)_mm256_extract_epi64(b1, 3));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    }

    static __m512i pow(__m512i a, __m512i b)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i b0 = _mm512_extracti64x4_epi64(b, 0);
        __m256i b1 = _mm512_extracti64x4_epi64(b, 1);
        __m256i lo = _mm256_setr_epi64x((U)std::pow((U)_mm256_extract_epi64(a0, 0), (U)_mm256_extract_epi64(b0, 0)),
                                        (U)std::pow((U)_mm256_extract_epi64(a0, 1), (U)_mm256_extract_epi64(b0, 1)),
                                        (U)std::pow((U)_mm256_extract_epi64(a0, 2), (U)_mm256_extract_epi64(b0, 2)),
                                        (U)std::pow((U)_mm256_extract_epi64(a0, 3), (U)_mm256_extract_epi64(b0, 3)));
        __m256i hi = _mm256_setr_epi64x((U)std::pow((U)_mm256_extract_epi64(a1, 0), (U)_mm256_extract_epi64(b1, 0)),
                                        (U)std::pow((U)_mm256_extract_epi64(a1, 1), (U)_mm256_extract_epi64(b1, 1)),
                                        (U)std::pow((U)_mm256_extract_epi64(a1, 2), (U)_mm256_extract_epi64(b1, 2)),
                                        (U)std::pow((U)_mm256_extract_epi64(a1, 3), (U)_mm256_extract_epi64(b1, 3)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    }

    static __m512i negate(__m512i a)
    {
        return _mm512_sub_epi64(_mm512_setzero_si512(), a);
    }

    static __m512i exp(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i lo = _mm256_setr_epi64x((U)std::exp((U)_mm256_extract_epi64(a0, 0)),
                                        (U)std::exp((U)_mm256_extract_epi64(a0, 1)),
                                        (U)std::exp((U)_mm256_extract_epi64(a0, 2)),
                                        (U)std::exp((U)_mm256_extract_epi64(a0, 3)));
        __m256i hi = _mm256_setr_epi64x((U)std::exp((U)_mm256_extract_epi64(a1, 0)),
                                        (U)std::exp((U)_mm256_extract_epi64(a1, 1)),
                                        (U)std::exp((U)_mm256_extract_epi64(a1, 2)),
                                        (U)std::exp((U)_mm256_extract_epi64(a1, 3)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    }

    static __m512i sqrt(__m512i a)
    {
        __m256i a0 = _mm512_extracti64x4_epi64(a, 0);
        __m256i a1 = _mm512_extracti64x4_epi64(a, 1);
        __m256i lo = _mm256_setr_epi64x((U)std::sqrt((U)_mm256_extract_epi64(a0, 0)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a0, 1)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a0, 2)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a0, 3)));
        __m256i hi = _mm256_setr_epi64x((U)std::sqrt((U)_mm256_extract_epi64(a1, 0)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a1, 1)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a1, 2)),
                                        (U)std::sqrt((U)_mm256_extract_epi64(a1, 3)));
        return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
    }
};

}

#endif

#endif //MARRAY_VECTOR_AVX512_HPP
