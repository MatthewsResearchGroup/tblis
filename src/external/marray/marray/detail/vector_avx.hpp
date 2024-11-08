#ifndef MARRAY_VECTOR_AVX_HPP
#define MARRAY_VECTOR_AVX_HPP

#if defined(__AVX__)

#include <immintrin.h>
#include <x86intrin.h>

#include "vector.hpp"

namespace MArray
{

template <>
struct vector_traits<float>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 32;
    typedef __m256 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256 v)
    {
        return _mm256_cvtps_pd(_mm256_castps256_ps128(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256 v)
    {
        __m128 lo = _mm256_extractf128_ps(v, 0);
        __m256 dup = _mm256_insertf128_ps(_mm256_permute_ps(v, _MM_SHUFFLE(1,0,1,0)),
                                          _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3,2,3,2)), 1);
        return _mm256_unpacklo_ps(dup, _mm256_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256 v)
    {
        __m256d dup = _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_permute_ps(v, _MM_SHUFFLE(1,1,0,0))));
        return _mm256_unpacklo_pd(dup, _mm256_setzero_pd());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m256i>
    convert(__m256 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(v);
#ifdef __AVX2__
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_SHUFFLE(2,0,2,0));
        return _mm256_packs_epi16(i16, i16);
#else
        __m128i i32lo = _mm256_castsi256_si128(i32);
        __m128i i32hi = _mm256_extractf128_si256(i32, 1);
        __m128i i16 = _mm_packs_epi32(i32lo, i32hi);
        __m128i i8 = _mm_packs_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(v);
#ifdef __AVX2__
        __m256i i16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_SHUFFLE(2,0,2,0));
        return _mm256_packus_epi16(i16, i16);
#else
        __m128i i32lo = _mm256_castsi256_si128(i32);
        __m128i i32hi = _mm256_extractf128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(i32lo, i32hi);
        __m128i i8 = _mm_packus_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m256i>
    convert(__m256 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(v);
#ifdef __AVX2__
        return _mm256_permute4x64_epi64(_mm256_packs_epi32(i32, i32), _MM_SHUFFLE(2,0,2,0));
#else
        __m128i i32lo = _mm256_castsi256_si128(i32);
        __m128i i32hi = _mm256_extractf128_si256(i32, 1);
        __m128i i16 = _mm_packs_epi32(i32lo, i32hi);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256 v)
    {
        __m256i i32 = _mm256_cvtps_epi32(v);
#ifdef __AVX2__
        return _mm256_permute4x64_epi64(_mm256_packus_epi32(i32, i32), _MM_SHUFFLE(2,0,2,0));
#else
        __m128i i32lo = _mm256_castsi256_si128(i32);
        __m128i i32hi = _mm256_extractf128_si256(i32, 1);
        __m128i i16 = _mm_packus_epi32(i32lo, i32hi);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_cvtps_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
                                 (uint32_t)v[2], (uint32_t)v[3],
                                 (uint32_t)v[4], (uint32_t)v[5],
                                 (uint32_t)v[6], (uint32_t)v[7]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_setr_epi64x((T)v[0], (T)v[1],
                                  (T)v[2], (T)v[3]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m256>
    load(const float* ptr)
    {
        return _mm256_loadu_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m256>
    load(const float* ptr)
    {
        return _mm256_load_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m256>
    load(const float* ptr)
    {
        return _mm256_broadcast_ps((__m128*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256>
    load(const float* ptr)
    {
        return _mm256_castpd_ps(_mm256_broadcast_sd((double*)ptr));
    }

    static __m256 load1(const float* ptr)
    {
        return _mm256_broadcast_ss(ptr);
    }

    static __m256 set1(float val)
    {
        return _mm256_set1_ps(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m256 v, float* ptr)
    {
        _mm256_storeu_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m256 v, float* ptr)
    {
        _mm256_store_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m256 v, float* ptr)
    {
        _mm_storeu_ps(ptr, _mm256_castps256_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m256 v, float* ptr)
    {
        _mm_store_ps(ptr, _mm256_castps256_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m256 v, float* ptr)
    {
        _mm_store_sd((double*)ptr, _mm_castps_pd(_mm256_castps256_ps128(v)));
    }

    static __m256 add(__m256 a, __m256 b)
    {
        return _mm256_add_ps(a, b);
    }

    static __m256 sub(__m256 a, __m256 b)
    {
        return _mm256_sub_ps(a, b);
    }

    static __m256 mul(__m256 a, __m256 b)
    {
        return _mm256_mul_ps(a, b);
    }

    static __m256 div(__m256 a, __m256 b)
    {
        return _mm256_div_ps(a, b);
    }

    static __m256 pow(__m256 a, __m256 b)
    {
        return _mm256_setr_ps(std::pow((float)a[0], (float)b[0]),
                              std::pow((float)a[1], (float)b[1]),
                              std::pow((float)a[2], (float)b[2]),
                              std::pow((float)a[3], (float)b[3]),
                              std::pow((float)a[4], (float)b[4]),
                              std::pow((float)a[5], (float)b[5]),
                              std::pow((float)a[6], (float)b[6]),
                              std::pow((float)a[7], (float)b[7]));
    }

    static __m256 negate(__m256 a)
    {
        return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
    }

    static __m256 exp(__m256 a)
    {
        return _mm256_setr_ps(std::exp((float)a[0]),
                              std::exp((float)a[1]),
                              std::exp((float)a[2]),
                              std::exp((float)a[3]),
                              std::exp((float)a[4]),
                              std::exp((float)a[5]),
                              std::exp((float)a[6]),
                              std::exp((float)a[7]));
    }

    static __m256 log(__m256 a)
    {
        return _mm256_setr_ps(std::log((float)a[0]),
                              std::log((float)a[1]),
                              std::log((float)a[2]),
                              std::log((float)a[3]),
                              std::log((float)a[4]),
                              std::log((float)a[5]),
                              std::log((float)a[6]),
                              std::log((float)a[7]));
    }

    static __m256 abs(__m256 a)
    {
        return _mm256_and_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    }

    static __m256 sqrt(__m256 a)
    {
        return _mm256_sqrt_ps(a);
    }
};

template <>
struct vector_traits<double>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 32;
    typedef __m256d vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256d v)
    {
        __m256 lo = _mm256_castps128_ps256(_mm256_cvtpd_ps(v));
        return _mm256_permute2f128_ps(lo, lo, 0x00);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256d v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256d v)
    {
        __m256 sp = _mm256_castps128_ps256(_mm256_cvtpd_ps(v));
        __m128 lo = _mm256_extractf128_ps(sp, 0);
        __m256 dup = _mm256_insertf128_ps(_mm256_permute_ps(sp, _MM_SHUFFLE(1,0,1,0)),
                                          _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3,2,3,2)), 1);
        return _mm256_unpacklo_ps(dup, _mm256_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256d v)
    {
        return _mm256_shuffle_pd(v, _mm256_setzero_pd(), 0xc);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(v);
        __m128i i16 = _mm_packs_epi32(i32, i32);
        __m128i i8 = _mm_packs_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(v);
        __m128i i16 = _mm_packus_epi32(i32, i32);
        __m128i i8 = _mm_packus_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(v);
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(v);
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m256i>
    convert(__m256d v)
    {
        return _mm256_castsi128_si256(_mm256_cvtpd_epi32(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256d v)
    {
        return _mm256_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
                                 (uint32_t)v[2], (uint32_t)v[3],
                                 (uint32_t)v[0], (uint32_t)v[1],
                                 (uint32_t)v[2], (uint32_t)v[3]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256d v)
    {
        return _mm256_setr_epi64x((T)v[0], (T)v[1],
                                  (T)v[2], (T)v[3]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m256d>
    load(const double* ptr)
    {
        return _mm256_loadu_pd(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m256d>
    load(const double* ptr)
    {
        return _mm256_load_pd(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256d>
    load(const double* ptr)
    {
        return _mm256_broadcast_pd((__m128d*)ptr);
    }

    static __m256d load1(const double* ptr)
    {
        return _mm256_broadcast_sd(ptr);
    }

    static __m256d set1(double val)
    {
        return _mm256_set1_pd(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m256d v, double* ptr)
    {
        _mm256_storeu_pd(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m256d v, double* ptr)
    {
        _mm256_store_pd(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m256d v, double* ptr)
    {
        _mm_storeu_pd(ptr, _mm256_castpd256_pd128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m256d v, double* ptr)
    {
        _mm_store_pd(ptr, _mm256_castpd256_pd128(v));
    }

    static __m256d add(__m256d a, __m256d b)
    {
        return _mm256_add_pd(a, b);
    }

    static __m256d sub(__m256d a, __m256d b)
    {
        return _mm256_sub_pd(a, b);
    }

    static __m256d mul(__m256d a, __m256d b)
    {
        return _mm256_mul_pd(a, b);
    }

    static __m256d div(__m256d a, __m256d b)
    {
        return _mm256_div_pd(a, b);
    }

    static __m256d pow(__m256d a, __m256d b)
    {
        return _mm256_setr_pd(std::pow((double)a[0], (double)b[0]),
                              std::pow((double)a[1], (double)b[1]),
                              std::pow((double)a[2], (double)b[2]),
                              std::pow((double)a[3], (double)b[3]));
    }

    static __m256d negate(__m256d a)
    {
        return _mm256_xor_pd(a, _mm256_set1_pd(-0.0));
    }

    static __m256d exp(__m256d a)
    {
        return _mm256_setr_pd(std::exp((double)a[0]),
                              std::exp((double)a[1]),
                              std::exp((double)a[2]),
                              std::exp((double)a[3]));
    }

    static __m256d log(__m256d a)
    {
        return _mm256_setr_pd(std::log((double)a[0]),
                              std::log((double)a[1]),
                              std::log((double)a[2]),
                              std::log((double)a[3]));
    }

    static __m256d abs(__m256d a)
    {
        return _mm256_and_pd(a, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFull)));
    }

    static __m256d sqrt(__m256d a)
    {
        return _mm256_sqrt_pd(a);
    }
};

template <>
struct vector_traits<std::complex<float>>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 32;
    typedef __m256 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256 v)
    {
        __m256 tmp = _mm256_permute_ps(v, _MM_SHUFFLE(2,0,2,0));
#ifdef __AVX2__
        return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(tmp), _MM_SHUFFLE(2,0,2,0)));
#else
        __m256 lo = _mm256_permute2f128_ps(tmp, tmp, 0x00);
        __m256 hi = _mm256_permute2f128_ps(tmp, tmp, 0x11);
        return _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(1,0,1,0));
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256 v)
    {
        return _mm256_cvtps_pd(_mm256_castps256_ps128(convert<float>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256 v)
    {
        return _mm256_cvtps_pd(_mm256_castps256_ps128(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m256i>
    convert(__m256 v)
    {
        __m128i i32 = _mm_cvtps_epi32(_mm256_castps256_ps128(convert<float>(v)));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        __m128i i8 = _mm_packs_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256 v)
    {
        __m128i i32 = _mm_cvtps_epi32(_mm256_castps256_ps128(convert<float>(v)));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        __m128i i8 = _mm_packus_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m256i>
    convert(__m256 v)
    {
        __m128i i32 = _mm_cvtps_epi32(_mm256_castps256_ps128(convert<float>(v)));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256 v)
    {
        __m128i i32 = _mm_cvtps_epi32(_mm256_castps256_ps128(convert<float>(v)));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_cvtps_epi32(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_setr_epi32((uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[4], (uint32_t)v[6]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256 v)
    {
        return _mm256_setr_epi64x((T)v[0], (T)v[2],
                                  (T)v[4], (T)v[6]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m256>
    load(const std::complex<float>* ptr)
    {
        return _mm256_loadu_ps((float*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m256>
    load(const std::complex<float>* ptr)
    {
        return _mm256_load_ps((float*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256>
    load(const std::complex<float>* ptr)
    {
        return _mm256_broadcast_ps((__m128*)ptr);
    }

    static __m256 load1(const std::complex<float>* ptr)
    {
        return set1(*ptr);
    }

    static __m256 set1(std::complex<float> val)
    {
        double dval;
        memcpy(&dval, &val, 8);
        return _mm256_castpd_ps(_mm256_set1_pd(dval));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m256 v, std::complex<float>* ptr)
    {
        _mm256_storeu_ps((float*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m256 v, std::complex<float>* ptr)
    {
        _mm256_store_ps((float*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m256 v, std::complex<float>* ptr)
    {
        _mm_storeu_ps((float*)ptr, _mm256_castps256_ps128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m256 v, std::complex<float>* ptr)
    {
        _mm_store_ps((float*)ptr, _mm256_castps256_ps128(v));
    }

    static __m256 add(__m256 a, __m256 b)
    {
        return _mm256_add_ps(a, b);
    }

    static __m256 sub(__m256 a, __m256 b)
    {
        return _mm256_sub_ps(a, b);
    }

    static __m256 mul(__m256 a, __m256 b)
    {
        __m256 ashuf = _mm256_permute_ps(a, _MM_SHUFFLE(2,3,0,1));
        __m256 breal = _mm256_moveldup_ps(b);
        __m256 bimag = _mm256_movehdup_ps(b);
        __m256 tmp1 = _mm256_mul_ps(    a, breal); // tmp1 = (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
        __m256 tmp2 = _mm256_mul_ps(ashuf, bimag); // tmp2 = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm256_addsub_ps(tmp1, tmp2);
    }

    static __m256 div(__m256 a, __m256 b)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> a2((float)a[4], (float)a[5]);
        std::complex<float> a3((float)a[6], (float)a[7]);
        std::complex<float> b0((float)b[0], (float)b[1]);
        std::complex<float> b1((float)b[2], (float)b[3]);
        std::complex<float> b2((float)b[4], (float)b[5]);
        std::complex<float> b3((float)b[6], (float)b[7]);
        std::complex<float> c0 = a0 / b0;
        std::complex<float> c1 = a1 / b1;
        std::complex<float> c2 = a2 / b2;
        std::complex<float> c3 = a3 / b3;
        return _mm256_setr_ps(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag());
    }

    static __m256 pow(__m256 a, __m256 b)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> a2((float)a[4], (float)a[5]);
        std::complex<float> a3((float)a[6], (float)a[7]);
        std::complex<float> b0((float)b[0], (float)b[1]);
        std::complex<float> b1((float)b[2], (float)b[3]);
        std::complex<float> b2((float)b[4], (float)b[5]);
        std::complex<float> b3((float)b[6], (float)b[7]);
        std::complex<float> c0 = std::pow(a0, b0);
        std::complex<float> c1 = std::pow(a1, b1);
        std::complex<float> c2 = std::pow(a2, b2);
        std::complex<float> c3 = std::pow(a3, b3);
        return _mm256_setr_ps(c0.real(), c0.imag(),
                              c1.real(), c1.imag(),
                              c2.real(), c2.imag(),
                              c3.real(), c3.imag());
    }

    static __m256 negate(__m256 a)
    {
        return _mm256_xor_ps(a, _mm256_set1_ps(-0.0f));
    }

    static __m256 exp(__m256 a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> a2((float)a[4], (float)a[5]);
        std::complex<float> a3((float)a[6], (float)a[7]);
        std::complex<float> b0 = std::exp(a0);
        std::complex<float> b1 = std::exp(a1);
        std::complex<float> b2 = std::exp(a2);
        std::complex<float> b3 = std::exp(a3);
        return _mm256_setr_ps(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag());
    }

    static __m256 sqrt(__m256 a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> a2((float)a[4], (float)a[5]);
        std::complex<float> a3((float)a[6], (float)a[7]);
        std::complex<float> b0 = std::sqrt(a0);
        std::complex<float> b1 = std::sqrt(a1);
        std::complex<float> b2 = std::sqrt(a2);
        std::complex<float> b3 = std::sqrt(a3);
        return _mm256_setr_ps(b0.real(), b0.imag(),
                              b1.real(), b1.imag(),
                              b2.real(), b2.imag(),
                              b3.real(), b3.imag());
    }
};

template <>
struct vector_traits<std::complex<double>>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 32;
    typedef __m256d vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256d v)
    {
        __m256 lo = _mm256_castps128_ps256(_mm256_cvtpd_ps(convert<double>(v)));
        return _mm256_permute2f128_ps(lo, lo, 0x00);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256d v)
    {
#ifdef __AVX2__
        return _mm256_permute4x64_pd(v, _MM_SHUFFLE(2,0,2,0));
#else
        __m256d lo = _mm256_permute2f128_pd(v, v, 0x00);
        __m256d hi = _mm256_permute2f128_pd(v, v, 0x11);
        return _mm256_shuffle_pd(lo, hi, 0x0);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256d v)
    {
        __m256 lo = _mm256_castps128_ps256(_mm256_cvtpd_ps(v));
        return _mm256_permute2f128_ps(lo, lo, 0x00);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256d v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(convert<double>(v));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        __m128i i8 = _mm_packs_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(convert<double>(v));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        __m128i i8 = _mm_packus_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i8), i8, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(convert<double>(v));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256d v)
    {
        __m128i i32 = _mm256_cvtpd_epi32(convert<double>(v));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(i16), i16, 1);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m256i>
    convert(__m256d v)
    {
        __m256i lo = _mm256_castsi128_si256(_mm256_cvtpd_epi32(convert<double>(v)));
        return _mm256_permute2f128_si256(lo, lo, 0x00);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256d v)
    {
        return _mm256_setr_epi32((uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[0], (uint32_t)v[2],
                                 (uint32_t)v[0], (uint32_t)v[2]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256d v)
    {
        return _mm256_setr_epi64x((T)v[0], (T)v[2],
                                  (T)v[0], (T)v[2]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m256d>
    load(const std::complex<double>* ptr)
    {
        return _mm256_loadu_pd((double*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m256d>
    load(const std::complex<double>* ptr)
    {
        return _mm256_load_pd((double*)ptr);
    }

    static __m256d load1(const std::complex<double>* ptr)
    {
        return _mm256_broadcast_pd((__m128d*)ptr);
    }

    static __m256d set1(std::complex<double> val)
    {
        return _mm256_setr_pd(val.real(), val.imag(), val.real(), val.imag());
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m256d v, std::complex<double>* ptr)
    {
        _mm256_storeu_pd((double*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m256d v, std::complex<double>* ptr)
    {
        _mm256_store_pd((double*)ptr, v);
    }

    static __m256d add(__m256d a, __m256d b)
    {
        return _mm256_add_pd(a, b);
    }

    static __m256d sub(__m256d a, __m256d b)
    {
        return _mm256_sub_pd(a, b);
    }

    static __m256d mul(__m256d a, __m256d b)
    {
        __m256d ashuf = _mm256_shuffle_pd(a, a, 0x5);
        __m256d breal = _mm256_shuffle_pd(b, b, 0x0);
        __m256d bimag = _mm256_shuffle_pd(b, b, 0xf);
        __m256d tmp1 = _mm256_mul_pd(    a, breal); // tmp1 = (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
        __m256d tmp2 = _mm256_mul_pd(ashuf, bimag); // tmp2 = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm256_addsub_pd(tmp1, tmp2);
    }

    static __m256d div(__m256d a, __m256d b)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> b0((double)b[0], (double)b[1]);
        std::complex<double> b1((double)b[2], (double)b[3]);
        std::complex<double> c0 = a0 / b0;
        std::complex<double> c1 = a1 / b1;
        return _mm256_setr_pd(c0.real(), c0.imag(),
                              c1.real(), c1.imag());
    }

    static __m256d pow(__m256d a, __m256d b)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> b0((double)b[0], (double)b[1]);
        std::complex<double> b1((double)b[2], (double)b[3]);
        std::complex<double> c0 = std::pow(a0, b0);
        std::complex<double> c1 = std::pow(a1, b1);
        return _mm256_setr_pd(c0.real(), c0.imag(),
                              c1.real(), c1.imag());
    }

    static __m256d negate(__m256d a)
    {
        return _mm256_xor_pd(a, _mm256_set1_pd(-0.0));
    }

    static __m256d exp(__m256d a)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> b0 = std::exp(a0);
        std::complex<double> b1 = std::exp(a1);
        return _mm256_setr_pd(b0.real(), b0.imag(),
                              b1.real(), b1.imag());
    }

    static __m256d sqrt(__m256d a)
    {
        std::complex<double> a0((double)a[0], (double)a[1]);
        std::complex<double> a1((double)a[2], (double)a[3]);
        std::complex<double> b0 = std::sqrt(a0);
        std::complex<double> b1 = std::sqrt(a1);
        return _mm256_setr_pd(b0.real(), b0.imag(),
                              b1.real(), b1.imag());
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int8_t>::value ||
                                            std::is_same<U,uint8_t>::value>>
{
    static constexpr int vector_width = 32;
    static constexpr size_t alignment = 32;
    typedef __m256i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_pd(_mm256_castsi256_si128(convert<int32_t>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256i v)
    {
        __m128i mask = _mm_set_epi8(3,2,3,2,1,0,1,0,3,2,3,2,1,0,1,0);
        __m128i dup = _mm_shuffle_epi8(_mm256_castsi256_si128(v), mask);
        return _mm256_unpacklo_ps(convert<float>(_mm256_castsi128_si256(dup)), _mm256_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256i v)
    {
        __m128i mask = _mm_set_epi8(1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0);
        __m128i dup = _mm_shuffle_epi8(_mm256_castsi256_si128(v), mask);
        return _mm256_unpacklo_pd(convert<double>(_mm256_castsi128_si256(dup)), _mm256_setzero_pd());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v));
#else
        __m128i lo8 = _mm256_castsi256_si128(v);
        __m128i hi8 = _mm_shuffle_epi32(lo8, _MM_SHUFFLE(1,0,3,2));
        __m128i lo16 = std::is_signed<U>::value ? _mm_cvtepi8_epi16(lo8)
                                                : _mm_cvtepu8_epi16(lo8);
        __m128i hi16 = std::is_signed<U>::value ? _mm_cvtepi8_epi16(hi8)
                                                : _mm_cvtepu8_epi16(hi8);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo16), hi16, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi8_epi32(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu8_epi32(_mm256_castsi256_si128(v));
#else
        __m128i lo16 = std::is_signed<U>::value ? _mm_cvtepi8_epi16(_mm256_castsi256_si128(v))
                                                : _mm_cvtepu8_epi16(_mm256_castsi256_si128(v));
        __m128i hi16 = _mm_shuffle_epi32(lo16, _MM_SHUFFLE(1,0,3,2));
        __m128i lo32 = std::is_signed<U>::value ? _mm_cvtepi16_epi32(lo16)
                                                : _mm_cvtepu16_epi32(lo16);
        __m128i hi32 = std::is_signed<U>::value ? _mm_cvtepi16_epi32(hi16)
                                                : _mm_cvtepu16_epi32(hi16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo32), hi32, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi8_epi64(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu8_epi64(_mm256_castsi256_si128(v));
#else
        __m128i lo16 = std::is_signed<U>::value ? _mm_cvtepi8_epi16(_mm256_castsi256_si128(v))
                                               : _mm_cvtepu8_epi16(_mm256_castsi256_si128(v));
        __m128i hi16 = _mm_shuffle_epi32(lo16, _MM_SHUFFLE(3,2,0,1));
        __m128i lo64 = std::is_signed<U>::value ? _mm_cvtepi16_epi64(lo16)
                                                : _mm_cvtepu16_epi64(lo16);
        __m128i hi64 = std::is_signed<U>::value ? _mm_cvtepi16_epi64(hi16)
                                                : _mm_cvtepu16_epi64(hi16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo64), hi64, 1);
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_loadu_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_load_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi64x(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi32(*(int32_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi16(*(int16_t*)ptr);
    }

    static __m256i load1(const U* ptr)
    {
        return _mm256_set1_epi8(*ptr);
    }

    static __m256i set1(U val)
    {
        return _mm256_set1_epi8(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 32 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8>
    store(__m256i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m256i v, U* ptr)
    {
        *(int32_t*)ptr = _mm_extract_epi32(_mm256_castsi256_si128(v), 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m256i v, U* ptr)
    {
        *(int16_t*)ptr = _mm_extract_epi16(_mm256_castsi256_si128(v), 0);
    }

    static __m256i add(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_add_epi8(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_add_epi8(a0, b0);
        __m128i ab1 = _mm_add_epi8(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i sub(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_sub_epi8(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_sub_epi8(a0, b0);
        __m128i ab1 = _mm_sub_epi8(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i mul(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        __m256i lo = _mm256_and_si256(_mm256_mullo_epi16(a, b), _mm256_set1_epi16(0xff));
        __m256i hi = _mm256_mullo_epi16(_mm256_srli_epi16(a, 8),_mm256_srli_epi16(b, 8));
        return _mm256_or_si256(_mm256_slli_epi16(hi, 8), lo);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i lo0 = _mm_and_si128(_mm_mullo_epi16(a0, b0), _mm_set1_epi16(0xff));
        __m128i lo1 = _mm_and_si128(_mm_mullo_epi16(a1, b1), _mm_set1_epi16(0xff));
        __m128i hi0 = _mm_mullo_epi16(_mm_srli_epi16(a0, 8),_mm_srli_epi16(b0, 8));
        __m128i hi1 = _mm_mullo_epi16(_mm_srli_epi16(a1, 8),_mm_srli_epi16(b1, 8));
        __m128i ab0 = _mm_or_si128(_mm_slli_epi16(hi0, 8), lo0);
        __m128i ab1 = _mm_or_si128(_mm_slli_epi16(hi1, 8), lo1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i div(__m256i a, __m256i b)
    {
        return _mm256_setr_epi8((U)_mm256_extract_epi8(a, 0) /
                                (U)_mm256_extract_epi8(b, 0),
                                (U)_mm256_extract_epi8(a, 1) /
                                (U)_mm256_extract_epi8(b, 1),
                                (U)_mm256_extract_epi8(a, 2) /
                                (U)_mm256_extract_epi8(b, 2),
                                (U)_mm256_extract_epi8(a, 3) /
                                (U)_mm256_extract_epi8(b, 3),
                                (U)_mm256_extract_epi8(a, 4) /
                                (U)_mm256_extract_epi8(b, 4),
                                (U)_mm256_extract_epi8(a, 5) /
                                (U)_mm256_extract_epi8(b, 5),
                                (U)_mm256_extract_epi8(a, 6) /
                                (U)_mm256_extract_epi8(b, 6),
                                (U)_mm256_extract_epi8(a, 7) /
                                (U)_mm256_extract_epi8(b, 7),
                                (U)_mm256_extract_epi8(a, 8) /
                                (U)_mm256_extract_epi8(b, 8),
                                (U)_mm256_extract_epi8(a, 9) /
                                (U)_mm256_extract_epi8(b, 9),
                                (U)_mm256_extract_epi8(a,10) /
                                (U)_mm256_extract_epi8(b,10),
                                (U)_mm256_extract_epi8(a,11) /
                                (U)_mm256_extract_epi8(b,11),
                                (U)_mm256_extract_epi8(a,12) /
                                (U)_mm256_extract_epi8(b,12),
                                (U)_mm256_extract_epi8(a,13) /
                                (U)_mm256_extract_epi8(b,13),
                                (U)_mm256_extract_epi8(a,14) /
                                (U)_mm256_extract_epi8(b,14),
                                (U)_mm256_extract_epi8(a,15) /
                                (U)_mm256_extract_epi8(b,15),
                                (U)_mm256_extract_epi8(a,16) /
                                (U)_mm256_extract_epi8(b,16),
                                (U)_mm256_extract_epi8(a,17) /
                                (U)_mm256_extract_epi8(b,17),
                                (U)_mm256_extract_epi8(a,18) /
                                (U)_mm256_extract_epi8(b,18),
                                (U)_mm256_extract_epi8(a,19) /
                                (U)_mm256_extract_epi8(b,19),
                                (U)_mm256_extract_epi8(a,20) /
                                (U)_mm256_extract_epi8(b,20),
                                (U)_mm256_extract_epi8(a,21) /
                                (U)_mm256_extract_epi8(b,21),
                                (U)_mm256_extract_epi8(a,22) /
                                (U)_mm256_extract_epi8(b,22),
                                (U)_mm256_extract_epi8(a,23) /
                                (U)_mm256_extract_epi8(b,23),
                                (U)_mm256_extract_epi8(a,24) /
                                (U)_mm256_extract_epi8(b,24),
                                (U)_mm256_extract_epi8(a,25) /
                                (U)_mm256_extract_epi8(b,25),
                                (U)_mm256_extract_epi8(a,26) /
                                (U)_mm256_extract_epi8(b,26),
                                (U)_mm256_extract_epi8(a,27) /
                                (U)_mm256_extract_epi8(b,27),
                                (U)_mm256_extract_epi8(a,28) /
                                (U)_mm256_extract_epi8(b,28),
                                (U)_mm256_extract_epi8(a,29) /
                                (U)_mm256_extract_epi8(b,29),
                                (U)_mm256_extract_epi8(a,30) /
                                (U)_mm256_extract_epi8(b,30),
                                (U)_mm256_extract_epi8(a,31) /
                                (U)_mm256_extract_epi8(b,31));
    }

    static __m256i pow(__m256i a, __m256i b)
    {
        return _mm256_setr_epi8((U)std::pow((U)_mm256_extract_epi8(a, 0),
                                            (U)_mm256_extract_epi8(b, 0)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 1),
                                            (U)_mm256_extract_epi8(b, 1)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 2),
                                            (U)_mm256_extract_epi8(b, 2)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 3),
                                            (U)_mm256_extract_epi8(b, 3)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 4),
                                            (U)_mm256_extract_epi8(b, 4)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 5),
                                            (U)_mm256_extract_epi8(b, 5)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 6),
                                            (U)_mm256_extract_epi8(b, 6)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 7),
                                            (U)_mm256_extract_epi8(b, 7)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 8),
                                            (U)_mm256_extract_epi8(b, 8)),
                                (U)std::pow((U)_mm256_extract_epi8(a, 9),
                                            (U)_mm256_extract_epi8(b, 9)),
                                (U)std::pow((U)_mm256_extract_epi8(a,10),
                                            (U)_mm256_extract_epi8(b,10)),
                                (U)std::pow((U)_mm256_extract_epi8(a,11),
                                            (U)_mm256_extract_epi8(b,11)),
                                (U)std::pow((U)_mm256_extract_epi8(a,12),
                                            (U)_mm256_extract_epi8(b,12)),
                                (U)std::pow((U)_mm256_extract_epi8(a,13),
                                            (U)_mm256_extract_epi8(b,13)),
                                (U)std::pow((U)_mm256_extract_epi8(a,14),
                                            (U)_mm256_extract_epi8(b,14)),
                                (U)std::pow((U)_mm256_extract_epi8(a,15),
                                            (U)_mm256_extract_epi8(b,15)),
                                (U)std::pow((U)_mm256_extract_epi8(a,16),
                                            (U)_mm256_extract_epi8(b,16)),
                                (U)std::pow((U)_mm256_extract_epi8(a,17),
                                            (U)_mm256_extract_epi8(b,17)),
                                (U)std::pow((U)_mm256_extract_epi8(a,18),
                                            (U)_mm256_extract_epi8(b,18)),
                                (U)std::pow((U)_mm256_extract_epi8(a,19),
                                            (U)_mm256_extract_epi8(b,19)),
                                (U)std::pow((U)_mm256_extract_epi8(a,20),
                                            (U)_mm256_extract_epi8(b,20)),
                                (U)std::pow((U)_mm256_extract_epi8(a,21),
                                            (U)_mm256_extract_epi8(b,21)),
                                (U)std::pow((U)_mm256_extract_epi8(a,22),
                                            (U)_mm256_extract_epi8(b,22)),
                                (U)std::pow((U)_mm256_extract_epi8(a,23),
                                            (U)_mm256_extract_epi8(b,23)),
                                (U)std::pow((U)_mm256_extract_epi8(a,24),
                                            (U)_mm256_extract_epi8(b,24)),
                                (U)std::pow((U)_mm256_extract_epi8(a,25),
                                            (U)_mm256_extract_epi8(b,25)),
                                (U)std::pow((U)_mm256_extract_epi8(a,26),
                                            (U)_mm256_extract_epi8(b,26)),
                                (U)std::pow((U)_mm256_extract_epi8(a,27),
                                            (U)_mm256_extract_epi8(b,27)),
                                (U)std::pow((U)_mm256_extract_epi8(a,28),
                                            (U)_mm256_extract_epi8(b,28)),
                                (U)std::pow((U)_mm256_extract_epi8(a,29),
                                            (U)_mm256_extract_epi8(b,29)),
                                (U)std::pow((U)_mm256_extract_epi8(a,30),
                                            (U)_mm256_extract_epi8(b,30)),
                                (U)std::pow((U)_mm256_extract_epi8(a,31),
                                            (U)_mm256_extract_epi8(b,31)));
    }

    static __m256i negate(__m256i a)
    {
#ifdef __AVX2__
        return _mm256_sub_epi8(_mm256_setzero_si256(), a);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i na0 = _mm_sub_epi8(_mm_setzero_si128(), a0);
        __m128i na1 = _mm_sub_epi8(_mm_setzero_si128(), a1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(na0), na1, 1);
#endif
    }

    static __m256i exp(__m256i a)
    {
        return _mm256_setr_epi8((U)std::exp((U)_mm256_extract_epi8(a, 0)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 1)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 2)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 3)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 4)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 5)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 6)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 7)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 8)),
                                (U)std::exp((U)_mm256_extract_epi8(a, 9)),
                                (U)std::exp((U)_mm256_extract_epi8(a,10)),
                                (U)std::exp((U)_mm256_extract_epi8(a,11)),
                                (U)std::exp((U)_mm256_extract_epi8(a,12)),
                                (U)std::exp((U)_mm256_extract_epi8(a,13)),
                                (U)std::exp((U)_mm256_extract_epi8(a,14)),
                                (U)std::exp((U)_mm256_extract_epi8(a,15)),
                                (U)std::exp((U)_mm256_extract_epi8(a,16)),
                                (U)std::exp((U)_mm256_extract_epi8(a,17)),
                                (U)std::exp((U)_mm256_extract_epi8(a,18)),
                                (U)std::exp((U)_mm256_extract_epi8(a,19)),
                                (U)std::exp((U)_mm256_extract_epi8(a,20)),
                                (U)std::exp((U)_mm256_extract_epi8(a,21)),
                                (U)std::exp((U)_mm256_extract_epi8(a,22)),
                                (U)std::exp((U)_mm256_extract_epi8(a,23)),
                                (U)std::exp((U)_mm256_extract_epi8(a,24)),
                                (U)std::exp((U)_mm256_extract_epi8(a,25)),
                                (U)std::exp((U)_mm256_extract_epi8(a,26)),
                                (U)std::exp((U)_mm256_extract_epi8(a,27)),
                                (U)std::exp((U)_mm256_extract_epi8(a,28)),
                                (U)std::exp((U)_mm256_extract_epi8(a,29)),
                                (U)std::exp((U)_mm256_extract_epi8(a,30)),
                                (U)std::exp((U)_mm256_extract_epi8(a,31)));
    }

    static __m256i sqrt(__m256i a)
    {
        return _mm256_setr_epi8((U)std::sqrt((U)_mm256_extract_epi8(a, 0)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 1)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 2)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 3)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 4)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 5)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 6)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 7)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 8)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a, 9)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,10)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,11)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,12)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,13)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,14)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,15)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,16)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,17)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,18)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,19)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,20)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,21)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,22)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,23)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,24)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,25)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,26)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,27)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,28)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,29)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,30)),
                                (U)std::sqrt((U)_mm256_extract_epi8(a,31)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int16_t>::value ||
                                            std::is_same<U,uint16_t>::value>>
{
    static constexpr int vector_width = 16;
    static constexpr size_t alignment = 32;
    typedef __m256i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_pd(_mm256_castsi256_si128(convert<int32_t>(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256i v)
    {
        __m128i tmp = _mm_shuffle_epi32(_mm256_castsi256_si128(v), _MM_SHUFFLE(3,1,2,0));
        __m128i dup = _mm_shufflehi_epi16(_mm_shufflelo_epi16(tmp, _MM_SHUFFLE(1,0,1,0)),
                                                                   _MM_SHUFFLE(1,0,1,0));
        return _mm256_unpacklo_ps(convert<float>(_mm256_castsi128_si256(dup)), _mm256_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256i v)
    {
        __m128i dup = _mm_shufflelo_epi16(_mm256_castsi256_si128(v), _MM_SHUFFLE(1,1,0,0));
        return _mm256_unpacklo_pd(convert<double>(_mm256_castsi128_si256(dup)), _mm256_setzero_pd());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi16(v, v), _MM_SHUFFLE(2,0,2,0))
                                        : _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), _MM_SHUFFLE(2,0,2,0));
#else
        __m128i lo16 = _mm256_castsi256_si128(v);
        __m128i hi16 = _mm256_extractf128_si256(v, 1);
        __m128i lo = std::is_signed<U>::value ? _mm_packs_epi16(lo16, hi16)
                                              : _mm_packus_epi16(lo16, hi16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), lo, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v));
#else
        __m128i lo16 = _mm256_castsi256_si128(v);
        __m128i hi16 = _mm_shuffle_epi32(lo16, _MM_SHUFFLE(1,0,3,2));
        __m128i lo32 = std::is_signed<U>::value ? _mm_cvtepi16_epi32(lo16)
                                                : _mm_cvtepu16_epi32(lo16);
        __m128i hi32 = std::is_signed<U>::value ? _mm_cvtepi16_epi32(hi16)
                                                : _mm_cvtepu16_epi32(hi16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo32), hi32, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi16_epi64(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu16_epi64(_mm256_castsi256_si128(v));
#else
        __m128i lo32 = std::is_signed<U>::value ? _mm_cvtepi16_epi32(_mm256_castsi256_si128(v))
                                                : _mm_cvtepu16_epi32(_mm256_castsi256_si128(v));
        __m128i hi32 = _mm_shuffle_epi32(lo32, _MM_SHUFFLE(1,0,3,2));
        __m128i lo64 =  std::is_signed<U>::value ? _mm_cvtepi32_epi64(lo32)
                                                 : _mm_cvtepu32_epi64(lo32);
        __m128i hi64 =  std::is_signed<U>::value ? _mm_cvtepi32_epi64(hi32)
                                                 : _mm_cvtepu32_epi64(hi32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo64), hi64, 1);
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_loadu_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_load_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi64x(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi32(*(int32_t*)ptr);
    }

    static __m256i load1(const U* ptr)
    {
        return _mm256_set1_epi16(*ptr);
    }

    static __m256i set1(U val)
    {
        return _mm256_set1_epi16(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m256i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m256i v, U* ptr)
    {
        *(int32_t*)ptr = _mm256_extract_epi32(v, 0);
    }

    static __m256i add(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_add_epi16(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_add_epi16(a0, b0);
        __m128i ab1 = _mm_add_epi16(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i sub(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_sub_epi16(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_sub_epi16(a0, b0);
        __m128i ab1 = _mm_sub_epi16(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i mul(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_mullo_epi16(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_mullo_epi16(a0, b0);
        __m128i ab1 = _mm_mullo_epi16(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i div(__m256i a, __m256i b)
    {
        return _mm256_setr_epi16((U)_mm256_extract_epi16(a, 0) /
                                 (U)_mm256_extract_epi16(b, 0),
                                 (U)_mm256_extract_epi16(a, 1) /
                                 (U)_mm256_extract_epi16(b, 1),
                                 (U)_mm256_extract_epi16(a, 2) /
                                 (U)_mm256_extract_epi16(b, 2),
                                 (U)_mm256_extract_epi16(a, 3) /
                                 (U)_mm256_extract_epi16(b, 3),
                                 (U)_mm256_extract_epi16(a, 4) /
                                 (U)_mm256_extract_epi16(b, 4),
                                 (U)_mm256_extract_epi16(a, 5) /
                                 (U)_mm256_extract_epi16(b, 5),
                                 (U)_mm256_extract_epi16(a, 6) /
                                 (U)_mm256_extract_epi16(b, 6),
                                 (U)_mm256_extract_epi16(a, 7) /
                                 (U)_mm256_extract_epi16(b, 7),
                                 (U)_mm256_extract_epi16(a, 8) /
                                 (U)_mm256_extract_epi16(b, 8),
                                 (U)_mm256_extract_epi16(a, 9) /
                                 (U)_mm256_extract_epi16(b, 9),
                                 (U)_mm256_extract_epi16(a,10) /
                                 (U)_mm256_extract_epi16(b,10),
                                 (U)_mm256_extract_epi16(a,11) /
                                 (U)_mm256_extract_epi16(b,11),
                                 (U)_mm256_extract_epi16(a,12) /
                                 (U)_mm256_extract_epi16(b,12),
                                 (U)_mm256_extract_epi16(a,13) /
                                 (U)_mm256_extract_epi16(b,13),
                                 (U)_mm256_extract_epi16(a,14) /
                                 (U)_mm256_extract_epi16(b,14),
                                 (U)_mm256_extract_epi16(a,15) /
                                 (U)_mm256_extract_epi16(b,15));
    }

    static __m256i pow(__m256i a, __m256i b)
    {
        return _mm256_setr_epi16((U)std::pow((U)_mm256_extract_epi16(a, 0),
                                             (U)_mm256_extract_epi16(b, 0)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 1),
                                             (U)_mm256_extract_epi16(b, 1)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 2),
                                             (U)_mm256_extract_epi16(b, 2)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 3),
                                             (U)_mm256_extract_epi16(b, 3)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 4),
                                             (U)_mm256_extract_epi16(b, 4)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 5),
                                             (U)_mm256_extract_epi16(b, 5)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 6),
                                             (U)_mm256_extract_epi16(b, 6)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 7),
                                             (U)_mm256_extract_epi16(b, 7)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 8),
                                             (U)_mm256_extract_epi16(b, 8)),
                                 (U)std::pow((U)_mm256_extract_epi16(a, 9),
                                             (U)_mm256_extract_epi16(b, 9)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,10),
                                             (U)_mm256_extract_epi16(b,10)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,11),
                                             (U)_mm256_extract_epi16(b,11)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,12),
                                             (U)_mm256_extract_epi16(b,12)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,13),
                                             (U)_mm256_extract_epi16(b,13)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,14),
                                             (U)_mm256_extract_epi16(b,14)),
                                 (U)std::pow((U)_mm256_extract_epi16(a,15),
                                             (U)_mm256_extract_epi16(b,15)));
    }

    static __m256i negate(__m256i a)
    {
#ifdef __AVX2__
        return _mm256_sub_epi16(_mm256_setzero_si256(), a);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i na0 = _mm_sub_epi16(_mm_setzero_si128(), a0);
        __m128i na1 = _mm_sub_epi16(_mm_setzero_si128(), a1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(na0), na1, 1);
#endif
    }

    static __m256i exp(__m256i a)
    {
        return _mm256_setr_epi16((U)std::exp((U)_mm256_extract_epi16(a, 0)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 1)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 2)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 3)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 4)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 5)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 6)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 7)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 8)),
                                 (U)std::exp((U)_mm256_extract_epi16(a, 9)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,10)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,11)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,12)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,13)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,14)),
                                 (U)std::exp((U)_mm256_extract_epi16(a,15)));
    }

    static __m256i sqrt(__m256i a)
    {
        return _mm256_setr_epi16((U)std::sqrt((U)_mm256_extract_epi16(a, 0)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 1)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 2)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 3)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 4)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 5)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 6)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 7)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 8)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a, 9)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,10)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,11)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,12)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,13)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,14)),
                                 (U)std::sqrt((U)_mm256_extract_epi16(a,15)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int32_t>::value ||
                                            std::is_same<U,uint32_t>::value>>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 32;
    typedef __m256i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_ps(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256i v)
    {
        return _mm256_cvtepi32_pd(_mm256_castsi256_si128(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256i v)
    {
        __m256 sp = convert<float>(v);
        __m128 lo = _mm256_extractf128_ps(sp, 0);
        __m256 dup = _mm256_insertf128_ps(_mm256_permute_ps(sp, _MM_SHUFFLE(1,0,1,0)),
                                          _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(3,2,3,2)), 1);
        return _mm256_unpacklo_ps(dup, _mm256_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256i v)
    {
        __m128i dup = _mm_shuffle_epi32(_mm256_castsi256_si128(v), _MM_SHUFFLE(1,1,0,0));
        return _mm256_unpacklo_pd(_mm256_cvtepi32_pd(dup), _mm256_setzero_pd());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        __m256i i16 = std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi32(v, v), _MM_SHUFFLE(2,0,2,0))
                                               : _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), _MM_SHUFFLE(2,0,2,0));
        return std::is_signed<U>::value ? _mm256_packs_epi16(i16, i16)
                                        : _mm256_packus_epi16(i16, i16);
#else
        __m128i lo32 = _mm256_castsi256_si128(v);
        __m128i hi32 = _mm256_extractf128_si256(v, 1);
        __m128i i16 = std::is_signed<U>::value ? _mm_packs_epi32(lo32, hi32)
                                               : _mm_packus_epi32(lo32, hi32);
        __m128i lo = std::is_signed<U>::value ? _mm_packs_epi16(i16, i16)
                                              : _mm_packus_epi16(i16, i16);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), lo, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_permute4x64_epi64(_mm256_packs_epi32(v, v), _MM_SHUFFLE(2,0,2,0))
                                        : _mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), _MM_SHUFFLE(2,0,2,0));
#else
        __m128i lo32 = _mm256_castsi256_si128(v);
        __m128i hi32 = _mm256_extractf128_si256(v, 1);
        __m128i lo = std::is_signed<U>::value ? _mm_packs_epi32(lo32, hi32)
                                              : _mm_packus_epi32(lo32, hi32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), lo, 1);
#endif
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256i v)
    {
#ifdef __AVX2__
        return std::is_signed<U>::value ? _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v))
                                        : _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v));
#else
        __m128i lo32 = _mm256_castsi256_si128(v);
        __m128i hi32 = _mm_shuffle_epi32(lo32, _MM_SHUFFLE(1,0,3,2));
        __m128i lo64 =  std::is_signed<U>::value ? _mm_cvtepi32_epi64(lo32)
                                                 : _mm_cvtepu32_epi64(lo32);
        __m128i hi64 =  std::is_signed<U>::value ? _mm_cvtepi32_epi64(hi32)
                                                 : _mm_cvtepu32_epi64(hi32);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo64), hi64, 1);
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_loadu_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_load_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m256i>
    load(const U* ptr)
    {
        return _mm256_set1_epi64x(*(int64_t*)ptr);
    }

    static __m256i load1(const U* ptr)
    {
        return _mm256_set1_epi32(*ptr);
    }

    static __m256i set1(U val)
    {
        return _mm256_set1_epi32(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m256i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    static __m256i add(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_add_epi32(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_add_epi32(a0, b0);
        __m128i ab1 = _mm_add_epi32(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i sub(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_sub_epi32(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_sub_epi32(a0, b0);
        __m128i ab1 = _mm_sub_epi32(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i mul(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_mullo_epi32(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_mullo_epi32(a0, b0);
        __m128i ab1 = _mm_mullo_epi32(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i div(__m256i a, __m256i b)
    {
        return _mm256_setr_epi32((U)_mm256_extract_epi32(a, 0) /
                                 (U)_mm256_extract_epi32(b, 0),
                                 (U)_mm256_extract_epi32(a, 1) /
                                 (U)_mm256_extract_epi32(b, 1),
                                 (U)_mm256_extract_epi32(a, 2) /
                                 (U)_mm256_extract_epi32(b, 2),
                                 (U)_mm256_extract_epi32(a, 3) /
                                 (U)_mm256_extract_epi32(b, 3),
                                 (U)_mm256_extract_epi32(a, 4) /
                                 (U)_mm256_extract_epi32(b, 4),
                                 (U)_mm256_extract_epi32(a, 5) /
                                 (U)_mm256_extract_epi32(b, 5),
                                 (U)_mm256_extract_epi32(a, 6) /
                                 (U)_mm256_extract_epi32(b, 6),
                                 (U)_mm256_extract_epi32(a, 7) /
                                 (U)_mm256_extract_epi32(b, 7));
    }

    static __m256i pow(__m256i a, __m256i b)
    {
        return _mm256_setr_epi32((U)std::pow((U)_mm256_extract_epi32(a, 0),
                                             (U)_mm256_extract_epi32(b, 0)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 1),
                                             (U)_mm256_extract_epi32(b, 1)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 2),
                                             (U)_mm256_extract_epi32(b, 2)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 3),
                                             (U)_mm256_extract_epi32(b, 3)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 4),
                                             (U)_mm256_extract_epi32(b, 4)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 5),
                                             (U)_mm256_extract_epi32(b, 5)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 6),
                                             (U)_mm256_extract_epi32(b, 6)),
                                 (U)std::pow((U)_mm256_extract_epi32(a, 7),
                                             (U)_mm256_extract_epi32(b, 7)));
    }

    static __m256i negate(__m256i a)
    {
#ifdef __AVX2__
        return _mm256_sub_epi32(_mm256_setzero_si256(), a);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i na0 = _mm_sub_epi32(_mm_setzero_si128(), a0);
        __m128i na1 = _mm_sub_epi32(_mm_setzero_si128(), a1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(na0), na1, 1);
#endif
    }

    static __m256i exp(__m256i a)
    {
        return _mm256_setr_epi32((U)std::exp((U)_mm256_extract_epi32(a, 0)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 1)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 2)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 3)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 4)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 5)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 6)),
                                 (U)std::exp((U)_mm256_extract_epi32(a, 7)));
    }

    static __m256i sqrt(__m256i a)
    {
        return _mm256_setr_epi32((U)std::sqrt((U)_mm256_extract_epi32(a, 0)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 1)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 2)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 3)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 4)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 5)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 6)),
                                 (U)std::sqrt((U)_mm256_extract_epi32(a, 7)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int64_t>::value ||
                                            std::is_same<U,uint64_t>::value>>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 32;
    typedef __m256i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m256>
    convert(__m256i v)
    {
        float a = (U)_mm256_extract_epi64(v, 0);
        float b = (U)_mm256_extract_epi64(v, 1);
        float c = (U)_mm256_extract_epi64(v, 2);
        float d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_ps(a, b, c, d, a, b, c, d);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m256d>
    convert(__m256i v)
    {
        double a = (U)_mm256_extract_epi64(v, 0);
        double b = (U)_mm256_extract_epi64(v, 1);
        double c = (U)_mm256_extract_epi64(v, 2);
        double d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_pd(a, b, c, d);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m256>
    convert(__m256i v)
    {
        float a = (U)_mm256_extract_epi64(v, 0);
        float b = (U)_mm256_extract_epi64(v, 1);
        float c = (U)_mm256_extract_epi64(v, 2);
        float d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_ps(a, 0, b, 0, c, 0, d, 0);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<double>>::value, __m256d>
    convert(__m256i v)
    {
        double a = (U)_mm256_extract_epi64(v, 0);
        double b = (U)_mm256_extract_epi64(v, 1);
        return _mm256_setr_pd(a, 0, b, 0);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m256i>
    convert(__m256i v)
    {
        T a = (U)_mm256_extract_epi64(v, 0);
        T b = (U)_mm256_extract_epi64(v, 1);
        T c = (U)_mm256_extract_epi64(v, 2);
        T d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_epi8(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d,
                                a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m256i>
    convert(__m256i v)
    {
        T a = (U)_mm256_extract_epi64(v, 0);
        T b = (U)_mm256_extract_epi64(v, 1);
        T c = (U)_mm256_extract_epi64(v, 2);
        T d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_epi16(a, b, c, d, a, b, c, d, a, b, c, d, a, b, c, d);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m256i>
    convert(__m256i v)
    {
        T a = (U)_mm256_extract_epi64(v, 0);
        T b = (U)_mm256_extract_epi64(v, 1);
        T c = (U)_mm256_extract_epi64(v, 2);
        T d = (U)_mm256_extract_epi64(v, 3);
        return _mm256_setr_epi32(a, b, c, d, a, b, c, d);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m256i>
    convert(__m256i v)
    {
        return v;
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_loadu_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m256i>
    load(const U* ptr)
    {
        return _mm256_load_si256((__m256i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m256i>
    load(const U* ptr)
    {
#ifdef __AVX2__
        return _mm256_broadcastsi128_si256(_mm_load_si128((__m128i*)ptr));
#else
        return _mm256_castps_si256(_mm256_broadcast_ps((__m128*)ptr));
#endif
    }

    static __m256i load1(const U* ptr)
    {
        return _mm256_set1_epi64x(*ptr);
    }

    static __m256i set1(U val)
    {
        return _mm256_set1_epi64x(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_storeu_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm256_store_si256((__m256i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m256i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, _mm256_castsi256_si128(v));
    }

    static __m256i add(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_add_epi64(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_add_epi64(a0, b0);
        __m128i ab1 = _mm_add_epi64(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i sub(__m256i a, __m256i b)
    {
#ifdef __AVX2__
        return _mm256_sub_epi64(a, b);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i b0 = _mm256_castsi256_si128(b);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i b1 = _mm256_extractf128_si256(b, 1);
        __m128i ab0 = _mm_sub_epi64(a0, b0);
        __m128i ab1 = _mm_sub_epi64(a1, b1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(ab0), ab1, 1);
#endif
    }

    static __m256i mul(__m256i a, __m256i b)
    {
        return _mm256_setr_epi64x((U)_mm256_extract_epi64(a, 0) *
                                  (U)_mm256_extract_epi64(b, 0),
                                  (U)_mm256_extract_epi64(a, 1) *
                                  (U)_mm256_extract_epi64(b, 1),
                                  (U)_mm256_extract_epi64(a, 2) *
                                  (U)_mm256_extract_epi64(b, 2),
                                  (U)_mm256_extract_epi64(a, 3) *
                                  (U)_mm256_extract_epi64(b, 3));
    }

    static __m256i div(__m256i a, __m256i b)
    {
        return _mm256_setr_epi64x((U)_mm256_extract_epi64(a, 0) /
                                  (U)_mm256_extract_epi64(b, 0),
                                  (U)_mm256_extract_epi64(a, 1) /
                                  (U)_mm256_extract_epi64(b, 1),
                                  (U)_mm256_extract_epi64(a, 2) /
                                  (U)_mm256_extract_epi64(b, 2),
                                  (U)_mm256_extract_epi64(a, 3) /
                                  (U)_mm256_extract_epi64(b, 3));
    }

    static __m256i pow(__m256i a, __m256i b)
    {
        return _mm256_setr_epi64x((U)std::pow((U)_mm256_extract_epi64(a, 0),
                                              (U)_mm256_extract_epi64(b, 0)),
                                  (U)std::pow((U)_mm256_extract_epi64(a, 1),
                                              (U)_mm256_extract_epi64(b, 1)),
                                  (U)std::pow((U)_mm256_extract_epi64(a, 2),
                                              (U)_mm256_extract_epi64(b, 2)),
                                  (U)std::pow((U)_mm256_extract_epi64(a, 3),
                                              (U)_mm256_extract_epi64(b, 3)));
    }

    static __m256i negate(__m256i a)
    {
#ifdef __AVX2__
        return _mm256_sub_epi64(_mm256_setzero_si256(), a);
#else
        __m128i a0 = _mm256_castsi256_si128(a);
        __m128i a1 = _mm256_extractf128_si256(a, 1);
        __m128i na0 = _mm_sub_epi64(_mm_setzero_si128(), a0);
        __m128i na1 = _mm_sub_epi64(_mm_setzero_si128(), a1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(na0), na1, 1);
#endif
    }

    static __m256i exp(__m256i a)
    {
        return _mm256_setr_epi64x((U)std::exp((U)_mm256_extract_epi64(a, 0)),
                                  (U)std::exp((U)_mm256_extract_epi64(a, 1)),
                                  (U)std::exp((U)_mm256_extract_epi64(a, 2)),
                                  (U)std::exp((U)_mm256_extract_epi64(a, 3)));
    }

    static __m256i sqrt(__m256i a)
    {
        return _mm256_setr_epi64x((U)std::sqrt((U)_mm256_extract_epi64(a, 0)),
                                  (U)std::sqrt((U)_mm256_extract_epi64(a, 1)),
                                  (U)std::sqrt((U)_mm256_extract_epi64(a, 2)),
                                  (U)std::sqrt((U)_mm256_extract_epi64(a, 3)));
    }
};

}

#endif

#endif //MARRAY_VECTOR_AVX_HPP
