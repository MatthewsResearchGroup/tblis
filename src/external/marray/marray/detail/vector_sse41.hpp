#ifndef MARRAY_VECTOR_SSE41_HPP
#define MARRAY_VECTOR_SSE41_HPP

#if defined(__SSE4_1__)

#include <emmintrin.h>
#include <x86intrin.h>

#include "vector.hpp"

namespace MArray
{

template <>
struct vector_traits<float>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 16;
    typedef __m128 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128 v)
    {
        return _mm_cvtps_pd(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128 v)
    {
        return _mm_unpacklo_ps(v, _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(v);
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm_packs_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(v);
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm_packus_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(v);
        return _mm_packs_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(v);
        return _mm_packus_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_cvtps_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
                              (uint32_t)v[2], (uint32_t)v[3]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_set_epi64x((T)v[1], (T)v[0]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m128>
    load(const float* ptr)
    {
        return _mm_loadu_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m128>
    load(const float* ptr)
    {
        return _mm_load_ps(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m128>
    load(const float* ptr)
    {
        return _mm_castpd_ps(_mm_load1_pd((double*)ptr));
    }

    static __m128 load1(const float* ptr)
    {
        return _mm_load1_ps(ptr);
    }

    static __m128 set1(float val)
    {
        return _mm_set1_ps(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m128 v, float* ptr)
    {
        _mm_storeu_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m128 v, float* ptr)
    {
        _mm_store_ps(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m128 v, float* ptr)
    {
        _mm_store_sd((double*)ptr, _mm_castps_pd(v));
    }

    static __m128 add(__m128 a, __m128 b)
    {
        return _mm_add_ps(a, b);
    }

    static __m128 sub(__m128 a, __m128 b)
    {
        return _mm_sub_ps(a, b);
    }

    static __m128 mul(__m128 a, __m128 b)
    {
        return _mm_mul_ps(a, b);
    }

    static __m128 div(__m128 a, __m128 b)
    {
        return _mm_div_ps(a, b);
    }

    static __m128 pow(__m128 a, __m128 b)
    {
        return _mm_setr_ps(std::pow((float)a[0], (float)b[0]),
                           std::pow((float)a[1], (float)b[1]),
                           std::pow((float)a[2], (float)b[2]),
                           std::pow((float)a[3], (float)b[3]));
    }

    static __m128 negate(__m128 a)
    {
        return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
    }

    static __m128 exp(__m128 a)
    {
        return _mm_setr_ps(std::exp((float)a[0]),
                           std::exp((float)a[1]),
                           std::exp((float)a[2]),
                           std::exp((float)a[3]));
    }

    static __m128 log(__m128 a)
    {
        return _mm_setr_ps(std::log((float)a[0]),
                           std::log((float)a[1]),
                           std::log((float)a[2]),
                           std::log((float)a[3]));
    }

    static __m128 abs(__m128 a)
    {
        return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
    }

    static __m128 sqrt(__m128 a)
    {
        return _mm_sqrt_ps(a);
    }
};

template <>
struct vector_traits<double>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef __m128d vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128d v)
    {
        return _mm_cvtpd_ps(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128d v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128d v)
    {
        return _mm_unpacklo_ps(_mm_cvtpd_ps(v), _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m128i>
    convert(__m128d v)
    {
        __m128i i32 = _mm_cvtpd_epi32(v);
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm_packs_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128d v)
    {
        __m128i i32 = _mm_cvtpd_epi32(v);
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm_packus_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m128i>
    convert(__m128d v)
    {
        __m128i i32 = _mm_cvtpd_epi32(v);
        return _mm_packs_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128d v)
    {
        __m128i i32 = _mm_cvtpd_epi32(v);
        return _mm_packus_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m128i>
    convert(__m128d v)
    {
        return _mm_cvtpd_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128d v)
    {
        return _mm_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
                              (uint32_t)v[0], (uint32_t)v[1]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128d v)
    {
        return _mm_set_epi64x((T)v[1], (T)v[0]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m128d>
    load(const double* ptr)
    {
        return _mm_loadu_pd(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m128d>
    load(const double* ptr)
    {
        return _mm_load_pd(ptr);
    }

    static __m128d load1(const double* ptr)
    {
        return _mm_load1_pd(ptr);
    }

    static __m128d set1(double val)
    {
        return _mm_set1_pd(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m128d v, double* ptr)
    {
        _mm_storeu_pd(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m128d v, double* ptr)
    {
        _mm_store_pd(ptr, v);
    }

    static __m128d add(__m128d a, __m128d b)
    {
        return _mm_add_pd(a, b);
    }

    static __m128d sub(__m128d a, __m128d b)
    {
        return _mm_sub_pd(a, b);
    }

    static __m128d mul(__m128d a, __m128d b)
    {
        return _mm_mul_pd(a, b);
    }

    static __m128d div(__m128d a, __m128d b)
    {
        return _mm_div_pd(a, b);
    }

    static __m128d pow(__m128d a, __m128d b)
    {
        return _mm_setr_pd(std::pow((double)a[0], (double)b[0]),
                          std::pow((double)a[1], (double)b[1]));
    }

    static __m128d negate(__m128d a)
    {
        return _mm_xor_pd(a, _mm_set1_pd(-0.0));
    }

    static __m128d exp(__m128d a)
    {
        return _mm_setr_pd(std::exp((double)a[0]),
                           std::exp((double)a[1]));
    }

    static __m128d log(__m128d a)
    {
        return _mm_setr_pd(std::log((double)a[0]),
                           std::log((double)a[1]));
    }

    static __m128d abs(__m128d a)
    {
        return _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFull)));
    }

    static __m128d sqrt(__m128d a)
    {
        return _mm_sqrt_pd(a);
    }
};

template <>
struct vector_traits<std::complex<float>>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef __m128 vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128 v)
    {
        return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,0,2,0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128 v)
    {
        return _mm_cvtps_pd(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128 v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(convert<float>(v));
        __m128i i16 = _mm_packs_epi32(i32, i32);
        return _mm_packs_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(convert<float>(v));
        __m128i i16 = _mm_packus_epi32(i32, i32);
        return _mm_packus_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(convert<float>(v));
        return _mm_packs_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128 v)
    {
        __m128i i32 = _mm_cvtps_epi32(convert<float>(v));
        return _mm_packus_epi32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_cvtps_epi32(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_setr_epi32((uint32_t)v[0], (uint32_t)v[2],
                              (uint32_t)v[0], (uint32_t)v[2]);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128 v)
    {
        return _mm_set_epi64x((T)v[2], (T)v[0]);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m128>
    load(const std::complex<float>* ptr)
    {
        return _mm_loadu_ps((float*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m128>
    load(const std::complex<float>* ptr)
    {
        return _mm_load_ps((float*)ptr);
    }

    static __m128 load1(const std::complex<float>* ptr)
    {
        return _mm_castpd_ps(_mm_load1_pd((double*)ptr));
    }

    static __m128 set1(std::complex<float> val)
    {
        return _mm_castpd_ps(_mm_set1_pd(*(double*)&val));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m128 v, std::complex<float>* ptr)
    {
        _mm_storeu_ps((float*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m128 v, std::complex<float>* ptr)
    {
        _mm_store_ps((float*)ptr, v);
    }

    static __m128 add(__m128 a, __m128 b)
    {
        return _mm_add_ps(a, b);
    }

    static __m128 sub(__m128 a, __m128 b)
    {
        return _mm_sub_ps(a, b);
    }

    static __m128 mul(__m128 a, __m128 b)
    {
        __m128 ashuf = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,3,0,1));
        __m128 breal = _mm_moveldup_ps(b);
        __m128 bimag = _mm_movehdup_ps(b);
        __m128 tmp1 = _mm_mul_ps(    a, breal); // tmp1 = (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
        __m128 tmp2 = _mm_mul_ps(ashuf, bimag); // tmp2 = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm_addsub_ps(tmp1, tmp2);
    }

    static __m128 div(__m128 a, __m128 b)
    {
        std::complex<float> a0{(float)a[0], (float)a[1]};
        std::complex<float> a1{(float)a[2], (float)a[3]};
        std::complex<float> b0{(float)b[0], (float)b[1]};
        std::complex<float> b1{(float)b[2], (float)b[3]};
        std::complex<float> c0 = a0 / b0;
        std::complex<float> c1 = a1 / b1;
        return _mm_setr_ps(c0.real(), c0.imag(),
                           c1.real(), c1.imag());
    }

    static __m128 pow(__m128 a, __m128 b)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0((float)b[0], (float)b[1]);
        std::complex<float> b1((float)b[2], (float)b[3]);
        std::complex<float> c0 = std::pow(a0, b0);
        std::complex<float> c1 = std::pow(a1, b1);
        return _mm_setr_ps(c0.real(), c0.imag(),
                           c1.real(), c1.imag());
    }

    static __m128 negate(__m128 a)
    {
        return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
    }

    static __m128 exp(__m128 a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0 = std::exp(a0);
        std::complex<float> b1 = std::exp(a1);
        return _mm_setr_ps(b0.real(), b0.imag(),
                           b1.real(), b1.imag());
    }

    static __m128 sqrt(__m128 a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0 = std::sqrt(a0);
        std::complex<float> b1 = std::sqrt(a1);
        return _mm_setr_ps(b0.real(), b0.imag(),
                           b1.real(), b1.imag());
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int8_t>::value ||
                                         std::is_same<U,uint8_t>::value>>
{
    static constexpr int vector_width = 16;
    static constexpr size_t alignment = 16;
    typedef __m128i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128i v)
    {
        return _mm_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128i v)
    {
        return _mm_cvtepi32_pd(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128i v)
    {
        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi8_epi16(v)
                                        : _mm_cvtepu8_epi16(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi8_epi32(v)
                                        : _mm_cvtepu8_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi8_epi64(v)
                                        : _mm_cvtepu8_epi64(v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_loadu_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_load_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi64x(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi32(*(int32_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi16(*(int16_t*)ptr);
    }

    static __m128i load1(const U* ptr)
    {
        return _mm_set1_epi8(*ptr);
    }

    static __m128i set1(U val)
    {
        return _mm_set1_epi8(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && !Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16 && Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8>
    store(__m128i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m128i v, U* ptr)
    {
        *(int32_t*)ptr = _mm_extract_epi32(v, 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m128i v, U* ptr)
    {
        *(int16_t*)ptr = _mm_extract_epi16(v, 0);
    }

    static __m128i add(__m128i a, __m128i b)
    {
        return _mm_add_epi8(a, b);
    }

    static __m128i sub(__m128i a, __m128i b)
    {
        return _mm_sub_epi8(a, b);
    }

    static __m128i mul(__m128i a, __m128i b)
    {
        __m128i lo = _mm_and_si128(_mm_mullo_epi16(a, b), _mm_set1_epi16(0xff));
        __m128i hi = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
        return _mm_or_si128(_mm_slli_epi16(hi, 8), lo);
    }

    static __m128i div(__m128i a, __m128i b)
    {
        return _mm_setr_epi8((U)_mm_extract_epi8(a, 0) /
                            (U)_mm_extract_epi8(b, 0),
                            (U)_mm_extract_epi8(a, 1) /
                            (U)_mm_extract_epi8(b, 1),
                            (U)_mm_extract_epi8(a, 2) /
                            (U)_mm_extract_epi8(b, 2),
                            (U)_mm_extract_epi8(a, 3) /
                            (U)_mm_extract_epi8(b, 3),
                            (U)_mm_extract_epi8(a, 4) /
                            (U)_mm_extract_epi8(b, 4),
                            (U)_mm_extract_epi8(a, 5) /
                            (U)_mm_extract_epi8(b, 5),
                            (U)_mm_extract_epi8(a, 6) /
                            (U)_mm_extract_epi8(b, 6),
                            (U)_mm_extract_epi8(a, 7) /
                            (U)_mm_extract_epi8(b, 7),
                            (U)_mm_extract_epi8(a, 8) /
                            (U)_mm_extract_epi8(b, 8),
                            (U)_mm_extract_epi8(a, 9) /
                            (U)_mm_extract_epi8(b, 9),
                            (U)_mm_extract_epi8(a,10) /
                            (U)_mm_extract_epi8(b,10),
                            (U)_mm_extract_epi8(a,11) /
                            (U)_mm_extract_epi8(b,11),
                            (U)_mm_extract_epi8(a,12) /
                            (U)_mm_extract_epi8(b,12),
                            (U)_mm_extract_epi8(a,13) /
                            (U)_mm_extract_epi8(b,13),
                            (U)_mm_extract_epi8(a,14) /
                            (U)_mm_extract_epi8(b,14),
                            (U)_mm_extract_epi8(a,15) /
                            (U)_mm_extract_epi8(b,15));
    }

    static __m128i pow(__m128i a, __m128i b)
    {
        return _mm_setr_epi8((U)std::pow((U)_mm_extract_epi8(a, 0),
                                        (U)_mm_extract_epi8(b, 0)),
                            (U)std::pow((U)_mm_extract_epi8(a, 1),
                                        (U)_mm_extract_epi8(b, 1)),
                            (U)std::pow((U)_mm_extract_epi8(a, 2),
                                        (U)_mm_extract_epi8(b, 2)),
                            (U)std::pow((U)_mm_extract_epi8(a, 3),
                                        (U)_mm_extract_epi8(b, 3)),
                            (U)std::pow((U)_mm_extract_epi8(a, 4),
                                        (U)_mm_extract_epi8(b, 4)),
                            (U)std::pow((U)_mm_extract_epi8(a, 5),
                                        (U)_mm_extract_epi8(b, 5)),
                            (U)std::pow((U)_mm_extract_epi8(a, 6),
                                        (U)_mm_extract_epi8(b, 6)),
                            (U)std::pow((U)_mm_extract_epi8(a, 7),
                                        (U)_mm_extract_epi8(b, 7)),
                            (U)std::pow((U)_mm_extract_epi8(a, 8),
                                        (U)_mm_extract_epi8(b, 8)),
                            (U)std::pow((U)_mm_extract_epi8(a, 9),
                                        (U)_mm_extract_epi8(b, 9)),
                            (U)std::pow((U)_mm_extract_epi8(a,10),
                                        (U)_mm_extract_epi8(b,10)),
                            (U)std::pow((U)_mm_extract_epi8(a,11),
                                        (U)_mm_extract_epi8(b,11)),
                            (U)std::pow((U)_mm_extract_epi8(a,12),
                                        (U)_mm_extract_epi8(b,12)),
                            (U)std::pow((U)_mm_extract_epi8(a,13),
                                        (U)_mm_extract_epi8(b,13)),
                            (U)std::pow((U)_mm_extract_epi8(a,14),
                                        (U)_mm_extract_epi8(b,14)),
                            (U)std::pow((U)_mm_extract_epi8(a,15),
                                        (U)_mm_extract_epi8(b,15)));
    }

    static __m128i negate(__m128i a)
    {
        return _mm_sub_epi8(_mm_setzero_si128(), a);
    }

    static __m128i exp(__m128i a)
    {
        return _mm_setr_epi8((U)std::exp((U)_mm_extract_epi8(a, 0)),
                            (U)std::exp((U)_mm_extract_epi8(a, 1)),
                            (U)std::exp((U)_mm_extract_epi8(a, 2)),
                            (U)std::exp((U)_mm_extract_epi8(a, 3)),
                            (U)std::exp((U)_mm_extract_epi8(a, 4)),
                            (U)std::exp((U)_mm_extract_epi8(a, 5)),
                            (U)std::exp((U)_mm_extract_epi8(a, 6)),
                            (U)std::exp((U)_mm_extract_epi8(a, 7)),
                            (U)std::exp((U)_mm_extract_epi8(a, 8)),
                            (U)std::exp((U)_mm_extract_epi8(a, 9)),
                            (U)std::exp((U)_mm_extract_epi8(a,10)),
                            (U)std::exp((U)_mm_extract_epi8(a,11)),
                            (U)std::exp((U)_mm_extract_epi8(a,12)),
                            (U)std::exp((U)_mm_extract_epi8(a,13)),
                            (U)std::exp((U)_mm_extract_epi8(a,14)),
                            (U)std::exp((U)_mm_extract_epi8(a,15)));
    }

    static __m128i sqrt(__m128i a)
    {
        return _mm_setr_epi8((U)std::sqrt((U)_mm_extract_epi8(a, 0)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 1)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 2)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 3)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 4)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 5)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 6)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 7)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 8)),
                            (U)std::sqrt((U)_mm_extract_epi8(a, 9)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,10)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,11)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,12)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,13)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,14)),
                            (U)std::sqrt((U)_mm_extract_epi8(a,15)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int16_t>::value ||
                                            std::is_same<U,uint16_t>::value>>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 16;
    typedef __m128i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128i v)
    {
        return _mm_cvtepi32_ps(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128i v)
    {
        return _mm_cvtepi32_pd(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128i v)
    {
        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_packs_epi16(v, v)
                                        : _mm_packus_epi16(v, v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi16_epi32(v)
                                        : _mm_cvtepu16_epi32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi16_epi64(v)
                                        : _mm_cvtepu16_epi64(v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_loadu_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_load_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi64x(*(int64_t*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi32(*(int32_t*)ptr);
    }

    static __m128i load1(const U* ptr)
    {
        return _mm_set1_epi16(*ptr);
    }

    static __m128i set1(U val)
    {
        return _mm_set1_epi16(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && !Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8 && Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(__m128i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m128i v, U* ptr)
    {
        *(int32_t*)ptr = _mm_extract_epi32(v, 0);
    }

    static __m128i add(__m128i a, __m128i b)
    {
        return _mm_add_epi16(a, b);
    }

    static __m128i sub(__m128i a, __m128i b)
    {
        return _mm_sub_epi16(a, b);
    }

    static __m128i mul(__m128i a, __m128i b)
    {
        return _mm_mullo_epi16(a, b);
    }

    static __m128i div(__m128i a, __m128i b)
    {
        return _mm_setr_epi16((U)_mm_extract_epi16(a, 0) /
                             (U)_mm_extract_epi16(b, 0),
                             (U)_mm_extract_epi16(a, 1) /
                             (U)_mm_extract_epi16(b, 1),
                             (U)_mm_extract_epi16(a, 2) /
                             (U)_mm_extract_epi16(b, 2),
                             (U)_mm_extract_epi16(a, 3) /
                             (U)_mm_extract_epi16(b, 3),
                             (U)_mm_extract_epi16(a, 4) /
                             (U)_mm_extract_epi16(b, 4),
                             (U)_mm_extract_epi16(a, 5) /
                             (U)_mm_extract_epi16(b, 5),
                             (U)_mm_extract_epi16(a, 6) /
                             (U)_mm_extract_epi16(b, 6),
                             (U)_mm_extract_epi16(a, 7) /
                             (U)_mm_extract_epi16(b, 7));
    }

    static __m128i pow(__m128i a, __m128i b)
    {
        return _mm_setr_epi16((U)std::pow((U)_mm_extract_epi16(a, 0),
                                         (U)_mm_extract_epi16(b, 0)),
                             (U)std::pow((U)_mm_extract_epi16(a, 1),
                                         (U)_mm_extract_epi16(b, 1)),
                             (U)std::pow((U)_mm_extract_epi16(a, 2),
                                         (U)_mm_extract_epi16(b, 2)),
                             (U)std::pow((U)_mm_extract_epi16(a, 3),
                                         (U)_mm_extract_epi16(b, 3)),
                             (U)std::pow((U)_mm_extract_epi16(a, 4),
                                         (U)_mm_extract_epi16(b, 4)),
                             (U)std::pow((U)_mm_extract_epi16(a, 5),
                                         (U)_mm_extract_epi16(b, 5)),
                             (U)std::pow((U)_mm_extract_epi16(a, 6),
                                         (U)_mm_extract_epi16(b, 6)),
                             (U)std::pow((U)_mm_extract_epi16(a, 7),
                                         (U)_mm_extract_epi16(b, 7)));
    }

    static __m128i negate(__m128i a)
    {
        return _mm_sub_epi16(_mm_setzero_si128(), a);
    }

    static __m128i exp(__m128i a)
    {
        return _mm_setr_epi16((U)std::exp((U)_mm_extract_epi16(a, 0)),
                             (U)std::exp((U)_mm_extract_epi16(a, 1)),
                             (U)std::exp((U)_mm_extract_epi16(a, 2)),
                             (U)std::exp((U)_mm_extract_epi16(a, 3)),
                             (U)std::exp((U)_mm_extract_epi16(a, 4)),
                             (U)std::exp((U)_mm_extract_epi16(a, 5)),
                             (U)std::exp((U)_mm_extract_epi16(a, 6)),
                             (U)std::exp((U)_mm_extract_epi16(a, 7)));
    }

    static __m128i sqrt(__m128i a)
    {
        return _mm_setr_epi16((U)std::sqrt((U)_mm_extract_epi16(a, 0)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 1)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 2)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 3)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 4)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 5)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 6)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 7)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int32_t>::value ||
                                            std::is_same<U,uint32_t>::value>>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 16;
    typedef __m128i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128i v)
    {
        return _mm_cvtepi32_ps(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128i v)
    {
        return _mm_cvtepi32_pd(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128i v)
    {
        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128i v)
    {
        __m128i i16 = std::is_signed<U>::value ? _mm_packs_epi32(v, v)
                                               : _mm_packus_epi32(v, v);
        return std::is_signed<U>::value ? _mm_packs_epi16(i16, i16)
                                        : _mm_packus_epi16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_packs_epi32(v, v)
                                        : _mm_packus_epi32(v, v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128i v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128i v)
    {
        return std::is_signed<U>::value ? _mm_cvtepi32_epi64(v)
                                        : _mm_cvtepu32_epi64(v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_loadu_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_load_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, __m128i>
    load(const U* ptr)
    {
        return _mm_set1_epi64x(*(int64_t*)ptr);
    }

    static __m128i load1(const U* ptr)
    {
        return _mm_set1_epi32(*ptr);
    }

    static __m128i set1(U val)
    {
        return _mm_set1_epi32(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && !Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4 && Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(__m128i v, U* ptr)
    {
        _mm_storel_epi64((__m128i*)ptr, v);
    }

    static __m128i add(__m128i a, __m128i b)
    {
        return _mm_add_epi32(a, b);
    }

    static __m128i sub(__m128i a, __m128i b)
    {
        return _mm_sub_epi32(a, b);
    }

    static __m128i mul(__m128i a, __m128i b)
    {
        return _mm_mullo_epi32(a, b);
    }

    static __m128i div(__m128i a, __m128i b)
    {
        return _mm_setr_epi32((U)_mm_extract_epi32(a, 0) /
                              (U)_mm_extract_epi32(b, 0),
                              (U)_mm_extract_epi32(a, 1) /
                              (U)_mm_extract_epi32(b, 1),
                              (U)_mm_extract_epi32(a, 2) /
                              (U)_mm_extract_epi32(b, 2),
                              (U)_mm_extract_epi32(a, 3) /
                              (U)_mm_extract_epi32(b, 3));
    }

    static __m128i pow(__m128i a, __m128i b)
    {
        return _mm_setr_epi32((U)std::pow((U)_mm_extract_epi32(a, 0),
                                          (U)_mm_extract_epi32(b, 0)),
                              (U)std::pow((U)_mm_extract_epi32(a, 1),
                                          (U)_mm_extract_epi32(b, 1)),
                              (U)std::pow((U)_mm_extract_epi32(a, 2),
                                          (U)_mm_extract_epi32(b, 2)),
                              (U)std::pow((U)_mm_extract_epi32(a, 3),
                                          (U)_mm_extract_epi32(b, 3)));
    }

    static __m128i negate(__m128i a)
    {
        return _mm_sub_epi32(_mm_setzero_si128(), a);
    }

    static __m128i exp(__m128i a)
    {
        return _mm_setr_epi32((U)std::exp((U)_mm_extract_epi32(a, 0)),
                              (U)std::exp((U)_mm_extract_epi32(a, 1)),
                              (U)std::exp((U)_mm_extract_epi32(a, 2)),
                              (U)std::exp((U)_mm_extract_epi32(a, 3)));
    }

    static __m128i sqrt(__m128i a)
    {
        return _mm_setr_epi32((U)std::sqrt((U)_mm_extract_epi32(a, 0)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 1)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 2)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 3)));
    }
};


template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same<U,int64_t>::value ||
                                            std::is_same<U,uint64_t>::value>>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef __m128i vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same<T,float>::value, __m128>
    convert(__m128i v)
    {
        float a = (U)_mm_extract_epi64(v, 0);
        float b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_ps(a, b, a, b);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,double>::value, __m128d>
    convert(__m128i v)
    {
        double a = (U)_mm_extract_epi64(v, 0);
        double b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_pd(a, b);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, __m128>
    convert(__m128i v)
    {
        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, __m128i>
    convert(__m128i v)
    {
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi8(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, __m128i>
    convert(__m128i v)
    {
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi16(a, b, a, b, a, b, a, b);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, __m128i>
    convert(__m128i v)
    {
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi32(a, b, a, b);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, __m128i>
    convert(__m128i v)
    {
        return v;
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_loadu_si128((__m128i*)ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned, __m128i>
    load(const U* ptr)
    {
        return _mm_load_si128((__m128i*)ptr);
    }

    static __m128i load1(const U* ptr)
    {
        return _mm_set1_epi64x(*ptr);
    }

    static __m128i set1(U val)
    {
        return _mm_set1_epi64x(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && !Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_storeu_si128((__m128i*)ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2 && Aligned>
    store(__m128i v, U* ptr)
    {
        _mm_store_si128((__m128i*)ptr, v);
    }

    static __m128i add(__m128i a, __m128i b)
    {
        return _mm_add_epi64(a, b);
    }

    static __m128i sub(__m128i a, __m128i b)
    {
        return _mm_sub_epi64(a, b);
    }

    static __m128i mul(__m128i a, __m128i b)
    {
        return _mm_set_epi64x((U)_mm_extract_epi64(a, 1) *
                              (U)_mm_extract_epi64(b, 1),
                              (U)_mm_extract_epi64(a, 0) *
                              (U)_mm_extract_epi64(b, 0));
    }

    static __m128i div(__m128i a, __m128i b)
    {
        return _mm_set_epi64x((U)_mm_extract_epi64(a, 1) /
                              (U)_mm_extract_epi64(b, 1),
                              (U)_mm_extract_epi64(a, 0) /
                              (U)_mm_extract_epi64(b, 0));
    }

    static __m128i pow(__m128i a, __m128i b)
    {
        return _mm_set_epi64x((U)std::pow((U)_mm_extract_epi64(a, 1),
                                          (U)_mm_extract_epi64(b, 1)),
                              (U)std::pow((U)_mm_extract_epi64(a, 0),
                                          (U)_mm_extract_epi64(b, 0)));
    }

    static __m128i negate(__m128i a)
    {
        return _mm_sub_epi64(_mm_setzero_si128(), a);
    }

    static __m128i exp(__m128i a)
    {
        return _mm_set_epi64x((U)std::exp((U)_mm_extract_epi64(a, 1)),
                              (U)std::exp((U)_mm_extract_epi64(a, 0)));
    }

    static __m128i sqrt(__m128i a)
    {
        return _mm_set_epi64x((U)std::sqrt((U)_mm_extract_epi64(a, 1)),
                              (U)std::sqrt((U)_mm_extract_epi64(a, 0)));
    }
};

}

#endif

#endif //MARRAY_VECTOR_SSE41_HPP
