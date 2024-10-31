#ifndef _MARRAY_VECTOR_GENERIC_HPP_
#define _MARRAY_VECTOR_GENERIC_HPP_

#include "vector.hpp"

#include <cstdint>
#include <type_traits>

namespace MArray
{

template <typename T, typename=void>
struct vector_base {
};

template<>
struct vector_base<float> {
    inline static float result_value[4];
};

template<>
struct vector_base<double> {
    inline static double result_value[2];
};

template<>
struct vector_base<std::complex<float>> {
    inline static float result_value[4];
};

template<>
struct vector_base<int8_t> {
    using value_type = uint8_t;
    inline static value_type result_value[16];
};

template<>
struct vector_base<uint16_t> {
    using value_type = uint16_t;
    inline static value_type result_value[8];
};

template<>
struct vector_base<uint32_t> {
    using value_type = uint32_t;
    inline static value_type result_value[4];
};

template<>
struct vector_base<uint64_t> {
    using value_type = uint64_t;
    inline static value_type result_value[2];
};

template <>
struct vector_traits<float>
{
    constexpr static unsigned vector_width = 4;
    constexpr static size_t alignment = 64;

    using vector_type = float*;
    using value_type = float;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float*>
    convert(float* v)
    {
        return v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double*>
    convert(float* v)
    {
        return (double*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float*>
    convert(float* v)
    {
        return v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value, int8_t*>
    convert(float* v)
    {
        //ivector_type i32 = _mm_cvtps_epi32(v);
        //ivector_type i16 = _mm_packs_epi32(i32, i32);
        //return _mm_packs_epi16(i16, i16);
        return (int8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint8_t>::value, uint8_t*>
    convert(float* v)
    {
        return (uint8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value, int16_t*>
    convert(float* v)
    {
        return (int16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint16_t>::value, uint16_t*>
    convert(float* v)
    {
        return (uint16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value, int32_t*>
    convert(float* v)
    {
        return (int32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint32_t>::value, uint32_t*>
    convert(float* v)
    {
        return (uint32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T*>
    convert(float* v)
    {
        return (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && !Aligned, float*>
    load(const float* ptr)
    {
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1];
        vector_base<value_type>::result_value[2] = ptr[2];
        vector_base<value_type>::result_value[3] = ptr[3];
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && Aligned, float*>
    load(const float* ptr)
    {
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1];
        vector_base<value_type>::result_value[2] = ptr[2];
        vector_base<value_type>::result_value[3] = ptr[3];
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2, float*>
    load(const float* ptr)
    {
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1];
        vector_base<value_type>::result_value[2] = ptr[2];
        vector_base<value_type>::result_value[3] = ptr[3];
        return vector_base<value_type>::result_value;
    }

    static float* load1(const float* ptr)
    {
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1];
        vector_base<value_type>::result_value[2] = ptr[2];
        vector_base<value_type>::result_value[3] = ptr[3];
        return vector_base<value_type>::result_value;
    }

    static float* set1(float val)
    {
        vector_base<value_type>::result_value[0] = val;
        vector_base<value_type>::result_value[1] = val;
        vector_base<value_type>::result_value[2] = val;
        vector_base<value_type>::result_value[3] = val;
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && !Aligned>
    store(float* v, float* ptr)
    {
        //_mm_storeu_ps(ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr[3] = v[3];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && Aligned>
    store(float * v, float* ptr)
    {
        //_mm_store_ps(ptr, v);
        std::copy(v, v+4, ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2>
    store(double * v, float* ptr)
    {
        //_mm_store_sd((double*)ptr, _mm_castps_pd(v));
        std::copy(v, v+1, (double*)ptr);
    }

    static float* add(float* a, float* b)
    {
        //return _mm_add_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] + b[0];
        vector_base<value_type>::result_value[1] = a[1] + b[1];
        vector_base<value_type>::result_value[2] = a[2] + b[2];
        vector_base<value_type>::result_value[3] = a[3] + b[3];
        return vector_base<value_type>::result_value;
    }

    static float* sub(float* a, float* b)
    {
        //return _mm_sub_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] - b[0];
        vector_base<value_type>::result_value[1] = a[1] - b[1];
        vector_base<value_type>::result_value[2] = a[2] - b[2];
        vector_base<value_type>::result_value[3] = a[3] - b[3];
        return vector_base<value_type>::result_value;
    }

    static float* mul(float* a, float* b)
    {
        //return _mm_mul_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] * b[0];
        vector_base<value_type>::result_value[1] = a[1] * b[1];
        vector_base<value_type>::result_value[2] = a[2] * b[2];
        vector_base<value_type>::result_value[3] = a[3] * b[3];
        return vector_base<value_type>::result_value; 
    }

    static float* div(float* a, float* b)
    {
        //return _mm_div_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] / b[0];
        vector_base<value_type>::result_value[1] = a[1] / b[1];
        vector_base<value_type>::result_value[2] = a[2] / b[2];
        vector_base<value_type>::result_value[3] = a[3] / b[3];
        return vector_base<value_type>::result_value; 
    }

    static float* pow(float* a, float* b)
    {
        //return _mm_setr_ps(std::pow((float)a[0], (float)b[0]),
        //                   std::pow((float)a[1], (float)b[1]),
        //                   std::pow((float)a[2], (float)b[2]),
        //                   std::pow((float)a[3], (float)b[3]));
        vector_base<value_type>::result_value[0] = std::pow(a[0] , b[0]);
        vector_base<value_type>::result_value[1] = std::pow(a[1] , b[1]);
        vector_base<value_type>::result_value[2] = std::pow(a[2] , b[2]);
        vector_base<value_type>::result_value[3] = std::pow(a[3] , b[3]);
        return vector_base<value_type>::result_value; 
    }

    static float* negate(float* a)
    {
        //return _mm_xor_ps(a, _mm_set1_ps(-0.0f));
        vector_base<value_type>::result_value[0] = -a[0];
        vector_base<value_type>::result_value[1] = -a[1];
        vector_base<value_type>::result_value[2] = -a[2];
        vector_base<value_type>::result_value[3] = -a[3];
        return vector_base<value_type>::result_value; 
    }

    static float* exp(float* a)
    {
        //return _mm_setr_ps(std::exp((float)a[0]),
        //                   std::exp((float)a[1]),
        //                   std::exp((float)a[2]),
        //                   std::exp((float)a[3]));
        vector_base<value_type>::result_value[0] = std::exp(a[0]);
        vector_base<value_type>::result_value[1] = std::exp(a[1]);
        vector_base<value_type>::result_value[2] = std::exp(a[2]);
        vector_base<value_type>::result_value[3] = std::exp(a[3]);
        return vector_base<value_type>::result_value; 
    }

    static float* sqrt(float* a)
    {
        //return _mm_sqrt_ps(a);
        vector_base<value_type>::result_value[0] = std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = std::sqrt(a[1]);
        vector_base<value_type>::result_value[2] = std::sqrt(a[2]);
        vector_base<value_type>::result_value[3] = std::sqrt(a[3]);
        return vector_base<value_type>::result_value;
    }
};

template <>
struct vector_traits<double>
{
    constexpr static unsigned vector_width = 2;
    constexpr static size_t alignment = 64;
    using vector_type = double*;
    using value_type = double;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float*>
    convert(double* v)
    {
        //return _mm_cvtpd_ps(v);
        return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double*>
    convert(double* v)
    {
        return v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float*>
    convert(double* v)
    {
        //return _mm_unpacklo_ps(_mm_cvtpd_ps(v), _mm_setzero_ps());
        return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value, int8_t*>
    convert(double* v)
    {
        //ivector_type i32 = _mm_cvtpd_epi32(v);
        //ivector_type i16 = _mm_packs_epi32(i32, i32);
        //return _mm_packs_epi16(i16, i16);
        return (int8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint8_t>::value, uint8_t*>
    convert(double* v)
    {
        //ivector_type i32 = _mm_cvtpd_epi32(v);
        //ivector_type i16 = _mm_packus_epi32(i32, i32);
        //return _mm_packus_epi16(i16, i16);
        return (uint8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value, int16_t*>
    convert(double* v)
    {
        //ivector_type i32 = _mm_cvtpd_epi32(v);
        //return _mm_packs_epi32(i32, i32);
        return (int16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint16_t>::value, uint16_t*>
    convert(double* v)
    {
        //ivector_type i32 = _mm_cvtpd_epi32(v);
        //return _mm_packus_epi32(i32, i32);
        return (uint16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value, int32_t*>
    convert(double* v)
    {
        //return _mm_cvtpd_epi32(v);
        return (int32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint32_t>::value, uint32_t*>
    convert(double* v)
    {
        //return _mm_setr_epi32((uint32_t)v[0], (uint32_t)v[1],
        //                      (uint32_t)v[0], (uint32_t)v[1]);
        return (uint32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T*>
    convert(double* v)
    {
        //return _mm_set_epi64x((T)v[1], (T)v[0]);
        return (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned, double*>
    load(const double* ptr)
    {
        //return _mm_loadu_pd(ptr);
        return const_cast<double*>(ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned, double*>
    load(const double* ptr)
    {
        //return _mm_load_pd(ptr);
        return const_cast<double*>(ptr);
    }

    static double * load1(const double* ptr)
    {
        //return _mm_load1_pd(ptr);
        return const_cast<double*>(ptr);
    }

    static double* set1(double val)
    {
        //return _mm_set1_pd(val);
        vector_base<value_type>::result_value[0] = val;
        vector_base<value_type>::result_value[1] = val;
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned>
    store(double* v, double* ptr)
    {
        //_mm_storeu_pd(ptr, v);
        std::copy(v, v+2, ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned>
    store(double* v, double* ptr)
    {
        //_mm_store_pd(ptr, v);
        std::copy(v, v+2, ptr);
    }

    static double* add(double* a, double* b)
    {
        //return _mm_add_pd(a, b);
        vector_base<value_type>::result_value[0] = a[0] + b[0];
        vector_base<value_type>::result_value[1] = a[1] + b[1];
        return vector_base<value_type>::result_value;
    }

    static double* sub(double* a, double* b)
    {
        //return _mm_sub_pd(a, b);
        vector_base<value_type>::result_value[0] = a[0] - b[0];
        vector_base<value_type>::result_value[1] = a[1] - b[1];
        return vector_base<value_type>::result_value;
    }

    static double* mul(double* a, double* b)
    {
        //return _mm_mul_pd(a, b);
        vector_base<value_type>::result_value[0] = a[0] * b[0];
        vector_base<value_type>::result_value[1] = a[1] * b[1];
        return vector_base<value_type>::result_value;
    }

    static double* div(double* a, double* b)
    {
        //return _mm_div_pd(a, b);
        vector_base<value_type>::result_value[0] = a[0] / b[0];
        vector_base<value_type>::result_value[1] = a[1] / b[1];
        return vector_base<value_type>::result_value;
    }

    static double* pow(double* a, double* b)
    {
        //return _mm_setr_pd(std::pow((double)a[0], (double)b[0]),
        //                  std::pow((double)a[1], (double)b[1]));
        vector_base<value_type>::result_value[0] = std::pow(a[0] , b[0]);
        vector_base<value_type>::result_value[1] = std::pow(a[1] , b[1]);
        return vector_base<value_type>::result_value;
    }

    static double* negate(double* a)
    {
        //return _mm_xor_pd(a, _mm_set1_pd(-0.0));
        vector_base<value_type>::result_value[0] = -a[0];
        vector_base<value_type>::result_value[1] = -a[1];
        return vector_base<value_type>::result_value;
    }

    static double* exp(double* a)
    {
        //return _mm_setr_pd(std::exp((double)a[0]),
        //                  std::exp((double)a[1]));
        vector_base<value_type>::result_value[0] = std::exp(a[0]);
        vector_base<value_type>::result_value[1] = std::exp(a[1]);
        return vector_base<value_type>::result_value;
    }

    static double* sqrt(double* a)
    {
        //return _mm_sqrt_pd(a);
        vector_base<value_type>::result_value[0] = std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = std::sqrt(a[1]);
        return vector_base<value_type>::result_value;
    }
};

template <>
struct vector_traits<std::complex<float>>
{
    constexpr static unsigned vector_width = 2;
    constexpr static size_t alignment = 64;
    using vector_type = float*;
    using value_type = float;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float* >
    convert(float* v)
    {
        //return _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,0,2,0));
        return v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double*>
    convert(float* v)
    {
        //return _mm_cvtps_pd(convert<float>(v));
        return (double*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float* >
    convert(float* v)
    {
        return v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value, int8_t* >
    convert(float* v)
    {
        //ivector_type i32 = _mm_cvtps_epi32(convert<float>(v));
        //ivector_type i16 = _mm_packs_epi32(i32, i32);
        //return _mm_packs_epi16(i16, i16);
        return (int8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint8_t>::value, uint8_t* >
    convert(float* v)
    {
        //ivector_type i32 = _mm_cvtps_epi32(convert<float>(v));
        //ivector_type i16 = _mm_packus_epi32(i32, i32);
        //return _mm_packus_epi16(i16, i16);
        return (uint8_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value, int16_t* >
    convert(float* v)
    {
        //ivector_type i32 = _mm_cvtps_epi32(convert<float>(v));
        //return _mm_packs_epi32(i32, i32);
        return (int16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint16_t>::value, uint16_t* >
    convert(float* v)
    {
        //ivector_type i32 = _mm_cvtps_epi32(convert<float>(v));
        //return _mm_packus_epi32(i32, i32);
        return (uint16_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value, int32_t* >
    convert(float* v)
    {
        //return _mm_cvtps_epi32(convert<float>(v));
        return (uint32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,uint32_t>::value, uint32_t* >
    convert(float* v)
    {
        //return _mm_setr_epi32((uint32_t)v[0], (uint32_t)v[2],
        //                      (uint32_t)v[0], (uint32_t)v[2]);
        return (uint32_t*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, int64_t* >
    convert(float* v)
    {
        //return _mm_set_epi64x((T)v[2], (T)v[0]);
        return (int64_t*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned, float* >
    load(const std::complex<float>* ptr)
    {
        //return _mm_loadu_ps((float*)ptr);
        vector_base<value_type>::result_value[0] = ptr->real();
        vector_base<value_type>::result_value[1] = ptr->imag();
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned, float* >
    load(const std::complex<float>* ptr)
    {
        //return _mm_load_ps((float*)ptr);
        vector_base<value_type>::result_value[0] = ptr->real();
        vector_base<value_type>::result_value[1] = ptr->imag();
        return vector_base<value_type>::result_value;
    }

    static float* load1(const std::complex<float>* ptr)
    {
        //return _mm_castpd_ps(_mm_load1_pd((double*)ptr));
        vector_base<value_type>::result_value[0] = ptr->real();
        vector_base<value_type>::result_value[1] = ptr->imag();
        return vector_base<value_type>::result_value;
    }

    static float* set1(std::complex<float> val)
    {
        //return _mm_castpd_ps(_mm_set1_pd(*(double*)&val));
        vector_base<value_type>::result_value[0] = val.real();
        vector_base<value_type>::result_value[1] = val.imag();
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned>
    store(float* v, std::complex<float>* ptr)
    {
        //_mm_storeu_ps((float*)ptr, v);
        (*ptr) = {v[0], v[1]};
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned>
    store(float* v, std::complex<float>* ptr)
    {
        //_mm_store_ps((float*)ptr, v);
        (*ptr) = {v[0], v[1]};
    }

    static float* add(float* a, float* b)
    {
        //return _mm_add_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] + b[0];
        vector_base<value_type>::result_value[1] = a[1] + b[1];
        vector_base<value_type>::result_value[2] = a[2] + b[2];
        vector_base<value_type>::result_value[3] = a[3] + b[3];
        return vector_base<value_type>::result_value;
    }

    static float* sub(float* a, float* b)
    {
        //return _mm_sub_ps(a, b);
        vector_base<value_type>::result_value[0] = a[0] - b[0];
        vector_base<value_type>::result_value[1] = a[1] - b[1];
        vector_base<value_type>::result_value[2] = a[2] - b[2];
        vector_base<value_type>::result_value[3] = a[3] - b[3];
        return vector_base<value_type>::result_value;
    }

    static float* mul(float* a, float* b)
    {
/*
        float* ashuf = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,3,0,1));
        float* breal = _mm_moveldup_ps(b);
        float* bimag = _mm_movehdup_ps(b);
        float* tmp1 = _mm_mul_ps(    a, breal); // tmp1 = (ar0*br0, ai0*br0, ar1*br1, ai1*br1)
        float* tmp2 = _mm_mul_ps(ashuf, bimag); // tmp2 = (ai0*bi0, ar0*bi0, ai1*bi1, ar1*bi1)
        return _mm_addsub_ps(tmp1, tmp2);
*/
        //(a + ib) (c + id) = (ac - bd) + i(ad + bc).

        vector_base<value_type>::result_value[0] = a[0]*b[0] + a[1]*b[1];
        vector_base<value_type>::result_value[1] = a[0]*b[1] + a[1]*b[0];

        vector_base<value_type>::result_value[2] = a[2]*b[2] + a[3]*b[3];
        vector_base<value_type>::result_value[3] = a[2]*b[3] + a[3]*b[2];

        return vector_base<value_type>::result_value;
    }

    static float* div(float* a, float* b)
    {
/*
        float* bsqr = _mm_mul_ps(b, b);
        bsqr = _mm_hadd_ps(bsqr, bsqr);
        bsqr = _mm_shuffle_ps(bsqr, bsqr, _MM_SHUFFLE(3,1,2,0)); // bsqr = (|b0|^2, |b0|^2, |b1|^2, |b1|^2)

        float* ashuf = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2,3,0,1));
        float* breal = _mm_moveldup_ps(b);
        float* bimag = _mm_movehdup_ps(b);
        float* tmp1 = _mm_mul_ps(    a, breal); // tmp1 = ( ar0*br0,  ai0*br0,  ar1*br1,  ai1*br1)
        float* tmp2 = _mm_mul_ps(ashuf, bimag);
        tmp2 = _mm_xor_ps(tmp2, _mm_set1_ps(-0.0f)); // tmp2 = (-ai0*bi0, -ar0*bi0, -ai1*bi1, -ar1*bi1)
        float* abconj = _mm_addsub_ps(tmp1, tmp2);

        return _mm_div_ps(abconj, bsqr);
*/
        const float denom_a = b[0] * b[0] + a[1] * a[1];
        vector_base<value_type>::result_value[0] = (a[0] + b[0]) + (a[1] * b[1]) / denom_a;
        vector_base<value_type>::result_value[1] = (a[1] * b[0]) - (a[0] * b[1]) / denom_a;

        const float denom_b = b[2] * b[2] + a[3] * a[3];
        vector_base<value_type>::result_value[2] = (a[2] + b[2]) + (a[3] * b[3]) / denom_b;
        vector_base<value_type>::result_value[3] = (a[3] * b[2]) - (a[2] * b[3]) / denom_b;

        return vector_base<value_type>::result_value;
    }

    static float* pow(float* a, float* b)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0((float)b[0], (float)b[1]);
        std::complex<float> b1((float)b[2], (float)b[3]);
        std::complex<float> c0 = std::pow(a0, b0);
        std::complex<float> c1 = std::pow(a1, b1);
        vector_base<value_type>::result_value[0] = c0.real();
        vector_base<value_type>::result_value[1] = c0.imag();
        vector_base<value_type>::result_value[2] = c1.real();
        vector_base<value_type>::result_value[3] = c1.imag();
        return vector_base<value_type>::result_value;
    }

    static float* negate(float* a)
    {
        vector_base<value_type>::result_value[0] = 0.-a[0];
        vector_base<value_type>::result_value[1] = 0.-a[1];
        vector_base<value_type>::result_value[2] = 0.-a[2];
        vector_base<value_type>::result_value[3] = 0.-a[3];
        return vector_base<value_type>::result_value;
    }

    static float* exp(float* a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0 = std::exp(a0);
        std::complex<float> b1 = std::exp(a1);
        vector_base<value_type>::result_value[0] = b0.real();
        vector_base<value_type>::result_value[1] = b0.imag();
        vector_base<value_type>::result_value[2] = b1.real();
        vector_base<value_type>::result_value[3] = b1.imag();
        return vector_base<value_type>::result_value;
    }

    static float* sqrt(float* a)
    {
        std::complex<float> a0((float)a[0], (float)a[1]);
        std::complex<float> a1((float)a[2], (float)a[3]);
        std::complex<float> b0 = std::sqrt(a0);
        std::complex<float> b1 = std::sqrt(a1);
        vector_base<value_type>::result_value[0] = b0.real();
        vector_base<value_type>::result_value[1] = b0.imag();
        vector_base<value_type>::result_value[2] = b1.real();
        vector_base<value_type>::result_value[3] = b1.imag();
        return vector_base<value_type>::result_value;
    }
};

template <typename U>
struct vector_traits<U, detail::enable_if_t<std::is_same<U,int8_t>::value ||
                                            std::is_same<U,uint8_t>::value>>
{
    constexpr static unsigned vector_width = 32;
    constexpr static size_t alignment = 64;

    using vector_type =
        std::conditional<std::is_same<U,int8_t>::value, 
            int8_t*,
            uint8_t*
        >;

    using value_type = typename vector_base<U>::value_type;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float* >
    convert(vector_type v)
    {
        //return _mm_cvtepi32_ps(convert<int32_t>(v));
        return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double* >
    convert(vector_type v)
    {
        //return _mm_cvtepi32_pd(convert<int32_t>(v));
        return (double*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float* >
    convert(vector_type v)
    {
        //return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
        return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,uint8_t>::value, T* >
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, T* >
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, T * >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ?  (int32_t*)v
                                        :  (uint32_t*)v;
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T * >
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 16 && !Aligned, U* >
    load(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 16 && Aligned, U* >
    load(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8, U* >
    load(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4, U* >
    load(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2, U* >
    load(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    static vector_type load1(const vector_type ptr)
    {
        return const_cast<vector_type>(ptr);
    }

    static vector_type set1(value_type val)
    {
       vector_base<value_type>::result_value[0] = val;
       vector_base<value_type>::result_value[1] = val;
       vector_base<value_type>::result_value[2] = val;
       vector_base<value_type>::result_value[3] = val;
       vector_base<value_type>::result_value[4] = val;
       vector_base<value_type>::result_value[5] = val;
       vector_base<value_type>::result_value[6] = val;
       vector_base<value_type>::result_value[7] = val;
       vector_base<value_type>::result_value[8] = val;
       vector_base<value_type>::result_value[9] = val;
       vector_base<value_type>::result_value[10] = val;
       vector_base<value_type>::result_value[11] = val;
       vector_base<value_type>::result_value[12] = val;
       vector_base<value_type>::result_value[13] = val;
       vector_base<value_type>::result_value[14] = val;
       vector_base<value_type>::result_value[15] = val;
       return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 16 && !Aligned>
    store(vector_type v, vector_type ptr)
    {
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr[3] = v[3];
        ptr[4] = v[4];
        ptr[5] = v[5];
        ptr[6] = v[6];
        ptr[7] = v[7];
        ptr[8] = v[8];
        ptr[9] = v[9];
        ptr[10] = v[10];
        ptr[11] = v[11];
        ptr[12] = v[12];
        ptr[13] = v[13];
        ptr[14] = v[14];
        ptr[15] = v[15];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 16 && Aligned>
    store(vector_type v, vector_type ptr)
    {
        ((int64_t*)ptr)[0] = v[7] << 56 | v[6] << 48 | v[5] << 40 | v[4] << 32 | v[3] << 24 | v[2] << 16 | v[1] << 8 | v[0];
        ((int64_t*)ptr)[1] = v[15] << 56 | v[14] << 48 | v[13] << 40 | v[12] << 32 | v[11] << 24 | v[10] << 16 | v[9] << 8 | v[8];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8>
    store(vector_type v, vector_type ptr)
    {
        (*((value_type*)ptr)) = v[7] << 56 | v[6] << 48 | v[5] << 40 | v[4] << 32 | v[3] << 24 | v[2] << 16 | v[1] << 8 | v[0];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4>
    store(vector_type v, vector_type ptr)
    {
        //*(int32_t*)ptr = _mm_extract_epi32(v, 0);
        // dst[31:0] := (a[127:0] >> (imm8[1:0] * 32))[31:0]
        (*((value_type*)ptr)) = v[3] << 24 | v[2] << 16 | v[1] << 8 | v[0];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2>
    store(vector_type v, vector_type ptr)
    {
        //*(int16_t*)ptr = _mm_extract_epi16(v, 0);
        // (a[127:0] >> (imm8[2:0] * 16))[15:0]
        //
        (*((value_type*)ptr)) = (v[1] << 8) | v[0];
    }

    static vector_type add(vector_type a, vector_type b)
    {
        //return _mm_add_epi8(a, b);
        vector_base<value_type>::result_value[0] = a[0] + b[0];
        vector_base<value_type>::result_value[1] = a[1] + b[1];
        vector_base<value_type>::result_value[2] = a[2] + b[2];
        vector_base<value_type>::result_value[3] = a[3] + b[3];
        vector_base<value_type>::result_value[4] = a[4] + b[4];
        vector_base<value_type>::result_value[5] = a[5] + b[5];
        vector_base<value_type>::result_value[6] = a[6] + b[6];
        vector_base<value_type>::result_value[7] = a[7] + b[7];
        vector_base<value_type>::result_value[8] = a[8] + b[8];
        vector_base<value_type>::result_value[9] = a[9] + b[9];
        vector_base<value_type>::result_value[10] = a[10] + b[10];
        vector_base<value_type>::result_value[11] = a[11] + b[11];
        vector_base<value_type>::result_value[12] = a[12] + b[12];
        vector_base<value_type>::result_value[13] = a[13] + b[13];
        vector_base<value_type>::result_value[14] = a[14] + b[14];
        vector_base<value_type>::result_value[15] = a[15] + b[15];
        return vector_base<value_type>::result_value;
    }

    static vector_type sub(vector_type a, vector_type b)
    {
        //return _mm_sub_epi8(a, b);
        vector_base<value_type>::result_value[0] = a[0] - b[0];
        vector_base<value_type>::result_value[1] = a[1] - b[1];
        vector_base<value_type>::result_value[2] = a[2] - b[2];
        vector_base<value_type>::result_value[3] = a[3] - b[3];
        vector_base<value_type>::result_value[4] = a[4] - b[4];
        vector_base<value_type>::result_value[5] = a[5] - b[5];
        vector_base<value_type>::result_value[6] = a[6] - b[6];
        vector_base<value_type>::result_value[7] = a[7] - b[7];
        vector_base<value_type>::result_value[8] = a[8] - b[8];
        vector_base<value_type>::result_value[9] = a[9] - b[9];
        vector_base<value_type>::result_value[10] = a[10] - b[10];
        vector_base<value_type>::result_value[11] = a[11] - b[11];
        vector_base<value_type>::result_value[12] = a[12] - b[12];
        vector_base<value_type>::result_value[13] = a[13] - b[13];
        vector_base<value_type>::result_value[14] = a[14] - b[14];
        vector_base<value_type>::result_value[15] = a[15] - b[15];
        return vector_base<value_type>::result_value;
    }

    static vector_type mul(vector_type a, vector_type b)
    {
        //ivector_type lo = _mm_and_si128(_mm_mullo_epi16(a, b), _mm_set1_epi16(0xff));
        //ivector_type hi = _mm_mullo_epi16(_mm_srli_epi16(a, 8),_mm_srli_epi16(b, 8));
        //return _mm_or_si128(_mm_slli_epi16(hi, 8), lo);
        vector_base<value_type>::result_value[0] = a[0] * b[0];
        vector_base<value_type>::result_value[1] = a[1] * b[1];
        vector_base<value_type>::result_value[2] = a[2] * b[2];
        vector_base<value_type>::result_value[3] = a[3] * b[3];
        vector_base<value_type>::result_value[4] = a[4] * b[4];
        vector_base<value_type>::result_value[5] = a[5] * b[5];
        vector_base<value_type>::result_value[6] = a[6] * b[6];
        vector_base<value_type>::result_value[7] = a[7] * b[7];
        vector_base<value_type>::result_value[8] = a[8] * b[8];
        vector_base<value_type>::result_value[9] = a[9] * b[9];
        vector_base<value_type>::result_value[10] = a[10] * b[10];
        vector_base<value_type>::result_value[11] = a[11] * b[11];
        vector_base<value_type>::result_value[12] = a[12] * b[12];
        vector_base<value_type>::result_value[13] = a[13] * b[13];
        vector_base<value_type>::result_value[14] = a[14] * b[14];
        vector_base<value_type>::result_value[15] = a[15] * b[15];
        return vector_base<value_type>::result_value;
    }

    static vector_type div(vector_type a, vector_type b)
    {
/*
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
*/
        vector_base<value_type>::result_value[0] = a[0] / b[0];
        vector_base<value_type>::result_value[1] = a[1] / b[1];
        vector_base<value_type>::result_value[2] = a[2] / b[2];
        vector_base<value_type>::result_value[3] = a[3] / b[3];
        vector_base<value_type>::result_value[4] = a[4] / b[4];
        vector_base<value_type>::result_value[5] = a[5] / b[5];
        vector_base<value_type>::result_value[6] = a[6] / b[6];
        vector_base<value_type>::result_value[7] = a[7] / b[7];
        vector_base<value_type>::result_value[8] = a[8] / b[8];
        vector_base<value_type>::result_value[9] = a[9] / b[9];
        vector_base<value_type>::result_value[10] = a[10] / b[10];
        vector_base<value_type>::result_value[11] = a[11] / b[11];
        vector_base<value_type>::result_value[12] = a[12] / b[12];
        vector_base<value_type>::result_value[13] = a[13] / b[13];
        vector_base<value_type>::result_value[14] = a[14] / b[14];
        vector_base<value_type>::result_value[15] = a[15] / b[15];
        return vector_base<value_type>::result_value;
    }

    static vector_type pow(vector_type a, vector_type b)
    {
/*
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
*/
        vector_base<value_type>::result_value[0] = (U)std::pow(a[0], b[0]);
        vector_base<value_type>::result_value[1] = (U)std::pow(a[1], b[1]);
        vector_base<value_type>::result_value[2] = (U)std::pow(a[2], b[2]);
        vector_base<value_type>::result_value[3] = (U)std::pow(a[3], b[3]);
        vector_base<value_type>::result_value[4] = (U)std::pow(a[4], b[4]);
        vector_base<value_type>::result_value[5] = (U)std::pow(a[5], b[5]);
        vector_base<value_type>::result_value[6] = (U)std::pow(a[6], b[6]);
        vector_base<value_type>::result_value[7] = (U)std::pow(a[7], b[7]);
        vector_base<value_type>::result_value[8] = (U)std::pow(a[8], b[8]);
        vector_base<value_type>::result_value[9] = (U)std::pow(a[9], b[9]);
        vector_base<value_type>::result_value[10] = (U)std::pow(a[10], b[10]);
        vector_base<value_type>::result_value[11] = (U)std::pow(a[11], b[11]);
        vector_base<value_type>::result_value[12] = (U)std::pow(a[12], b[12]);
        vector_base<value_type>::result_value[13] = (U)std::pow(a[13], b[13]);
        vector_base<value_type>::result_value[14] = (U)std::pow(a[14], b[14]);
        vector_base<value_type>::result_value[15] = (U)std::pow(a[15], b[15]);

        return vector_base<value_type>::result_value;
    }

    static vector_type negate(vector_type a)
    {
        //return _mm_sub_epi8(_mm_setzero_si128(), a);
        vector_base<value_type>::result_value[0] = -(a[0]);
        vector_base<value_type>::result_value[1] = -(a[1]);
        vector_base<value_type>::result_value[2] = -(a[2]);
        vector_base<value_type>::result_value[3] = -(a[3]);
        vector_base<value_type>::result_value[4] = -(a[4]);
        vector_base<value_type>::result_value[5] = -(a[5]);
        vector_base<value_type>::result_value[6] = -(a[6]);
        vector_base<value_type>::result_value[7] = -(a[7]);
        vector_base<value_type>::result_value[8] = -(a[8]);
        vector_base<value_type>::result_value[9] = -(a[9]);
        vector_base<value_type>::result_value[10] = -(a[10]);
        vector_base<value_type>::result_value[11] = -(a[11]);
        vector_base<value_type>::result_value[12] = -(a[12]);
        vector_base<value_type>::result_value[13] = -(a[13]);
        vector_base<value_type>::result_value[14] = -(a[14]);
        vector_base<value_type>::result_value[15] = -(a[15]);

        return vector_base<value_type>::result_value;
    }

    static vector_type exp(vector_type a)
    {
/*
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
*/
        vector_base<value_type>::result_value[0] = (U)std::exp(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::exp(a[1]);
        vector_base<value_type>::result_value[2] = (U)std::exp(a[2]);
        vector_base<value_type>::result_value[3] = (U)std::exp(a[3]);
        vector_base<value_type>::result_value[4] = (U)std::exp(a[4]);
        vector_base<value_type>::result_value[5] = (U)std::exp(a[5]);
        vector_base<value_type>::result_value[6] = (U)std::exp(a[6]);
        vector_base<value_type>::result_value[7] = (U)std::exp(a[7]);
        vector_base<value_type>::result_value[8] = (U)std::exp(a[8]);
        vector_base<value_type>::result_value[9] = (U)std::exp(a[9]);
        vector_base<value_type>::result_value[10] = (U)std::exp(a[10]);
        vector_base<value_type>::result_value[11] = (U)std::exp(a[11]);
        vector_base<value_type>::result_value[12] = (U)std::exp(a[12]);
        vector_base<value_type>::result_value[13] = (U)std::exp(a[13]);
        vector_base<value_type>::result_value[14] = (U)std::exp(a[14]);
        vector_base<value_type>::result_value[15] = (U)std::exp(a[15]);

        return vector_base<value_type>::result_value;
    }

    static U* sqrt(U* a)
    {
/*
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
*/
        vector_base<value_type>::result_value[0] = (U)std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::sqrt(a[1]);
        vector_base<value_type>::result_value[2] = (U)std::sqrt(a[2]);
        vector_base<value_type>::result_value[3] = (U)std::sqrt(a[3]);
        vector_base<value_type>::result_value[4] = (U)std::sqrt(a[4]);
        vector_base<value_type>::result_value[5] = (U)std::sqrt(a[5]);
        vector_base<value_type>::result_value[6] = (U)std::sqrt(a[6]);
        vector_base<value_type>::result_value[7] = (U)std::sqrt(a[7]);
        vector_base<value_type>::result_value[8] = (U)std::sqrt(a[8]);
        vector_base<value_type>::result_value[9] = (U)std::sqrt(a[9]);
        vector_base<value_type>::result_value[10] = (U)std::sqrt(a[10]);
        vector_base<value_type>::result_value[11] = (U)std::sqrt(a[11]);
        vector_base<value_type>::result_value[12] = (U)std::sqrt(a[12]);
        vector_base<value_type>::result_value[13] = (U)std::sqrt(a[13]);
        vector_base<value_type>::result_value[14] = (U)std::sqrt(a[14]);
        vector_base<value_type>::result_value[15] = (U)std::sqrt(a[15]);

        return vector_base<value_type>::result_value;
    }
};


template <typename U>
struct vector_traits<U, detail::enable_if_t<std::is_same<U,int16_t>::value ||
                                            std::is_same<U,uint16_t>::value>>
{
    constexpr static unsigned vector_width = 16;
    constexpr static size_t alignment = 64;

    using vector_type =
        std::conditional<std::is_same<U,uint16_t>::value,
            int16_t*,
            uint16_t*
        >;

    using value_type = typename vector_base<U>::value_type;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float*>
    convert(vector_type v)
    {
//        return _mm_cvtepi32_ps(convert<int32_t>(v));
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double*>
    convert(vector_type v)
    {
//        return _mm_cvtepi32_pd(convert<int32_t>(v));
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float*>
    convert(vector_type v)
    {
//        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,U>::value, T* >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ? _mm_packs_epi16(v, v)
                                        : _mm_packus_epi16(v, v);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, T* >
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, T* >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ? _mm_cvtepi16_epi32(v)
                                        : _mm_cvtepu16_epi32(v);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T* >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ? _mm_cvtepi16_epi64(v)
                                        : _mm_cvtepu16_epi64(v);
*/
        return (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8 && !Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_loadu_si128((ivector_type*)ptr);
       vector_base<value_type>::result_value[0] = ptr[0];
       vector_base<value_type>::result_value[1] = ptr[1];
       vector_base<value_type>::result_value[2] = ptr[2];
       vector_base<value_type>::result_value[3] = ptr[3];
       vector_base<value_type>::result_value[4] = ptr[4];
       vector_base<value_type>::result_value[5] = ptr[5];
       vector_base<value_type>::result_value[6] = ptr[6];
       vector_base<value_type>::result_value[7] = ptr[7];
       return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8 && Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_load_si128((ivector_type*)ptr);
       vector_base<value_type>::result_value[0] = ptr[0];
       vector_base<value_type>::result_value[1] = ptr[1];
       vector_base<value_type>::result_value[2] = ptr[2];
       vector_base<value_type>::result_value[3] = ptr[3];
       vector_base<value_type>::result_value[4] = ptr[4];
       vector_base<value_type>::result_value[5] = ptr[5];
       vector_base<value_type>::result_value[6] = ptr[6];
       vector_base<value_type>::result_value[7] = ptr[7];
       return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_set1_epi64x(*(int64_t*)ptr);
       vector_base<value_type>::result_value[0] = ptr[0];
       vector_base<value_type>::result_value[1] = ptr[1];
       vector_base<value_type>::result_value[2] = ptr[2];
       vector_base<value_type>::result_value[3] = ptr[3];
       vector_base<value_type>::result_value[4] = ptr[4];
       vector_base<value_type>::result_value[5] = ptr[5];
       vector_base<value_type>::result_value[6] = ptr[6];
       vector_base<value_type>::result_value[7] = ptr[7];
       return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_set1_epi32(*(int32_t*)ptr);
       vector_base<value_type>::result_value[0] = ptr[0];
       vector_base<value_type>::result_value[1] = ptr[1];
       vector_base<value_type>::result_value[2] = ptr[2];
       vector_base<value_type>::result_value[3] = ptr[3];
       vector_base<value_type>::result_value[4] = ptr[4];
       vector_base<value_type>::result_value[5] = ptr[5];
       vector_base<value_type>::result_value[6] = ptr[6];
       vector_base<value_type>::result_value[7] = ptr[7];
       return vector_base<value_type>::result_value;
    }

    static vector_type load1(const vector_type ptr)
    {
//        return _mm_set1_epi16(*ptr);
       vector_base<value_type>::result_value[0] = ptr[0];
       vector_base<value_type>::result_value[1] = ptr[1];
       vector_base<value_type>::result_value[2] = ptr[2];
       vector_base<value_type>::result_value[3] = ptr[3];
       vector_base<value_type>::result_value[4] = ptr[4];
       vector_base<value_type>::result_value[5] = ptr[5];
       vector_base<value_type>::result_value[6] = ptr[6];
       vector_base<value_type>::result_value[7] = ptr[7];
       return vector_base<value_type>::result_value;
    }

    static vector_type set1(value_type val)
    {
//        return _mm_set1_epi16(val);
       vector_base<value_type>::result_value[0] = val;
       vector_base<value_type>::result_value[1] = val;
       vector_base<value_type>::result_value[2] = val;
       vector_base<value_type>::result_value[3] = val;
       vector_base<value_type>::result_value[4] = val;
       vector_base<value_type>::result_value[5] = val;
       vector_base<value_type>::result_value[6] = val;
       vector_base<value_type>::result_value[7] = val;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8 && !Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_storeu_si128((ivector_type*)ptr, v);
       ptr[0] = v[0];
       ptr[1] = v[1];
       ptr[2] = v[2];
       ptr[3] = v[3];
       ptr[4] = v[4];
       ptr[5] = v[5];
       ptr[6] = v[6];
       ptr[7] = v[7];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 8 && Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_store_si128((ivector_type*)ptr, v);
       ptr[0] = v[0];
       ptr[1] = v[1];
       ptr[2] = v[2];
       ptr[3] = v[3];
       ptr[4] = v[4];
       ptr[5] = v[5];
       ptr[6] = v[6];
       ptr[7] = v[7];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4>
    store(vector_type v, vector_type ptr)
    {
//        _mm_storel_epi64((ivector_type*)ptr, v);
       ptr[0] = v[0];
       ptr[1] = v[1];
       ptr[2] = v[2];
       ptr[3] = v[3];
       ptr[4] = v[4];
       ptr[5] = v[5];
       ptr[6] = v[6];
       ptr[7] = v[7];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2>
    store(vector_type v, vector_type ptr)
    {
//        *(int32_t*)ptr = _mm_extract_epi32(v, 0);
       ptr[0] = v[0];
       ptr[1] = v[1];
       ptr[2] = v[2];
       ptr[3] = v[3];
       ptr[4] = v[4];
       ptr[5] = v[5];
       ptr[6] = v[6];
       ptr[7] = v[7];
    }

    static vector_type add(vector_type a, vector_type b)
    {
//        return _mm_add_epi16(a, b);
        vector_base<value_type>::result_value[0] = (U)(a[0]) + (U)(b[0]);
        vector_base<value_type>::result_value[1] = (U)(a[1]) + (U)(b[1]);
        vector_base<value_type>::result_value[2] = (U)(a[2]) + (U)(b[2]);
        vector_base<value_type>::result_value[3] = (U)(a[3]) + (U)(b[3]);
        vector_base<value_type>::result_value[4] = (U)(a[4]) + (U)(b[4]);
        vector_base<value_type>::result_value[5] = (U)(a[5]) + (U)(b[5]);
        vector_base<value_type>::result_value[6] = (U)(a[6]) + (U)(b[6]);
        vector_base<value_type>::result_value[7] = (U)(a[7]) + (U)(b[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type sub(vector_type a, vector_type b)
    {
//        return _mm_sub_epi16(a, b);
        vector_base<value_type>::result_value[0] = (U)(a[0]) - (U)(b[0]);
        vector_base<value_type>::result_value[1] = (U)(a[1]) - (U)(b[1]);
        vector_base<value_type>::result_value[2] = (U)(a[2]) - (U)(b[2]);
        vector_base<value_type>::result_value[3] = (U)(a[3]) - (U)(b[3]);
        vector_base<value_type>::result_value[4] = (U)(a[4]) - (U)(b[4]);
        vector_base<value_type>::result_value[5] = (U)(a[5]) - (U)(b[5]);
        vector_base<value_type>::result_value[6] = (U)(a[6]) - (U)(b[6]);
        vector_base<value_type>::result_value[7] = (U)(a[7]) - (U)(b[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type mul(vector_type a, vector_type b)
    {
//        return _mm_mullo_epi16(a, b);
        vector_base<value_type>::result_value[0] = (U)(a[0]) * (U)(b[0]);
        vector_base<value_type>::result_value[1] = (U)(a[1]) * (U)(b[1]);
        vector_base<value_type>::result_value[2] = (U)(a[2]) * (U)(b[2]);
        vector_base<value_type>::result_value[3] = (U)(a[3]) * (U)(b[3]);
        vector_base<value_type>::result_value[4] = (U)(a[4]) * (U)(b[4]);
        vector_base<value_type>::result_value[5] = (U)(a[5]) * (U)(b[5]);
        vector_base<value_type>::result_value[6] = (U)(a[6]) * (U)(b[6]);
        vector_base<value_type>::result_value[7] = (U)(a[7]) * (U)(b[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type div(vector_type a, vector_type b)
    {
/*
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
*/
        vector_base<value_type>::result_value[0] = (U)(a[0]) / (U)(b[0]);
        vector_base<value_type>::result_value[1] = (U)(a[1]) / (U)(b[1]);
        vector_base<value_type>::result_value[2] = (U)(a[2]) / (U)(b[2]);
        vector_base<value_type>::result_value[3] = (U)(a[3]) / (U)(b[3]);
        vector_base<value_type>::result_value[4] = (U)(a[4]) / (U)(b[4]);
        vector_base<value_type>::result_value[5] = (U)(a[5]) / (U)(b[5]);
        vector_base<value_type>::result_value[6] = (U)(a[6]) / (U)(b[6]);
        vector_base<value_type>::result_value[7] = (U)(a[7]) / (U)(b[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type pow(vector_type a, vector_type b)
    {
/*
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

*/
        vector_base<value_type>::result_value[0] = (U)std::pow(a[0], b[0]);
        vector_base<value_type>::result_value[1] = (U)std::pow(a[1], b[1]);
        vector_base<value_type>::result_value[2] = (U)std::pow(a[2], b[2]);
        vector_base<value_type>::result_value[3] = (U)std::pow(a[3], b[3]);
        vector_base<value_type>::result_value[4] = (U)std::pow(a[4], b[4]);
        vector_base<value_type>::result_value[5] = (U)std::pow(a[5], b[5]);
        vector_base<value_type>::result_value[6] = (U)std::pow(a[6], b[6]);
        vector_base<value_type>::result_value[7] = (U)std::pow(a[7], b[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type negate(vector_type a)
    {
//        return _mm_sub_epi16(_mm_setzero_si128(), a);
        vector_base<value_type>::result_value[0] = (U)(0) - (U)(a[0]);
        vector_base<value_type>::result_value[1] = (U)(0) - (U)(a[1]);
        vector_base<value_type>::result_value[2] = (U)(0) - (U)(a[2]);
        vector_base<value_type>::result_value[3] = (U)(0) - (U)(a[3]);
        vector_base<value_type>::result_value[4] = (U)(0) - (U)(a[4]);
        vector_base<value_type>::result_value[5] = (U)(0) - (U)(a[5]);
        vector_base<value_type>::result_value[6] = (U)(0) - (U)(a[6]);
        vector_base<value_type>::result_value[7] = (U)(0) - (U)(a[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type exp(vector_type a)
    {
/*
        return _mm_setr_epi16((U)std::exp((U)_mm_extract_epi16(a, 0)),
                             (U)std::exp((U)_mm_extract_epi16(a, 1)),
                             (U)std::exp((U)_mm_extract_epi16(a, 2)),
                             (U)std::exp((U)_mm_extract_epi16(a, 3)),
                             (U)std::exp((U)_mm_extract_epi16(a, 4)),
                             (U)std::exp((U)_mm_extract_epi16(a, 5)),
                             (U)std::exp((U)_mm_extract_epi16(a, 6)),
                             (U)std::exp((U)_mm_extract_epi16(a, 7)));
*/
        vector_base<value_type>::result_value[0] = (U)std::exp(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::exp(a[1]);
        vector_base<value_type>::result_value[2] = (U)std::exp(a[2]);
        vector_base<value_type>::result_value[3] = (U)std::exp(a[3]);
        vector_base<value_type>::result_value[4] = (U)std::exp(a[4]);
        vector_base<value_type>::result_value[5] = (U)std::exp(a[5]);
        vector_base<value_type>::result_value[6] = (U)std::exp(a[6]);
        vector_base<value_type>::result_value[7] = (U)std::exp(a[7]);
        return vector_base<value_type>::result_value;
    }

    static vector_type sqrt(vector_type a)
    {
/*
        return _mm_setr_epi16((U)std::sqrt((U)_mm_extract_epi16(a, 0)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 1)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 2)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 3)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 4)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 5)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 6)),
                             (U)std::sqrt((U)_mm_extract_epi16(a, 7)));
*/
        vector_base<value_type>::result_value[0] = (U)std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::sqrt(a[1]);
        vector_base<value_type>::result_value[2] = (U)std::sqrt(a[2]);
        vector_base<value_type>::result_value[3] = (U)std::sqrt(a[3]);
        vector_base<value_type>::result_value[4] = (U)std::sqrt(a[4]);
        vector_base<value_type>::result_value[5] = (U)std::sqrt(a[5]);
        vector_base<value_type>::result_value[6] = (U)std::sqrt(a[6]);
        vector_base<value_type>::result_value[7] = (U)std::sqrt(a[7]);
        return vector_base<value_type>::result_value;
    }
};


template <typename U>
struct vector_traits<U, detail::enable_if_t<std::is_same<U,int32_t>::value ||
                                            std::is_same<U,uint32_t>::value>>
{
    constexpr static unsigned vector_width = 8;
    constexpr static size_t alignment = 64;

    using vector_type =
        std::conditional<std::is_same<U,uint16_t>::value,
            int32_t*,
            uint32_t*
        >;

    using value_type = typename vector_base<U>::value_type;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float*>
    convert(vector_type v)
    {
//        return _mm_cvtepi32_ps(v);
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double*>
    convert(vector_type v)
    {
        //return _mm_cvtepi32_pd(v);
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float*>
    convert(vector_type v)
    {
        //return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,U>::value, U* >
    convert(vector_type v)
    {
/*
        U* i16 = std::is_signed<U>::value ? _mm_packs_epi32(v, v)
                                               : _mm_packus_epi32(v, v);
        return std::is_signed<U>::value ? _mm_packs_epi16(i16, i16)
                                        : _mm_packus_epi16(i16, i16);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, T >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ? _mm_packs_epi32(v, v)
                                        : _mm_packus_epi32(v, v);
*/
        return std::is_signed<U>::value ? (T*)v
                                        : (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, T >
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T >
    convert(vector_type v)
    {
/*
        return std::is_signed<U>::value ? _mm_cvtepi32_epi64(v)
                                        : _mm_cvtepu32_epi64(v);
*/
        return std::is_signed<U>::value ? (T*)v
                                        : (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && !Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_loadu_si128((ivector_type*)ptr);
       return ptr;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_load_si128((ivector_type*)ptr);
       return ptr;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_set1_epi64x(*(int64_t*)ptr);
       return ptr;
    }

    static vector_type load1(const vector_type ptr)
    {
//        return _mm_set1_epi32(*ptr);
       return ptr;
    }

    static vector_type set1(value_type val)
    {
//        return _mm_set1_epi32(val);
        vector_base<value_type>::result_value[0] = val;
        vector_base<value_type>::result_value[1] = val;
        vector_base<value_type>::result_value[2] = val;
        vector_base<value_type>::result_value[3] = val;
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && !Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_storeu_si128((ivector_type*)ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr[3] = v[3];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 4 && Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_store_si128((ivector_type*)ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr[3] = v[3];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2>
    store(vector_type v, vector_type ptr)
    {
//        _mm_storel_epi64((ivector_type*)ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
        ptr[2] = v[2];
        ptr[3] = v[3];
    }

    static vector_type add(vector_type a, vector_type b)
    {
//        return _mm_add_epi32(a, b);
        vector_base<value_type>::result_value[0] = a[0] + b[0];
        vector_base<value_type>::result_value[1] = a[1] + b[1];
        vector_base<value_type>::result_value[2] = a[2] + b[2];
        vector_base<value_type>::result_value[3] = a[3] + b[3];
        return vector_base<value_type>::result_value;
    }

    static vector_type sub(vector_type a, vector_type b)
    {
//        return _mm_sub_epi32(a, b);
        vector_base<value_type>::result_value[0] = a[0] - b[0];
        vector_base<value_type>::result_value[1] = a[1] - b[1];
        vector_base<value_type>::result_value[2] = a[2] - b[2];
        vector_base<value_type>::result_value[3] = a[3] - b[3];
        return vector_base<value_type>::result_value;
    }

    static vector_type mul(vector_type a, vector_type b)
    {
//        return _mm_mullo_epi32(a, b);
        vector_base<value_type>::result_value[0] = a[0] * b[0];
        vector_base<value_type>::result_value[1] = a[1] * b[1];
        vector_base<value_type>::result_value[2] = a[2] * b[2];
        vector_base<value_type>::result_value[3] = a[3] * b[3];
        return vector_base<value_type>::result_value;
    }

    static vector_type div(vector_type a, vector_type b)
    {
/*
        return _mm_setr_epi32((U)_mm_extract_epi32(a, 0) /
                              (U)_mm_extract_epi32(b, 0),
                              (U)_mm_extract_epi32(a, 1) /
                              (U)_mm_extract_epi32(b, 1),
                              (U)_mm_extract_epi32(a, 2) /
                              (U)_mm_extract_epi32(b, 2),
                              (U)_mm_extract_epi32(a, 3) /
                              (U)_mm_extract_epi32(b, 3));
*/
        vector_base<value_type>::result_value[0] = a[0] / b[0];
        vector_base<value_type>::result_value[1] = a[1] / b[1];
        vector_base<value_type>::result_value[2] = a[2] / b[2];
        vector_base<value_type>::result_value[3] = a[3] / b[3];
        return vector_base<value_type>::result_value;
    }

    static vector_type pow(vector_type a, vector_type b)
    {
/*
        return _mm_setr_epi32((U)std::pow((U)_mm_extract_epi32(a, 0),
                                          (U)_mm_extract_epi32(b, 0)),
                              (U)std::pow((U)_mm_extract_epi32(a, 1),
                                          (U)_mm_extract_epi32(b, 1)),
                              (U)std::pow((U)_mm_extract_epi32(a, 2),
                                          (U)_mm_extract_epi32(b, 2)),
                              (U)std::pow((U)_mm_extract_epi32(a, 3),
                                          (U)_mm_extract_epi32(b, 3)));
*/
        vector_base<value_type>::result_value[0] = std::pow(a[0], b[0]);
        vector_base<value_type>::result_value[1] = std::pow(a[1], b[1]);
        vector_base<value_type>::result_value[2] = std::pow(a[2], b[2]);
        vector_base<value_type>::result_value[3] = std::pow(a[3], b[3]);
        return vector_base<value_type>::result_value;
    }

    static vector_type negate(vector_type a)
    {
//        return _mm_sub_epi32(_mm_setzero_si128(), a);
        vector_base<value_type>::result_value[0] = 0 - a[0];
        vector_base<value_type>::result_value[1] = 0 - a[1];
        vector_base<value_type>::result_value[2] = 0 - a[2];
        vector_base<value_type>::result_value[3] = 0 - a[3];
        return vector_base<value_type>::result_value;
    }

    static vector_type exp(vector_type a)
    {
/*
        return _mm_setr_epi32((U)std::exp((U)_mm_extract_epi32(a, 0)),
                              (U)std::exp((U)_mm_extract_epi32(a, 1)),
                              (U)std::exp((U)_mm_extract_epi32(a, 2)),
                              (U)std::exp((U)_mm_extract_epi32(a, 3)));
*/
        vector_base<value_type>::result_value[0] = std::exp(a[0]);
        vector_base<value_type>::result_value[1] = std::exp(a[1]);
        vector_base<value_type>::result_value[2] = std::exp(a[2]);
        vector_base<value_type>::result_value[3] = std::exp(a[3]);
        return vector_base<value_type>::result_value;
    }

    static vector_type sqrt(vector_type a)
    {
/*
        return _mm_setr_epi32((U)std::sqrt((U)_mm_extract_epi32(a, 0)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 1)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 2)),
                              (U)std::sqrt((U)_mm_extract_epi32(a, 3)));
*/
        vector_base<value_type>::result_value[0] = std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = std::sqrt(a[1]);
        vector_base<value_type>::result_value[2] = std::sqrt(a[2]);
        vector_base<value_type>::result_value[3] = std::sqrt(a[3]);
        return vector_base<value_type>::result_value;
    }
};


template <typename U>
struct vector_traits<U, detail::enable_if_t<std::is_same<U,int64_t>::value ||
                                            std::is_same<U,uint64_t>::value>>
{
    constexpr static unsigned vector_width = 4;
    constexpr static size_t alignment = 64;

    using vector_type =
        std::conditional<std::is_same<U,uint16_t>::value,
            int64_t*,
            uint64_t*
        >;

    using value_type = typename vector_base<U>::value_type;

    template <typename T> static
    detail::enable_if_t<std::is_same<T,float>::value, float* >
    convert(vector_type v)
    {
/*
        float a = (U)_mm_extract_epi64(v, 0);
        float b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_ps(a, b, a, b);
*/
        return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,double>::value, double* >
    convert(vector_type v)
    {
/*
        double a = (U)_mm_extract_epi64(v, 0);
        double b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_pd(a, b);
*/
        return (double*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,std::complex<float>>::value, float* >
    convert(vector_type v)
    {
//        return _mm_unpacklo_ps(convert<float>(v), _mm_setzero_ps());
       return (float*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int8_t>::value ||
                        std::is_same<T,U>::value, T* >
    convert(vector_type v)
    {
/*
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi8(a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int16_t>::value ||
                        std::is_same<T,uint16_t>::value, T* >
    convert(vector_type v)
    {
/*
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi16(a, b, a, b, a, b, a, b);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int32_t>::value ||
                        std::is_same<T,uint32_t>::value, T* >
    convert(vector_type v)
    {
/*
        T a = (U)_mm_extract_epi64(v, 0);
        T b = (U)_mm_extract_epi64(v, 1);
        return _mm_setr_epi32(a, b, a, b);
*/
        return (T*)v;
    }

    template <typename T> static
    detail::enable_if_t<std::is_same<T,int64_t>::value ||
                        std::is_same<T,uint64_t>::value, T*>
    convert(vector_type v)
    {
        return (T*)v;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_loadu_si128((ivector_type*)ptr);
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1]; 
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned, vector_type >
    load(const vector_type ptr)
    {
//        return _mm_load_si128((ivector_type*)ptr);
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1]; 
        return vector_base<value_type>::result_value;
    }

    static vector_type load1(const vector_type ptr)
    {
//        return _mm_set1_epi64x(*ptr);
        vector_base<value_type>::result_value[0] = ptr[0];
        vector_base<value_type>::result_value[1] = ptr[1]; 
        return vector_base<value_type>::result_value;
    }

    static vector_type set1(value_type val)
    {
//        return _mm_set1_epi64x(val);
        vector_base<value_type>::result_value[0] = val;
        vector_base<value_type>::result_value[1] = val; 
        return vector_base<value_type>::result_value;
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && !Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_storeu_si128((ivector_type*)ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
    }

    template <unsigned Width, bool Aligned> static
    detail::enable_if_t<Width == 2 && Aligned>
    store(vector_type v, vector_type ptr)
    {
//        _mm_store_si128((ivector_type*)ptr, v);
        ptr[0] = v[0];
        ptr[1] = v[1];
    }

    static vector_type add( vector_type a, vector_type b )
    {
//        return _mm_add_epi64(a, b);
        vector_base<value_type>::result_value[0] = ((U)a[0] + (U)b[0]);
        vector_base<value_type>::result_value[1] = ((U)a[1] + (U)b[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type sub( vector_type a, vector_type b )
    {
//        return _mm_sub_epi64(a, b);

        vector_base<value_type>::result_value[0] = ((U)a[0] - (U)b[0]);
        vector_base<value_type>::result_value[1] = ((U)a[1] - (U)b[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type mul( vector_type a, vector_type b )
    {
/*
        return _mm_set_epi64x((U)_mm_extract_epi64(a, 1) *
                              (U)_mm_extract_epi64(b, 1),
                              (U)_mm_extract_epi64(a, 0) *
                              (U)_mm_extract_epi64(b, 0));
*/
        vector_base<value_type>::result_value[0] = ((U)a[0] * (U)b[0]);
        vector_base<value_type>::result_value[1] = ((U)a[1] * (U)b[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type div( vector_type a, vector_type b )
    {
/*
        return _mm_set_epi64x((U)_mm_extract_epi64(a, 1) /
                              (U)_mm_extract_epi64(b, 1),
                              (U)_mm_extract_epi64(a, 0) /
                              (U)_mm_extract_epi64(b, 0));
*/
        vector_base<value_type>::result_value[0] = ((U)a[0] / (U)b[0]);
        vector_base<value_type>::result_value[1] = ((U)a[1] / (U)b[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type pow( vector_type a, vector_type b)
    {
/*
        return _mm_set_epi64x((U)std::pow((U)_mm_extract_epi64(a, 1),
                                          (U)_mm_extract_epi64(b, 1)),
                              (U)std::pow((U)_mm_extract_epi64(a, 0),
                                          (U)_mm_extract_epi64(b, 0)));
*/
        vector_base<value_type>::result_value[0] = (U)std::pow((U)a[0], (U)b[0]);
        vector_base<value_type>::result_value[1] = (U)std::pow((U)a[1], (U)b[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type negate( vector_type a )
    {
        //return _mm_sub_epi64(_mm_setzero_si128(), a);
        return ((U)0)-a;
    }

    static vector_type exp(vector_type a)
    {
        //return _mm_set_epi64x((U)std::exp((U)_mm_extract_epi64(a, 1)),
        //                      (U)std::exp((U)_mm_extract_epi64(a, 0)));
        vector_base<value_type>::result_value[0] = (U)std::exp(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::exp(a[1]);
        return vector_base<value_type>::result_value;
    }

    static vector_type sqrt(vector_type a)
    {
        //return _mm_set_epi64x((U)std::sqrt((U)_mm_extract_epi64(a, 1)),
        //                      (U)std::sqrt((U)_mm_extract_epi64(a, 0)));

        vector_base<value_type>::result_value[0] = (U)std::sqrt(a[0]);
        vector_base<value_type>::result_value[1] = (U)std::sqrt(a[1]);
        return vector_base<value_type>::result_value;
    }
};

}

#endif
