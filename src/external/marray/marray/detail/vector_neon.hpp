#ifndef MARRAY_VECTOR_NEON_HPP
#define MARRAY_VECTOR_NEON_HPP

#if defined(__ARM_NEON)

#include <arm_neon.h>

#include "vector.hpp"

namespace MArray
{

template <>
struct vector_traits<float>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 16;
    typedef float32x4_t vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(float32x4_t v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(float32x4_t v)
    {
        return vcvt_f64_f32(vget_low_f32(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(float32x4_t v)
    {
        return vzip1q_f32(v, vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t>, int8x16_t>
    convert(float32x4_t v)
    {
        int8x8_t i8 = vmovn_s16(convert<int16_t>(v));
        return vcombine_s8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint8_t>, uint8x16_t>
    convert(float32x4_t v)
    {
        uint8x8_t i8 = vmovn_u16(convert<uint16_t>(v));
        return vcombine_u8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t>, int16x8_t>
    convert(float32x4_t v)
    {
        int16x4_t i16 = vmovn_s32(convert<int32_t>(v));
        return vcombine_s16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint16_t>, uint16x8_t>
    convert(float32x4_t v)
    {
        uint16x4_t i16 = vmovn_u32(convert<uint32_t>(v));
        return vcombine_u16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t>, int32x4_t>
    convert(float32x4_t v)
    {
        return vcvtq_s32_f32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint32_t>, uint32x4_t>
    convert(float32x4_t v)
    {
        return vcvtq_u32_f32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t>, int64x2_t>
    convert(float32x4_t v)
    {
        return vcvtq_s64_f64(convert<double>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint64_t>, uint64x2_t>
    convert(float32x4_t v)
    {
        return vcvtq_u64_f64(convert<double>(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, float32x4_t>
    load(const float* ptr)
    {
        return vld1q_f32(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, float32x4_t>
    load(const float* ptr)
    {
        float32x2_t f32 = vld1_f32(ptr);
        return vcombine_f32(f32, f32);
    }

    static float32x4_t load1(const float* ptr)
    {
        return vld1q_dup_f32(ptr);
    }

    static float32x4_t set1(float val)
    {
        return vdupq_n_f32(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(float32x4_t v, float* ptr)
    {
        vst1q_f32(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(float32x4_t v, float* ptr)
    {
        vst1_f32(ptr, vget_low_f32(v));
    }

    static float32x4_t add(float32x4_t a, float32x4_t b)
    {
        return vaddq_f32(a, b);
    }

    static float32x4_t sub(float32x4_t a, float32x4_t b)
    {
        return vsubq_f32(a, b);
    }

    static float32x4_t mul(float32x4_t a, float32x4_t b)
    {
        return vmulq_f32(a, b);
    }

    static float32x4_t div(float32x4_t a, float32x4_t b)
    {
        return vdivq_f32(a, b);
    }

    static float32x4_t pow(float32x4_t a, float32x4_t b)
    {
        float32x4_t c;
        c = vsetq_lane_f32(std::pow(vgetq_lane_f32(a, 0), vgetq_lane_f32(b, 0)), c, 0);
        c = vsetq_lane_f32(std::pow(vgetq_lane_f32(a, 1), vgetq_lane_f32(b, 1)), c, 1);
        c = vsetq_lane_f32(std::pow(vgetq_lane_f32(a, 2), vgetq_lane_f32(b, 2)), c, 2);
        c = vsetq_lane_f32(std::pow(vgetq_lane_f32(a, 3), vgetq_lane_f32(b, 3)), c, 3);
        return c;
    }

    static float32x4_t negate(float32x4_t a)
    {
        return vnegq_f32(a);
    }

    static float32x4_t exp(float32x4_t a)
    {
        float32x4_t c;
        c = vsetq_lane_f32(std::exp(vgetq_lane_f32(a, 0)), c, 0);
        c = vsetq_lane_f32(std::exp(vgetq_lane_f32(a, 1)), c, 1);
        c = vsetq_lane_f32(std::exp(vgetq_lane_f32(a, 2)), c, 2);
        c = vsetq_lane_f32(std::exp(vgetq_lane_f32(a, 3)), c, 3);
        return c;
    }

    static float32x4_t sqrt(float32x4_t a)
    {
        return vsqrtq_f32(a);
    }
};

template <>
struct vector_traits<double>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef float64x2_t vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(float64x2_t v)
    {
        float32x2_t f32 = vcvt_f32_f64(v);
        return vcombine_f32(f32, f32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(float64x2_t v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(float64x2_t v)
    {
        float32x2_t f32 = vcvt_f32_f64(v);
        return vzip1q_f32(vcombine_f32(f32, f32), vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t>, int8x16_t>
    convert(float64x2_t v)
    {
        int8x8_t i8 = vmovn_s16(convert<int16_t>(v));
        return vcombine_s8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint8_t>, uint8x16_t>
    convert(float64x2_t v)
    {
        uint8x8_t i8 = vmovn_u16(convert<uint16_t>(v));
        return vcombine_u8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t>, int16x8_t>
    convert(float64x2_t v)
    {
        int16x4_t i16 = vmovn_s32(convert<int32_t>(v));
        return vcombine_s16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint16_t>, uint16x8_t>
    convert(float64x2_t v)
    {
        uint16x4_t i16 = vmovn_u32(convert<uint32_t>(v));
        return vcombine_u16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t>, int32x4_t>
    convert(float64x2_t v)
    {
        int32x2_t i32 = vmovn_s64(convert<int64_t>(v));
        return vcombine_s32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint32_t>, uint32x4_t>
    convert(float64x2_t v)
    {
        uint32x2_t i32 = vmovn_u64(convert<uint64_t>(v));
        return vcombine_u32(i32, i32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t>, int64x2_t>
    convert(float64x2_t v)
    {
        return vcvtq_s64_f64(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint64_t>, uint64x2_t>
    convert(float64x2_t v)
    {
        return vcvtq_u64_f64(v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, float64x2_t>
    load(const double* ptr)
    {
        return vld1q_f64(ptr);
    }

    static float64x2_t load1(const double* ptr)
    {
        return vld1q_dup_f64(ptr);
    }

    static float64x2_t set1(double val)
    {
        return vdupq_n_f64(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(float64x2_t v, double* ptr)
    {
        vst1q_f64(ptr, v);
    }

    static float64x2_t add(float64x2_t a, float64x2_t b)
    {
        auto c = vaddq_f64(a, b);
        return c;
    }

    static float64x2_t sub(float64x2_t a, float64x2_t b)
    {
        return vsubq_f64(a, b);
    }

    static float64x2_t mul(float64x2_t a, float64x2_t b)
    {
        return vmulq_f64(a, b);
    }

    static float64x2_t div(float64x2_t a, float64x2_t b)
    {
        return vdivq_f64(a, b);
    }

    static float64x2_t pow(float64x2_t a, float64x2_t b)
    {
        float64x2_t c;
        c = vsetq_lane_f64(std::pow(vgetq_lane_f64(a, 0), vgetq_lane_f64(b, 0)), c, 0);
        c = vsetq_lane_f64(std::pow(vgetq_lane_f64(a, 1), vgetq_lane_f64(b, 1)), c, 1);
        return c;
    }

    static float64x2_t negate(float64x2_t a)
    {
        return vnegq_f64(a);
    }

    static float64x2_t exp(float64x2_t a)
    {
        float64x2_t c;
        c = vsetq_lane_f64(std::exp(vgetq_lane_f64(a, 0)), c, 0);
        c = vsetq_lane_f64(std::exp(vgetq_lane_f64(a, 1)), c, 1);
        return c;
    }

    static float64x2_t sqrt(float64x2_t a)
    {
        return vsqrtq_f64(a);
    }
};

template <>
struct vector_traits<std::complex<float>>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef float32x4_t vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(float32x4_t v)
    {
        float32x2_t f32 = vuzp1_f32(vget_low_f32(v), vget_high_f32(v));
        return vcombine_f32(f32, f32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(float32x4_t v)
    {
        return vcvt_f64_f32(vuzp1_f32(vget_low_f32(v), vget_high_f32(v)));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(float32x4_t v)
    {
        return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t>, int8x16_t>
    convert(float32x4_t v)
    {
        int8x8_t i8 = vmovn_s16(convert<int16_t>(v));
        return vcombine_s8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint8_t>, uint8x16_t>
    convert(float32x4_t v)
    {
        uint8x8_t i8 = vmovn_u16(convert<uint16_t>(v));
        return vcombine_u8(i8, i8);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t>, int16x8_t>
    convert(float32x4_t v)
    {
        int16x4_t i16 = vmovn_s32(convert<int32_t>(v));
        return vcombine_s16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint16_t>, uint16x8_t>
    convert(float32x4_t v)
    {
        uint16x4_t i16 = vmovn_u32(convert<uint32_t>(v));
        return vcombine_u16(i16, i16);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t>, int32x4_t>
    convert(float32x4_t v)
    {
        return vcvtq_s32_f32(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint32_t>, uint32x4_t>
    convert(float32x4_t v)
    {
        return vcvtq_u32_f32(convert<float>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t>, int64x2_t>
    convert(float32x4_t v)
    {
        return vcvtq_s64_f64(convert<double>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,uint64_t>, uint64x2_t>
    convert(float32x4_t v)
    {
        return vcvtq_u64_f64(convert<double>(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, float32x4_t>
    load(const std::complex<float>* ptr)
    {
        return vld1q_f32(reinterpret_cast<const float*>(ptr));
    }

    static float32x4_t load1(const std::complex<float>* ptr)
    {
        float32x2_t f32 = vld1_f32((const float*)ptr);
        return vcombine_f32(f32, f32);
    }

    static float32x4_t set1(const std::complex<float>& val)
    {
        return load1(&val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(float32x4_t v, std::complex<float>* ptr)
    {
        vst1q_f32((float*)ptr, v);
    }

    static float32x4_t add(float32x4_t a, float32x4_t b)
    {
        return vaddq_f32(a, b);
    }

    static float32x4_t sub(float32x4_t a, float32x4_t b)
    {
        return vsubq_f32(a, b);
    }

    static float32x4_t mul(float32x4_t a, float32x4_t b)
    {
        float32x4_t ashuf = vrev64q_f32(a);
        float32x4_t breal = vtrn1q_f32(b, b);
        float32x4_t bimag = vtrn2q_f32(vnegq_f32(b), b);
        float32x4_t tmp1 = vmulq_f32(    a, breal); // tmp1 = ( ar0*br0, ai0*br0,  ar1*br1, ai1*br1)
        float32x4_t tmp2 = vmulq_f32(ashuf, bimag); // tmp2 = (-ai0*bi0, ar0*bi0, -ai1*bi1, ar1*bi1)
        return vaddq_f32(tmp1, tmp2);
    }

    static float32x4_t div(float32x4_t a, float32x4_t b)
    {
        std::complex<float> a0{vgetq_lane_f32(a, 0), vgetq_lane_f32(a, 1)};
        std::complex<float> a1{vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 3)};
        std::complex<float> b0{vgetq_lane_f32(b, 0), vgetq_lane_f32(b, 1)};
        std::complex<float> b1{vgetq_lane_f32(b, 2), vgetq_lane_f32(b, 3)};
        std::complex<float> c0 = a0 / b0;
        std::complex<float> c1 = a1 / b1;
        float32x4_t c;
        c = vsetq_lane_f32(c0.real(), c, 0);
        c = vsetq_lane_f32(c0.imag(), c, 1);
        c = vsetq_lane_f32(c1.real(), c, 2);
        c = vsetq_lane_f32(c1.imag(), c, 3);
        return c;
    }

    static float32x4_t pow(float32x4_t a, float32x4_t b)
    {
        std::complex<float> a0{vgetq_lane_f32(a, 0), vgetq_lane_f32(a, 1)};
        std::complex<float> a1{vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 3)};
        std::complex<float> b0{vgetq_lane_f32(b, 0), vgetq_lane_f32(b, 1)};
        std::complex<float> b1{vgetq_lane_f32(b, 2), vgetq_lane_f32(b, 3)};
        std::complex<float> c0 = std::pow(a0, b0);
        std::complex<float> c1 = std::pow(a1, b1);
        float32x4_t c;
        c = vsetq_lane_f32(c0.real(), c, 0);
        c = vsetq_lane_f32(c0.imag(), c, 1);
        c = vsetq_lane_f32(c1.real(), c, 2);
        c = vsetq_lane_f32(c1.imag(), c, 3);
        return c;
    }

    static float32x4_t negate(float32x4_t a)
    {
        return vnegq_f32(a);
    }

    static float32x4_t exp(float32x4_t a)
    {
        std::complex<float> a0{vgetq_lane_f32(a, 0), vgetq_lane_f32(a, 1)};
        std::complex<float> a1{vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 3)};
        std::complex<float> b0 = std::exp(a0);
        std::complex<float> b1 = std::exp(a1);
        float32x4_t b;
        b = vsetq_lane_f32(b0.real(), b, 0);
        b = vsetq_lane_f32(b0.imag(), b, 1);
        b = vsetq_lane_f32(b1.real(), b, 2);
        b = vsetq_lane_f32(b1.imag(), b, 3);
        return b;
    }

    static float32x4_t sqrt(float32x4_t a)
    {
        std::complex<float> a0{vgetq_lane_f32(a, 0), vgetq_lane_f32(a, 1)};
        std::complex<float> a1{vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 3)};
        std::complex<float> b0 = std::sqrt(a0);
        std::complex<float> b1 = std::sqrt(a1);
        float32x4_t b;
        b = vsetq_lane_f32(b0.real(), b, 0);
        b = vsetq_lane_f32(b0.imag(), b, 1);
        b = vsetq_lane_f32(b1.real(), b, 2);
        b = vsetq_lane_f32(b1.imag(), b, 3);
        return b;
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same_v<U,int8_t> ||
                                         std::is_same_v<U,uint8_t>>>
{
    static constexpr int vector_width = 16;
    static constexpr size_t alignment = 16;
    typedef std::conditional_t<std::is_signed_v<U>,int8x16_t,uint8x16_t> vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(vector_type v)
    {
        return vcvtq_f32_s32(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(vector_type v)
    {
        return vcvtq_f64_s64(convert<int64_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(vector_type v)
    {
        return vzip1q_f32(convert<float>(v), vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t> ||
                     std::is_same_v<T,uint8_t>, std::conditional_t<std::is_signed_v<T>,int8x16_t,uint8x16_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<T> && !std::is_signed_v<U>)
            return vreinterpretq_s8_u8(v);
        else if constexpr (!std::is_signed_v<T> && std::is_signed_v<U>)
            return vreinterpretq_u8_s8(v);
        else
            return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t> ||
                     std::is_same_v<T,uint16_t>, std::conditional_t<std::is_signed_v<T>,int16x8_t,uint16x8_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s8(vget_low_s8(v));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u16_s16(ret);
        }
        else
        {
            auto ret = vmovl_u8(vget_low_u8(v));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s16_u16(ret);
            else
                return ret;
        }
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t> ||
                     std::is_same_v<T,uint32_t>, std::conditional_t<std::is_signed_v<T>,int32x4_t,uint32x4_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s16(vget_low_s16(convert<int16_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u32_s32(ret);
        }
        else
        {
            auto ret = vmovl_u16(vget_low_u16(convert<uint16_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s32_u32(ret);
            else
                return ret;
        }
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t> ||
                     std::is_same_v<T,uint64_t>, std::conditional_t<std::is_signed_v<T>,int64x2_t,uint64x2_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s32(vget_low_s32(convert<int32_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u64_s64(ret);
        }
        else
        {
            auto ret = vmovl_u32(vget_low_u32(convert<uint32_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s64_u64(ret);
            else
                return ret;
        }
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16, vector_type>
    load(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_s8(ptr);
        else
            return vld1q_u8(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8, vector_type>
    load(const U* ptr)
    {
        int64x2_t i64 = vld1q_dup_s64((const int64_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s8_s64(i64);
        else
            return vreinterpretq_u8_s64(i64);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, vector_type>
    load(const U* ptr)
    {
        int32x4_t i32 = vld1q_dup_s32((const int32_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s8_s32(i32);
        else
            return vreinterpretq_u8_s32(i32);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, vector_type>
    load(const U* ptr)
    {
        int16x8_t i16 = vld1q_dup_s16((const int16_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s8_s16(i16);
        else
            return vreinterpretq_u8_s16(i16);
    }

    static vector_type load1(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_dup_s8(ptr);
        else
            return vld1q_dup_u8(ptr);
    }

    static vector_type set1(U val)
    {
        if constexpr (std::is_signed_v<U>)
            return vdupq_n_s8(val);
        else
            return vdupq_n_u8(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 16>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1q_s8(ptr, v);
        else
            return vst1q_u8(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1_s8(ptr, vget_low_s8(v));
        else
            return vst1_u8(ptr, vget_low_u8(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            *(int32_t*)ptr = vgetq_lane_s32(vreinterpretq_s32_s8(v), 0);
        else
            *(int32_t*)ptr = vgetq_lane_s32(vreinterpretq_s32_u8(v), 0);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            *(int16_t*)ptr = vgetq_lane_s16(vreinterpretq_s16_s8(v), 0);
        else
            *(int16_t*)ptr = vgetq_lane_s16(vreinterpretq_s16_u8(v), 0);
    }

    static vector_type add(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vaddq_s8(a, b);
        else
            return vaddq_u8(a, b);
    }

    static vector_type sub(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vsubq_s8(a, b);
        else
            return vsubq_u8(a, b);
    }

    static vector_type mul(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vmulq_s8(a, b);
        else
            return vmulq_u8(a, b);
    }

    static vector_type div(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int8x16_t c;
            c = vsetq_lane_s8(vgetq_lane_s8(a,  0) / vgetq_lane_s8(b,  0), c,  0);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  1) / vgetq_lane_s8(b,  1), c,  1);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  2) / vgetq_lane_s8(b,  2), c,  2);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  3) / vgetq_lane_s8(b,  3), c,  3);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  4) / vgetq_lane_s8(b,  4), c,  4);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  5) / vgetq_lane_s8(b,  5), c,  5);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  6) / vgetq_lane_s8(b,  6), c,  6);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  7) / vgetq_lane_s8(b,  7), c,  7);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  8) / vgetq_lane_s8(b,  8), c,  8);
            c = vsetq_lane_s8(vgetq_lane_s8(a,  9) / vgetq_lane_s8(b,  9), c,  9);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 10) / vgetq_lane_s8(b, 10), c, 10);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 11) / vgetq_lane_s8(b, 11), c, 11);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 12) / vgetq_lane_s8(b, 12), c, 12);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 13) / vgetq_lane_s8(b, 13), c, 13);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 14) / vgetq_lane_s8(b, 14), c, 14);
            c = vsetq_lane_s8(vgetq_lane_s8(a, 15) / vgetq_lane_s8(b, 15), c, 15);
            return c;
        }
        else
        {
            uint8x16_t c;
            c = vsetq_lane_u8(vgetq_lane_u8(a,  0) / vgetq_lane_u8(b,  0), c,  0);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  1) / vgetq_lane_u8(b,  1), c,  1);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  2) / vgetq_lane_u8(b,  2), c,  2);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  3) / vgetq_lane_u8(b,  3), c,  3);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  4) / vgetq_lane_u8(b,  4), c,  4);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  5) / vgetq_lane_u8(b,  5), c,  5);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  6) / vgetq_lane_u8(b,  6), c,  6);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  7) / vgetq_lane_u8(b,  7), c,  7);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  8) / vgetq_lane_u8(b,  8), c,  8);
            c = vsetq_lane_u8(vgetq_lane_u8(a,  9) / vgetq_lane_u8(b,  9), c,  9);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 10) / vgetq_lane_u8(b, 10), c, 10);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 11) / vgetq_lane_u8(b, 11), c, 11);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 12) / vgetq_lane_u8(b, 12), c, 12);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 13) / vgetq_lane_u8(b, 13), c, 13);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 14) / vgetq_lane_u8(b, 14), c, 14);
            c = vsetq_lane_u8(vgetq_lane_u8(a, 15) / vgetq_lane_u8(b, 15), c, 15);
            return c;
        }
    }

    static vector_type pow(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int8x16_t c;
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  0), vgetq_lane_s8(b,  0)), c,  0);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  1), vgetq_lane_s8(b,  1)), c,  1);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  2), vgetq_lane_s8(b,  2)), c,  2);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  3), vgetq_lane_s8(b,  3)), c,  3);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  4), vgetq_lane_s8(b,  4)), c,  4);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  5), vgetq_lane_s8(b,  5)), c,  5);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  6), vgetq_lane_s8(b,  6)), c,  6);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  7), vgetq_lane_s8(b,  7)), c,  7);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  8), vgetq_lane_s8(b,  8)), c,  8);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a,  9), vgetq_lane_s8(b,  9)), c,  9);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 10), vgetq_lane_s8(b, 10)), c, 10);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 11), vgetq_lane_s8(b, 11)), c, 11);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 12), vgetq_lane_s8(b, 12)), c, 12);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 13), vgetq_lane_s8(b, 13)), c, 13);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 14), vgetq_lane_s8(b, 14)), c, 14);
            c = vsetq_lane_s8((U)std::pow(vgetq_lane_s8(a, 15), vgetq_lane_s8(b, 15)), c, 15);
            return c;
        }
        else
        {
            uint8x16_t c;
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  0), vgetq_lane_u8(b,  0)), c,  0);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  1), vgetq_lane_u8(b,  1)), c,  1);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  2), vgetq_lane_u8(b,  2)), c,  2);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  3), vgetq_lane_u8(b,  3)), c,  3);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  4), vgetq_lane_u8(b,  4)), c,  4);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  5), vgetq_lane_u8(b,  5)), c,  5);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  6), vgetq_lane_u8(b,  6)), c,  6);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  7), vgetq_lane_u8(b,  7)), c,  7);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  8), vgetq_lane_u8(b,  8)), c,  8);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a,  9), vgetq_lane_u8(b,  9)), c,  9);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 10), vgetq_lane_u8(b, 10)), c, 10);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 11), vgetq_lane_u8(b, 11)), c, 11);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 12), vgetq_lane_u8(b, 12)), c, 12);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 13), vgetq_lane_u8(b, 13)), c, 13);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 14), vgetq_lane_u8(b, 14)), c, 14);
            c = vsetq_lane_u8((U)std::pow(vgetq_lane_u8(a, 15), vgetq_lane_u8(b, 15)), c, 15);
            return c;
        }
    }

    static vector_type negate(vector_type a)
    {
        vnegq_s8(a);
    }

    static vector_type exp(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int8x16_t c;
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  0)), c,  0);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  1)), c,  1);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  2)), c,  2);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  3)), c,  3);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  4)), c,  4);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  5)), c,  5);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  6)), c,  6);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  7)), c,  7);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  8)), c,  8);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a,  9)), c,  9);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 10)), c, 10);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 11)), c, 11);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 12)), c, 12);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 13)), c, 13);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 14)), c, 14);
            c = vsetq_lane_s8((U)std::exp(vgetq_lane_s8(a, 15)), c, 15);
            return c;
        }
        else
        {
            uint8x16_t c;
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  0)), c,  0);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  1)), c,  1);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  2)), c,  2);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  3)), c,  3);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  4)), c,  4);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  5)), c,  5);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  6)), c,  6);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  7)), c,  7);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  8)), c,  8);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a,  9)), c,  9);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 10)), c, 10);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 11)), c, 11);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 12)), c, 12);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 13)), c, 13);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 14)), c, 14);
            c = vsetq_lane_u8((U)std::exp(vgetq_lane_u8(a, 15)), c, 15);
            return c;
        }
    }

    static vector_type sqrt(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int8x16_t c;
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  0)), c,  0);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  1)), c,  1);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  2)), c,  2);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  3)), c,  3);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  4)), c,  4);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  5)), c,  5);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  6)), c,  6);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  7)), c,  7);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  8)), c,  8);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a,  9)), c,  9);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 10)), c, 10);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 11)), c, 11);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 12)), c, 12);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 13)), c, 13);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 14)), c, 14);
            c = vsetq_lane_s8((U)std::sqrt(vgetq_lane_s8(a, 15)), c, 15);
            return c;
        }
        else
        {
            uint8x16_t c;
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  0)), c,  0);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  1)), c,  1);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  2)), c,  2);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  3)), c,  3);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  4)), c,  4);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  5)), c,  5);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  6)), c,  6);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  7)), c,  7);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  8)), c,  8);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a,  9)), c,  9);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 10)), c, 10);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 11)), c, 11);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 12)), c, 12);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 13)), c, 13);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 14)), c, 14);
            c = vsetq_lane_u8((U)std::sqrt(vgetq_lane_u8(a, 15)), c, 15);
            return c;
        }
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same_v<U,int16_t> ||
                                         std::is_same_v<U,uint16_t>>>
{
    static constexpr int vector_width = 8;
    static constexpr size_t alignment = 16;
    typedef std::conditional_t<std::is_signed_v<U>,int16x8_t,uint16x8_t> vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(vector_type v)
    {
        return vcvtq_f32_s32(convert<int32_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(vector_type v)
    {
        return vcvtq_f64_s64(convert<int64_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(vector_type v)
    {
        return vzip1q_f32(convert<float>(v), vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t> ||
                     std::is_same_v<T,uint8_t>, std::conditional_t<std::is_signed_v<T>,int8x16_t,uint8x16_t>>
    convert(vector_type v)
    {
        int8x8_t i8;
        if constexpr (std::is_signed_v<U>)
            i8 = vmovn_s16(v);
        else
            i8 = vreinterpret_s8_u8(vmovn_u16(v));
        auto ret = vcombine_s8(i8, i8);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u8_s8(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t> ||
                     std::is_same_v<T,uint16_t>, std::conditional_t<std::is_signed_v<T>,int16x8_t,uint16x8_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<T> && !std::is_signed_v<U>)
            return vreinterpretq_s16_u16(v);
        else if constexpr (!std::is_signed_v<T> && std::is_signed_v<U>)
            return vreinterpretq_u16_s16(v);
        else
            return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t> ||
                     std::is_same_v<T,uint32_t>, std::conditional_t<std::is_signed_v<T>,int32x4_t,uint32x4_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s16(vget_low_s16(v));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u32_s32(ret);
        }
        else
        {
            auto ret = vmovl_u16(vget_low_u16(v));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s32_u32(ret);
            else
                return ret;
        }
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t> ||
                     std::is_same_v<T,uint64_t>, std::conditional_t<std::is_signed_v<T>,int64x2_t,uint64x2_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s32(vget_low_s32(convert<int32_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u64_s64(ret);
        }
        else
        {
            auto ret = vmovl_u32(vget_low_u32(convert<uint32_t>(v)));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s64_u64(ret);
            else
                return ret;
        }
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8, vector_type>
    load(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_s16(ptr);
        else
            return vld1q_u16(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, vector_type>
    load(const U* ptr)
    {
        int64x2_t i64 = vld1q_dup_s64((const int64_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s16_s64(i64);
        else
            return vreinterpretq_u16_s64(i64);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, vector_type>
    load(const U* ptr)
    {
        int32x4_t i32 = vld1q_dup_s32((const int32_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s16_s32(i32);
        else
            return vreinterpretq_u16_s32(i32);
    }

    static vector_type load1(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_dup_s16(ptr);
        else
            return vld1q_dup_u16(ptr);
    }

    static vector_type set1(U val)
    {
        if constexpr (std::is_signed_v<U>)
            return vdupq_n_s16(val);
        else
            return vdupq_n_u16(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 8>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1q_s16(ptr, v);
        else
            return vst1q_u16(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1_s16(ptr, vget_low_s16(v));
        else
            return vst1_u16(ptr, vget_low_u16(v));
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            *(int32_t*)ptr = vgetq_lane_s32(vreinterpretq_s32_s16(v), 0);
        else
            *(int32_t*)ptr = vgetq_lane_s32(vreinterpretq_s32_u16(v), 0);
    }

    static vector_type add(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vaddq_s16(a, b);
        else
            return vaddq_u16(a, b);
    }

    static vector_type sub(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vsubq_s16(a, b);
        else
            return vsubq_u16(a, b);
    }

    static vector_type mul(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vmulq_s16(a, b);
        else
            return vmulq_u16(a, b);
    }

    static vector_type div(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int16x8_t c;
            c = vsetq_lane_s16(vgetq_lane_s16(a, 0) / vgetq_lane_s16(b, 0), c, 0);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 1) / vgetq_lane_s16(b, 1), c, 1);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 2) / vgetq_lane_s16(b, 2), c, 2);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 3) / vgetq_lane_s16(b, 3), c, 3);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 4) / vgetq_lane_s16(b, 4), c, 4);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 5) / vgetq_lane_s16(b, 5), c, 5);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 6) / vgetq_lane_s16(b, 6), c, 6);
            c = vsetq_lane_s16(vgetq_lane_s16(a, 7) / vgetq_lane_s16(b, 7), c, 7);
            return c;
        }
        else
        {
            uint16x8_t c;
            c = vsetq_lane_u16(vgetq_lane_u16(a, 0) / vgetq_lane_u16(b, 0), c, 0);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 1) / vgetq_lane_u16(b, 1), c, 1);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 2) / vgetq_lane_u16(b, 2), c, 2);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 3) / vgetq_lane_u16(b, 3), c, 3);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 4) / vgetq_lane_u16(b, 4), c, 4);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 5) / vgetq_lane_u16(b, 5), c, 5);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 6) / vgetq_lane_u16(b, 6), c, 6);
            c = vsetq_lane_u16(vgetq_lane_u16(a, 7) / vgetq_lane_u16(b, 7), c, 7);
            return c;
        }
    }

    static vector_type pow(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int16x8_t c;
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 0), vgetq_lane_s16(b, 0)), c, 0);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 1), vgetq_lane_s16(b, 1)), c, 1);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 2), vgetq_lane_s16(b, 2)), c, 2);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 3), vgetq_lane_s16(b, 3)), c, 3);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 4), vgetq_lane_s16(b, 4)), c, 4);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 5), vgetq_lane_s16(b, 5)), c, 5);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 6), vgetq_lane_s16(b, 6)), c, 6);
            c = vsetq_lane_s16((U)std::pow(vgetq_lane_s16(a, 7), vgetq_lane_s16(b, 7)), c, 7);
            return c;
        }
        else
        {
            uint16x8_t c;
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 0), vgetq_lane_u16(b, 0)), c, 0);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 1), vgetq_lane_u16(b, 1)), c, 1);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 2), vgetq_lane_u16(b, 2)), c, 2);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 3), vgetq_lane_u16(b, 3)), c, 3);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 4), vgetq_lane_u16(b, 4)), c, 4);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 5), vgetq_lane_u16(b, 5)), c, 5);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 6), vgetq_lane_u16(b, 6)), c, 6);
            c = vsetq_lane_u16((U)std::pow(vgetq_lane_u16(a, 7), vgetq_lane_u16(b, 7)), c, 7);
            return c;
        }
    }

    static vector_type negate(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
            return vnegq_s16(a);
        else
            return vnegq_s16(vreinterpretq_s16_u16(a));
    }

    static vector_type exp(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int16x8_t c;
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 0)), c, 0);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 1)), c, 1);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 2)), c, 2);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 3)), c, 3);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 4)), c, 4);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 5)), c, 5);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 6)), c, 6);
            c = vsetq_lane_s16((U)std::exp(vgetq_lane_s16(a, 7)), c, 7);
            return c;
        }
        else
        {
            uint16x8_t c;
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 0)), c, 0);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 1)), c, 1);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 2)), c, 2);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 3)), c, 3);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 4)), c, 4);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 5)), c, 5);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 6)), c, 6);
            c = vsetq_lane_u16((U)std::exp(vgetq_lane_u16(a, 7)), c, 7);
            return c;
        }
    }

    static vector_type sqrt(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int16x8_t c;
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 0)), c, 0);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 1)), c, 1);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 2)), c, 2);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 3)), c, 3);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 4)), c, 4);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 5)), c, 5);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 6)), c, 6);
            c = vsetq_lane_s16((U)std::sqrt(vgetq_lane_s16(a, 7)), c, 7);
            return c;
        }
        else
        {
            uint16x8_t c;
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 0)), c, 0);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 1)), c, 1);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 2)), c, 2);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 3)), c, 3);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 4)), c, 4);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 5)), c, 5);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 6)), c, 6);
            c = vsetq_lane_u16((U)std::sqrt(vgetq_lane_u16(a, 7)), c, 7);
            return c;
        }
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same_v<U,int32_t> ||
                                         std::is_same_v<U,uint32_t>>>
{
    static constexpr int vector_width = 4;
    static constexpr size_t alignment = 16;
    typedef std::conditional_t<std::is_signed_v<U>,int32x4_t,uint32x4_t> vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
            return vcvtq_f32_s32(v);
        else
            return vcvtq_f32_u32(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(vector_type v)
    {
        return vcvtq_f64_s64(convert<int64_t>(v));
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(vector_type v)
    {
        return vzip1q_f32(convert<float>(v), vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t> ||
                     std::is_same_v<T,uint8_t>, std::conditional_t<std::is_signed_v<T>,int8x16_t,uint8x16_t>>
    convert(vector_type v)
    {
        int8x8_t i8;
        if constexpr (std::is_signed_v<U>)
            i8 = vmovn_s16(convert<int16_t>(v));
        else
            i8 = vreinterpret_s8_u8(vmovn_u16(convert<uint16_t>(v)));
        auto ret = vcombine_s8(i8, i8);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u8_s8(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t> ||
                     std::is_same_v<T,uint16_t>, std::conditional_t<std::is_signed_v<T>,int16x8_t,uint16x8_t>>
    convert(vector_type v)
    {
        int16x4_t i16;
        if constexpr (std::is_signed_v<U>)
            i16 = vmovn_s32(v);
        else
            i16 = vreinterpret_s16_u16(vmovn_u32(v));
        auto ret = vcombine_s16(i16, i16);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u16_s16(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t> ||
                     std::is_same_v<T,uint32_t>, std::conditional_t<std::is_signed_v<T>,int32x4_t,uint32x4_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<T> && !std::is_signed_v<U>)
            return vreinterpretq_s32_u32(v);
        else if constexpr (!std::is_signed_v<T> && std::is_signed_v<U>)
            return vreinterpretq_u32_s32(v);
        else
            return v;
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t> ||
                     std::is_same_v<T,uint64_t>, std::conditional_t<std::is_signed_v<T>,int64x2_t,uint64x2_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
        {
            auto ret = vmovl_s32(vget_low_s32(v));
            if constexpr (std::is_signed_v<T>)
                return ret;
            else
                return vreinterpretq_u64_s64(ret);
        }
        else
        {
            auto ret = vmovl_u32(vget_low_u32(v));
            if constexpr (std::is_signed_v<T>)
                return vreinterpretq_s64_u64(ret);
            else
                return ret;
        }
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4, vector_type>
    load(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_s32(ptr);
        else
            return vld1q_u32(ptr);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, vector_type>
    load(const U* ptr)
    {
        int64x2_t i64 = vld1q_dup_s64((const int64_t*)ptr);

        if constexpr (std::is_signed_v<U>)
            return vreinterpretq_s32_s64(i64);
        else
            return vreinterpretq_u32_s64(i64);
    }

    static vector_type load1(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_dup_s32(ptr);
        else
            return vld1q_dup_u32(ptr);
    }

    static vector_type set1(U val)
    {
        if constexpr (std::is_signed_v<U>)
            return vdupq_n_s32(val);
        else
            return vdupq_n_u32(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 4>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1q_s32(ptr, v);
        else
            return vst1q_u32(ptr, v);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1_s32(ptr, vget_low_s32(v));
        else
            return vst1_u32(ptr, vget_low_u32(v));
    }

    static vector_type add(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vaddq_s32(a, b);
        else
            return vaddq_u32(a, b);
    }

    static vector_type sub(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vsubq_s32(a, b);
        else
            return vsubq_u32(a, b);
    }

    static vector_type mul(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vmulq_s32(a, b);
        else
            return vmulq_u32(a, b);
    }

    static vector_type div(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int32x4_t c;
            c = vsetq_lane_s32(vgetq_lane_s32(a, 0) / vgetq_lane_s32(b, 0), c, 0);
            c = vsetq_lane_s32(vgetq_lane_s32(a, 1) / vgetq_lane_s32(b, 1), c, 1);
            c = vsetq_lane_s32(vgetq_lane_s32(a, 2) / vgetq_lane_s32(b, 2), c, 2);
            c = vsetq_lane_s32(vgetq_lane_s32(a, 3) / vgetq_lane_s32(b, 3), c, 3);
            return c;
        }
        else
        {
            uint32x4_t c;
            c = vsetq_lane_u32(vgetq_lane_u32(a, 0) / vgetq_lane_u32(b, 0), c, 0);
            c = vsetq_lane_u32(vgetq_lane_u32(a, 1) / vgetq_lane_u32(b, 1), c, 1);
            c = vsetq_lane_u32(vgetq_lane_u32(a, 2) / vgetq_lane_u32(b, 2), c, 2);
            c = vsetq_lane_u32(vgetq_lane_u32(a, 3) / vgetq_lane_u32(b, 3), c, 3);
            return c;
        }
    }

    static vector_type pow(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int32x4_t c;
            c = vsetq_lane_s32((U)std::pow(vgetq_lane_s32(a, 0), vgetq_lane_s32(b, 0)), c, 0);
            c = vsetq_lane_s32((U)std::pow(vgetq_lane_s32(a, 1), vgetq_lane_s32(b, 1)), c, 1);
            c = vsetq_lane_s32((U)std::pow(vgetq_lane_s32(a, 2), vgetq_lane_s32(b, 2)), c, 2);
            c = vsetq_lane_s32((U)std::pow(vgetq_lane_s32(a, 3), vgetq_lane_s32(b, 3)), c, 3);
            return c;
        }
        else
        {
            uint32x4_t c;
            c = vsetq_lane_u32((U)std::pow(vgetq_lane_u32(a, 0), vgetq_lane_u32(b, 0)), c, 0);
            c = vsetq_lane_u32((U)std::pow(vgetq_lane_u32(a, 1), vgetq_lane_u32(b, 1)), c, 1);
            c = vsetq_lane_u32((U)std::pow(vgetq_lane_u32(a, 2), vgetq_lane_u32(b, 2)), c, 2);
            c = vsetq_lane_u32((U)std::pow(vgetq_lane_u32(a, 3), vgetq_lane_u32(b, 3)), c, 3);
            return c;
        }
    }

    static vector_type negate(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
            return vnegq_s32(a);
        else
            return vnegq_s32(vreinterpretq_s32_u32(a));
    }

    static vector_type exp(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int32x4_t c;
            c = vsetq_lane_s32((U)std::exp(vgetq_lane_s32(a, 0)), c, 0);
            c = vsetq_lane_s32((U)std::exp(vgetq_lane_s32(a, 1)), c, 1);
            c = vsetq_lane_s32((U)std::exp(vgetq_lane_s32(a, 2)), c, 2);
            c = vsetq_lane_s32((U)std::exp(vgetq_lane_s32(a, 3)), c, 3);
            return c;
        }
        else
        {
            uint32x4_t c;
            c = vsetq_lane_u32((U)std::exp(vgetq_lane_u32(a, 0)), c, 0);
            c = vsetq_lane_u32((U)std::exp(vgetq_lane_u32(a, 1)), c, 1);
            c = vsetq_lane_u32((U)std::exp(vgetq_lane_u32(a, 2)), c, 2);
            c = vsetq_lane_u32((U)std::exp(vgetq_lane_u32(a, 3)), c, 3);
            return c;
        }
    }

    static vector_type sqrt(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int32x4_t c;
            c = vsetq_lane_s32((U)std::sqrt(vgetq_lane_s32(a, 0)), c, 0);
            c = vsetq_lane_s32((U)std::sqrt(vgetq_lane_s32(a, 1)), c, 1);
            c = vsetq_lane_s32((U)std::sqrt(vgetq_lane_s32(a, 2)), c, 2);
            c = vsetq_lane_s32((U)std::sqrt(vgetq_lane_s32(a, 3)), c, 3);
            return c;
        }
        else
        {
            uint32x4_t c;
            c = vsetq_lane_u32((U)std::sqrt(vgetq_lane_u32(a, 0)), c, 0);
            c = vsetq_lane_u32((U)std::sqrt(vgetq_lane_u32(a, 1)), c, 1);
            c = vsetq_lane_u32((U)std::sqrt(vgetq_lane_u32(a, 2)), c, 2);
            c = vsetq_lane_u32((U)std::sqrt(vgetq_lane_u32(a, 3)), c, 3);
            return c;
        }
    }
};

template <typename U>
struct vector_traits<U, std::enable_if_t<std::is_same_v<U,int64_t> ||
                                         std::is_same_v<U,uint64_t>>>
{
    static constexpr int vector_width = 2;
    static constexpr size_t alignment = 16;
    typedef std::conditional_t<std::is_signed_v<U>,int64x2_t,uint64x2_t> vector_type;

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,float>, float32x4_t>
    convert(vector_type v)
    {
        float32x2_t f32 = vcvt_f32_f64(convert<double>(v));
        return vcombine_f32(f32, f32);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,double>, float64x2_t>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<U>)
            return vcvtq_f64_s64(v);
        else
            return vcvtq_f64_u64(v);
    }

    template <typename T> static
    std::enable_if_t<std::is_same<T,std::complex<float>>::value, float32x4_t>
    convert(vector_type v)
    {
        return vzip1q_f32(convert<float>(v), vdupq_n_f32(0.0));
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int8_t> ||
                     std::is_same_v<T,uint8_t>, std::conditional_t<std::is_signed_v<T>,int8x16_t,uint8x16_t>>
    convert(vector_type v)
    {
        int8x8_t i8;
        if constexpr (std::is_signed_v<U>)
            i8 = vmovn_s16(convert<int16_t>(v));
        else
            i8 = vreinterpret_s8_u8(vmovn_u16(convert<uint16_t>(v)));
        auto ret = vcombine_s8(i8, i8);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u8_s8(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int16_t> ||
                     std::is_same_v<T,uint16_t>, std::conditional_t<std::is_signed_v<T>,int16x8_t,uint16x8_t>>
    convert(vector_type v)
    {
        int16x4_t i16;
        if constexpr (std::is_signed_v<U>)
            i16 = vmovn_s32(convert<int32_t>(v));
        else
            i16 = vreinterpret_s16_u16(vmovn_u32(convert<uint32_t>(v)));
        auto ret = vcombine_s16(i16, i16);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u16_s16(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int32_t> ||
                     std::is_same_v<T,uint32_t>, std::conditional_t<std::is_signed_v<T>,int32x4_t,uint32x4_t>>
    convert(vector_type v)
    {
        int32x2_t i32;
        if constexpr (std::is_signed_v<U>)
            i32 = vmovn_s64(v);
        else
            i32 = vreinterpret_s32_u32(vmovn_u64(v));
        auto ret = vcombine_s32(i32, i32);
        if constexpr (std::is_signed_v<T>)
            return ret;
        else
            return vreinterpretq_u32_s32(ret);
    }

    template <typename T> static
    std::enable_if_t<std::is_same_v<T,int64_t> ||
                     std::is_same_v<T,uint64_t>, std::conditional_t<std::is_signed_v<T>,int64x2_t,uint64x2_t>>
    convert(vector_type v)
    {
        if constexpr (std::is_signed_v<T> && !std::is_signed_v<U>)
            return vreinterpretq_s64_u64(v);
        else if constexpr (!std::is_signed_v<T> && std::is_signed_v<U>)
            return vreinterpretq_u64_s64(v);
        else
            return v;
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2, vector_type>
    load(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_s64(ptr);
        else
            return vld1q_u64(ptr);
    }

    static vector_type load1(const U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vld1q_dup_s64(ptr);
        else
            return vld1q_dup_u64(ptr);
    }

    static vector_type set1(U val)
    {
        if constexpr (std::is_signed_v<U>)
            return vdupq_n_s64(val);
        else
            return vdupq_n_u64(val);
    }

    template <int Width, bool Aligned> static
    std::enable_if_t<Width == 2>
    store(vector_type v, U* ptr)
    {
        if constexpr (std::is_signed_v<U>)
            return vst1q_s64(ptr, v);
        else
            return vst1q_u64(ptr, v);
    }

    static vector_type add(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vaddq_s64(a, b);
        else
            return vaddq_u64(a, b);
    }

    static vector_type sub(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
            return vsubq_s64(a, b);
        else
            return vsubq_u64(a, b);
    }

    static vector_type mul(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int64x2_t c;
            c = vsetq_lane_s64(vgetq_lane_s64(a, 0) * vgetq_lane_s64(b, 0), c, 0);
            c = vsetq_lane_s64(vgetq_lane_s64(a, 1) * vgetq_lane_s64(b, 1), c, 1);
            return c;
        }
        else
        {
            uint64x2_t c;
            c = vsetq_lane_u64(vgetq_lane_u64(a, 0) * vgetq_lane_u64(b, 0), c, 0);
            c = vsetq_lane_u64(vgetq_lane_u64(a, 1) * vgetq_lane_u64(b, 1), c, 1);
            return c;
        }
    }

    static vector_type div(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int64x2_t c;
            c = vsetq_lane_s64(vgetq_lane_s64(a, 0) / vgetq_lane_s64(b, 0), c, 0);
            c = vsetq_lane_s64(vgetq_lane_s64(a, 1) / vgetq_lane_s64(b, 1), c, 1);
            return c;
        }
        else
        {
            uint64x2_t c;
            c = vsetq_lane_u64(vgetq_lane_u64(a, 0) / vgetq_lane_u64(b, 0), c, 0);
            c = vsetq_lane_u64(vgetq_lane_u64(a, 1) / vgetq_lane_u64(b, 1), c, 1);
            return c;
        }
    }

    static vector_type pow(vector_type a, vector_type b)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int64x2_t c;
            c = vsetq_lane_s64((U)std::pow(vgetq_lane_s64(a, 0), vgetq_lane_s64(b, 0)), c, 0);
            c = vsetq_lane_s64((U)std::pow(vgetq_lane_s64(a, 1), vgetq_lane_s64(b, 1)), c, 1);
            return c;
        }
        else
        {
            uint64x2_t c;
            c = vsetq_lane_u64((U)std::pow(vgetq_lane_u64(a, 0), vgetq_lane_u64(b, 0)), c, 0);
            c = vsetq_lane_u64((U)std::pow(vgetq_lane_u64(a, 1), vgetq_lane_u64(b, 1)), c, 1);
            return c;
        }
    }

    static vector_type negate(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
            return vnegq_s64(a);
        else
            return vnegq_s64(vreinterpretq_s64_u64(a));
    }

    static vector_type exp(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int64x2_t c;
            c = vsetq_lane_s64((U)std::exp(vgetq_lane_s64(a, 0)), c, 0);
            c = vsetq_lane_s64((U)std::exp(vgetq_lane_s64(a, 1)), c, 1);
            return c;
        }
        else
        {
            uint64x2_t c;
            c = vsetq_lane_u64((U)std::exp(vgetq_lane_u64(a, 0)), c, 0);
            c = vsetq_lane_u64((U)std::exp(vgetq_lane_u64(a, 1)), c, 1);
            return c;
        }
    }

    static vector_type sqrt(vector_type a)
    {
        if constexpr (std::is_signed_v<U>)
        {
            int64x2_t c;
            c = vsetq_lane_s64((U)std::sqrt(vgetq_lane_s64(a, 0)), c, 0);
            c = vsetq_lane_s64((U)std::sqrt(vgetq_lane_s64(a, 1)), c, 1);
            return c;
        }
        else
        {
            uint64x2_t c;
            c = vsetq_lane_u64((U)std::sqrt(vgetq_lane_u64(a, 0)), c, 0);
            c = vsetq_lane_u64((U)std::sqrt(vgetq_lane_u64(a, 1)), c, 1);
            return c;
        }
    }
};

}

#endif

#endif //MARRAY_VECTOR_NEON_HPP
