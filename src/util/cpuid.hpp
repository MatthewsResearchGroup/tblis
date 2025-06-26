#ifndef _TBLIS_CPUID_HPP_
#define _TBLIS_CPUID_HPP_

#include <cstring>
#include <cstdio>

#include "basic_types.h"

inline bool check_features(int have, int want)
{
    return (have&want) == want;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#include <cpuid.h>

namespace tblis
{

enum {VENDOR_INTEL, VENDOR_AMD, VENDOR_UNKNOWN};
enum {FEATURE_SSE3     = 0x0001,
      FEATURE_SSSE3    = 0x0002,
      FEATURE_SSE41    = 0x0004,
      FEATURE_SSE42    = 0x0008,
      FEATURE_AVX      = 0x0010,
      FEATURE_AVX2     = 0x0020,
      FEATURE_FMA3     = 0x0040,
      FEATURE_FMA4     = 0x0080,
      FEATURE_AVX512F  = 0x0100,
      FEATURE_AVX512PF = 0x0200,
      FEATURE_AVX512DQ = 0x0400,
      FEATURE_AVX512BW = 0x0800,
      FEATURE_AVX512VL = 0x1000};

int get_cpu_type(int& family, int& model, int& features);

}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

namespace tblis
{

enum {VENDOR_ARM, VENDOR_UNKNOWN};
enum {MODEL_ARMV7, MODEL_ARMV8, MODEL_UNKNOWN};
enum {FEATURE_NEON = 0x1};

int get_cpu_type(int& model, int& part, int& features);

}

#endif

#endif
