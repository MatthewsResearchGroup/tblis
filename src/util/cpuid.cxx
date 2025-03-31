#include "cpuid.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

namespace tblis
{

enum {FEATURE_MASK_SSE3     = (1u<< 0), //CPUID[EAX=1]:ECX[0]
      FEATURE_MASK_SSSE3    = (1u<< 9), //CPUID[EAX=1]:ECX[9]
      FEATURE_MASK_SSE41    = (1u<<19), //CPUID[EAX=1]:ECX[19]
      FEATURE_MASK_SSE42    = (1u<<20), //CPUID[EAX=1]:ECX[20]
      FEATURE_MASK_AVX      = (1u<<28), //CPUID[EAX=1]:ECX[28]
      FEATURE_MASK_AVX2     = (1u<< 5), //CPUID[EAX=7,ECX=0]:EBX[5]
      FEATURE_MASK_FMA3     = (1u<<12), //CPUID[EAX=1]:ECX[12]
      FEATURE_MASK_FMA4     = (1u<<16), //CPUID[EAX=0x80000001]:ECX[16]
      FEATURE_MASK_AVX512F  = (1u<<16), //CPUID[EAX=7,ECX=0]:EBX[16]
      FEATURE_MASK_AVX512PF = (1u<<26), //CPUID[EAX=7,ECX=0]:EBX[26]
      FEATURE_MASK_AVX512DQ = (1u<<17), //CPUID[EAX=7,ECX=0]:EBX[17]
      FEATURE_MASK_AVX512BW = (1u<<30), //CPUID[EAX=7,ECX=0]:EBX[30]
      FEATURE_MASK_AVX512VL = (1u<<31), //CPUID[EAX=7,ECX=0]:EBX[31]
      FEATURE_MASK_XGETBV   = (1u<<26)|
                              (1u<<27), //CPUID[EAX=1]:ECX[27:26]
      XGETBV_MASK_XMM       = 0x02u,     //XCR0[1]
      XGETBV_MASK_YMM       = 0x04u,     //XCR0[2]

      XGETBV_MASK_ZMM       = 0xE0u};    //XCR0[7:5]

/*
static void print_binary(uint32_t x)
{
    uint32_t mask = (1u<<31);

    while (mask)
    {
        for (int i = 0;i < 4;i++)
        {
            fprintf(stderr, "%d", !!(x&mask));
            mask >>= 1;
        }
        fprintf(stderr, " ");
    }
    fprintf(stderr, "\n");
}
*/

int get_cpu_type(int& family, int& model, int& features)
{
    uint32_t eax, ebx, ecx, edx;

    family = model = features = 0;

    //fprintf(stderr, "checking cpuid\n");

    unsigned cpuid_max = __get_cpuid_max(0, 0);
    unsigned cpuid_max_ext = __get_cpuid_max(0x80000000u, 0);

    //fprintf(stderr, "max cpuid leaf: %d\n", cpuid_max);
    //fprintf(stderr, "max extended cpuid leaf: %08x\n", cpuid_max_ext);

    if (cpuid_max < 1) return VENDOR_UNKNOWN;

    uint32_t vendor_string[4] = {0};
    __cpuid(0, eax, vendor_string[0],
                    vendor_string[2],
                    vendor_string[1]);

    if (cpuid_max >= 7)
    {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        //fprintf(stderr, "cpuid leaf 7:\n");
        //print_binary(eax);
        //print_binary(ebx);
        //print_binary(ecx);
        //print_binary(edx);

        if (check_features(ebx, FEATURE_MASK_AVX2)) features |= FEATURE_AVX2;
        if (check_features(ebx, FEATURE_MASK_AVX512F)) features |= FEATURE_AVX512F;
        if (check_features(ebx, FEATURE_MASK_AVX512PF)) features |= FEATURE_AVX512PF;
        if (check_features(ebx, FEATURE_MASK_AVX512DQ)) features |= FEATURE_AVX512DQ;
        if (check_features(ebx, FEATURE_MASK_AVX512BW)) features |= FEATURE_AVX512BW;
        if (check_features(ebx, FEATURE_MASK_AVX512VL)) features |= FEATURE_AVX512VL;
    }

    if (cpuid_max_ext >= 0x80000001u)
    {
        __cpuid(0x80000001u, eax, ebx, ecx, edx);
        //fprintf(stderr, "extended cpuid leaf 0x80000001:\n");
        //print_binary(eax);
        //print_binary(ebx);
        //print_binary(ecx);
        //print_binary(edx);

        if (check_features(ecx, FEATURE_MASK_FMA4)) features |= FEATURE_FMA4;
    }

    __cpuid(1, eax, ebx, ecx, edx);
    //fprintf(stderr, "cpuid leaf 1:\n");
    //print_binary(eax);
    //print_binary(ebx);
    //print_binary(ecx);
    //print_binary(edx);

    family = (eax>>8)&(0xF); // bits 11:8
    model = (eax>>4)&(0xF); // bits 7:4

    if (family == 6)
    {
        model += (eax>>12)&(0xF<<4); // bits 19:16 shifted left by 4
    }
    else if (family == 15)
    {
        family += (eax>>20)&(0xFF); // bits 27:20
        model += (eax>>12)&(0xF<<4); // bits 19:16 shifted left by 4
    }

    if (check_features(ecx, FEATURE_MASK_SSE3)) features |= FEATURE_SSE3;
    if (check_features(ecx, FEATURE_MASK_SSSE3)) features |= FEATURE_SSSE3;
    if (check_features(ecx, FEATURE_MASK_SSE41)) features |= FEATURE_SSE41;
    if (check_features(ecx, FEATURE_MASK_SSE42)) features |= FEATURE_SSE42;
    if (check_features(ecx, FEATURE_MASK_AVX)) features |= FEATURE_AVX;
    if (check_features(ecx, FEATURE_MASK_FMA3)) features |= FEATURE_FMA3;

    if (check_features(ecx, FEATURE_MASK_XGETBV))
    {
        // call xgetbv to get XCR0
        int xcr = 0;
        __asm__ __volatile__
        (
            ".byte 0x0F, 0x01, 0xD0"
            : "=a" (eax), "=d" (edx)
            : "c" (xcr)
            : "cc"
        );

        //fprintf(stderr, "xcr0:\n");
        //print_binary(eax);
        //print_binary(edx);

        //fprintf(stderr, "xgetbv: xmm: %d\n", check_features(eax, XGETBV_MASK_XMM));
        //fprintf(stderr, "xgetbv: ymm: %d\n", check_features(eax, XGETBV_MASK_XMM|
        //                                                XGETBV_MASK_YMM));
        //fprintf(stderr, "xgetbv: zmm: %d\n", check_features(eax, XGETBV_MASK_XMM|
        //                                                XGETBV_MASK_YMM|
        //                                                XGETBV_MASK_ZMM));

        if (!check_features(eax, XGETBV_MASK_XMM|
                                 XGETBV_MASK_YMM|
                                 XGETBV_MASK_ZMM))
        {
            features &= ~(FEATURE_AVX512F|
                          FEATURE_AVX512PF|
                          FEATURE_AVX512DQ|
                          FEATURE_AVX512BW|
                          FEATURE_AVX512VL);
        }

        if (!check_features(eax, XGETBV_MASK_XMM|
                                 XGETBV_MASK_YMM))
        {
            features &= ~(FEATURE_AVX|
                          FEATURE_AVX2|
                          FEATURE_FMA3|
                          FEATURE_FMA4);
        }

        if (!check_features(eax, XGETBV_MASK_XMM))
        {
            features = 0;
        }
    }
    else
    {
        //fprintf(stderr, "xgetbv: no\n");
        features = 0;
    }

    //fprintf(stderr, "vendor: %12s\n", vendor_string);
    //fprintf(stderr, "family: %d\n", family);
    //fprintf(stderr, "model: %d\n", model);
    //fprintf(stderr, "sse3: %d\n", check_features(features, FEATURE_SSE3));
    //fprintf(stderr, "ssse3: %d\n", check_features(features, FEATURE_SSSE3));
    //fprintf(stderr, "sse4.1: %d\n", check_features(features, FEATURE_SSE41));
    //fprintf(stderr, "sse4.2: %d\n", check_features(features, FEATURE_SSE42));
    //fprintf(stderr, "avx: %d\n", check_features(features, FEATURE_AVX));
    //fprintf(stderr, "avx2: %d\n", check_features(features, FEATURE_AVX2));
    //fprintf(stderr, "fma3: %d\n", check_features(features, FEATURE_FMA3));
    //fprintf(stderr, "fma4: %d\n", check_features(features, FEATURE_FMA4));
    //fprintf(stderr, "avx512f: %d\n", check_features(features, FEATURE_AVX512F));
    //fprintf(stderr, "avx512pf: %d\n", check_features(features, FEATURE_AVX512PF));
    //fprintf(stderr, "avx512dq: %d\n", check_features(features, FEATURE_AVX512DQ));

    if (strcmp(reinterpret_cast<char*>(&vendor_string[0]), "AuthenticAMD") == 0)
        return VENDOR_AMD;
    else if (strcmp(reinterpret_cast<char*>(&vendor_string[0]), "GenuineIntel") == 0)
        return VENDOR_INTEL;
    else
        return VENDOR_UNKNOWN;
}

}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

namespace tblis
{

int get_cpu_type(int& model, int& part, int& features)
{
    model = MODEL_UNKNOWN;
    features = 0;

    FILE *fd1 = popen("grep -m 1 Processor /proc/cpuinfo", "r");
    if (!fd1) return VENDOR_ARM;
    FILE *fd2 = popen("grep -m 1 'CPU part' /proc/cpuinfo", "r");
    if (!fd2)
    {
        pclose(fd1);
        return VENDOR_ARM;
    }
    FILE *fd3 = popen("grep -m 1 Features /proc/cpuinfo", "r");
    if (!fd3)
    {
        pclose(fd1);
        pclose(fd2);
        return VENDOR_ARM;
    }

    char c;
    std::string proc, ptno, feat;
    while ((c = fgetc(fd1)) != EOF) proc.push_back(c);
    while ((c = fgetc(fd2)) != EOF) ptno.push_back(c);
    while ((c = fgetc(fd3)) != EOF) feat.push_back(c);

    pclose(fd1);
    pclose(fd2);
    pclose(fd3);

    if (feat.find("neon") != std::string::npos ||
        feat.find("asimd") != std::string::npos)
        features |= FEATURE_NEON;

    if (proc.find("ARMv7") != std::string::npos)
        model = MODEL_ARMV7;
    else if (proc.find("AArch64") != std::string::npos)
        model = MODEL_ARMV8;

    auto pos = ptno.find("0x");
    TBLIS_ASSERT(pos != std::string::npos);
#if defined(__APPLE__) && defined(__MACH__)
      part = (int)strtol(ptno.c_str(), (char**)pos, 16);
#else
    part = strtoi(ptno, pos, 16);
#endif

    return VENDOR_ARM;
}

}

#endif
