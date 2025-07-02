/*
 * Copyright 2017 Devin Matthews and Intel Corp.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdint>
#include <string>

#include <cpuid.h>

static std::string get_cpu_name()
{
    char cpu_name[48] = {};
    uint32_t eax, ebx, ecx, edx;

    __cpuid(0x80000002u, eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[0]  = eax;
    *(uint32_t *)&cpu_name[4]  = ebx;
    *(uint32_t *)&cpu_name[8]  = ecx;
    *(uint32_t *)&cpu_name[12] = edx;

    __cpuid(0x80000003u, eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[16+0]  = eax;
    *(uint32_t *)&cpu_name[16+4]  = ebx;
    *(uint32_t *)&cpu_name[16+8]  = ecx;
    *(uint32_t *)&cpu_name[16+12] = edx;

    __cpuid(0x80000004u, eax, ebx, ecx, edx);

    *(uint32_t *)&cpu_name[32+0]  = eax;
    *(uint32_t *)&cpu_name[32+4]  = ebx;
    *(uint32_t *)&cpu_name[32+8]  = ecx;
    *(uint32_t *)&cpu_name[32+12] = edx;

    return std::string(cpu_name);
}

int vpu_count()
{
    std::string name = get_cpu_name();

    if (name.find("Intel(R) Xeon(R)") != std::string::npos)
    {
        auto loc = name.find("Platinum");
        if (loc == std::string::npos) loc = name.find("Gold");
        if (loc == std::string::npos) loc = name.find("Silver");
        if (loc == std::string::npos) loc = name.find("Bronze");
        if (loc == std::string::npos) loc = name.find("W");
        if (loc == std::string::npos) return -1;
        loc = name.find_first_of("- ", loc+1)+1;

        auto sku = atoi(name.substr(loc, 4).c_str());
        if      (8399 >= sku && sku >= 8300) return 2; // Gen 3 (Cooper lake & Ice lake).
        else if (6399 >= sku && sku >= 6300) return 2;
        else if (5399 >= sku && sku >= 5300) return 2;
        else if (4399 >= sku && sku >= 4300) return 2;
        else if (9299 >= sku && sku >= 9200) return 2; // Gen 2.
        else if (8299 >= sku && sku >= 8200) return 2;
        else if (6299 >= sku && sku >= 6200) return 2;
        else if (sku == 5222)                return 2;
        else if (5299 >= sku && sku >= 5200) return 1;
        else if (4299 >= sku && sku >= 4200) return 1;
        else if (3219 >= sku && sku >= 3200) return 1;
        else if (3299 >= sku && sku >= 3220) return 2; // Gen 2 W.
        else if (2299 >= sku && sku >= 2220) return 2;
        else if (8199 >= sku && sku >= 8100) return 2; // Gen 1.
        else if (6199 >= sku && sku >= 6100) return 2;
        else if (sku == 5122)                return 2;
        else if (5199 >= sku && sku >= 5100) return 1;
        else if (4199 >= sku && sku >= 4100) return 1;
        else if (3199 >= sku && sku >= 3100) return 1;
        else if (2199 >= sku && sku >= 2120) return 2; // Gen 1 W.
        else if (2119 >= sku && sku >= 2100) return 1;
        else return -1;
    }
    else if (name.find("Intel(R) Core(TM) i9") != std::string::npos)
    {
        return 2;
    }
    else if (name.find("Intel(R) Core(TM) i7") != std::string::npos)
    {
        if (name.find("7800X") != std::string::npos ||
            name.find("7820X") != std::string::npos) return 2;
        else return -1;
    }
    else
    {
        return -1;
    }
}
