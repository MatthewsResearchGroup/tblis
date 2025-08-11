#include "tblis.h"
#include "tblis/frame/base/env.hpp"
#include "tblis/plugin/bli_plugin_tblis.h"

#if TBLIS_HAVE_SYSCTL
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if TBLIS_HAVE_SYSCONF
#include <unistd.h>
#endif

#if TBLIS_HAVE_HWLOC_H
#include <hwloc.h>
#endif

const tblis_comm* const tblis_single = tci_single;

namespace
{

struct thread_configuration
{
    unsigned num_threads = 1;

    thread_configuration()
    {
        const char* str = nullptr;

        str = getenv("TBLIS_NUM_THREADS");
        if (!str) str = getenv("BLIS_NUM_THREADS");
        if (!str) str = getenv("OMP_NUM_THREADS");

        if (str)
        {
            num_threads = strtol(str, NULL, 10);
            if(tblis::get_verbose() > 0)
                fprintf(stderr, "setting nt from environment variable: %d\n", num_threads);
        }
        else
        {
            #if TBLIS_HAVE_HWLOC_H

            hwloc_topology_t topo;
            hwloc_topology_init(&topo);
            hwloc_topology_load(topo);

            num_threads = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_CORE);
            if(tblis::get_verbose() > 0)
                fprintf(stderr, "setting nt with hwloc: %d\n", num_threads);

            hwloc_topology_destroy(topo);

            #elif TBLIS_HAVE_LSCPU

            FILE *fd = popen("lscpu --parse=core | grep '^[0-9]' | sort -rn | head -n 1", "r");

            std::string s;
            int c;
            while ((c = fgetc(fd)) != EOF) s.push_back(c);

            pclose(fd);

            num_threads = strtol(s.c_str(), NULL, 10) + 1;
            if(tblis::get_verbose() > 0)
                fprintf(stderr, "setting nt with lscpu: %d\n", num_threads);

            #elif TBLIS_HAVE_SYSCTLBYNAME

            size_t len = sizeof(num_threads);
            sysctlbyname("hw.physicalcpu", &num_threads, &len, NULL, 0);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_ONLN

            num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            if(tblis::get_verbose() > 0)
                fprintf(stderr, "setting nt with _SC_NPROCESSORS_ONLN: %d\n", num_threads);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

            num_threads = sysconf(_SC_NPROCESSORS_CONF);
            if(tblis::get_verbose() > 0)
                fprintf(stderr, "setting nt with _SC_NPROCESSORS_CONF: %d\n", num_threads);

            #endif
        }
    }
};

thread_configuration& get_thread_configuration()
{
    static thread_configuration cfg;
    return cfg;
}

}

namespace tblis
{

tci::communicator single;

std::atomic<long> flops{0};
len_type inout_ratio = 200000;
int outer_threading = 1;

void thread_blis(const communicator& comm,
                 const obj_t* a,
                 const obj_t* b,
                 const obj_t* c,
                 const cntx_t* cntx,
                 const cntl_t* cntl)
{
    rntm_t rntm = BLIS_RNTM_INITIALIZER;
    bli_rntm_init_from_global(&rntm);
    bli_rntm_set_num_threads(comm.num_threads(), &rntm);

    bli_rntm_factorize(bli_obj_length(c),
                       bli_obj_width(c),
                       bli_obj_width(a), &rntm);

    thrcomm_t* gl_comm = nullptr;
    // This can be NULL if SBA pools aren't used
    array_t* array = nullptr;

    if (comm.master())
    {
        gl_comm = bli_thrcomm_create(bli_thread_get_thread_impl(), nullptr, comm.num_threads());
    }

    comm.broadcast(
    [&](auto array, auto gl_comm)
    {
        thrinfo_t* thread = bli_l3_thrinfo_create(comm.thread_num(), gl_comm, array, &rntm, cntl);

        bli_l3_int
        (
          a,
          b,
          c,
          cntx,
          cntl,
          thread
        );

        bli_thrinfo_barrier(thread);
        bli_thrinfo_free(thread);
    }, array, gl_comm);

    if (comm.master())
    {
        bli_thrcomm_free(nullptr, gl_comm);
    }
}

}

TBLIS_EXPORT
unsigned tblis_get_num_threads()
{
    return get_thread_configuration().num_threads;
}

TBLIS_EXPORT
void tblis_set_num_threads(unsigned num_threads)
{
    get_thread_configuration().num_threads = num_threads;
}
