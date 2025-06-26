#include "communicator.h"
#include "task_set.h"

#ifdef __cplusplus
#include <cstdlib>

extern "C" {
#else
#include <stdlib.h>
#endif

#if TCI_USE_OPENMP_THREADS || TCI_USE_PTHREADS_THREADS || TCI_USE_WINDOWS_THREADS

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask,
                       uint64_t work)
{
    set->comm = comm;
    set->ntask = ntask;

    if (tci_comm_is_master(comm))
    {
        set->slots = (tci_slot*)malloc((ntask+1)*sizeof(tci_slot));
        for (unsigned task = 0;task < ntask;task++)
            tci_slot_init(set->slots+task+1, 0);
    }
    tci_comm_bcast(comm, (void**)&set->slots, 0);

    unsigned nt = comm->nthread;
    unsigned nt_outer, nt_inner;
    tci_partition_2x2(nt, work, (work == 0 ? 1 : nt),
                      ntask, ntask, &nt_inner, &nt_outer);
    tci_comm_gang(comm, &set->subcomm, TCI_EVENLY, nt_outer, 0);
}

void tci_task_set_destroy(tci_task_set* set)
{
    tci_comm_barrier(set->comm);
    tci_comm_destroy(&set->subcomm);
    if (tci_comm_is_master(set->comm))
        free((void*)set->slots);
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task+1, 0, set->subcomm.gid+1))
        return EALREADY;

    func(&set->subcomm, task, payload);

    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
    {
        int ret = tci_task_set_visit(set, func, task, payload);
        if (ret != 0) return ret;
    }

    return 0;
}

#elif TCI_USE_TBB_THREADS

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    (void)comm;
    (void)work;

    set->comm = (tci_comm*)new tbb::task_group();
    set->ntask = ntask;
    set->slots = new tci_slot[ntask];
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);
}

void tci_task_set_destroy(tci_task_set* set)
{
    ((tbb::task_group*)set->comm)->wait();
    delete[] set->slots;
    delete (tbb::task_group*)set->comm;
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    ((tbb::task_group*)set->comm)->run(
    [&,func,task,payload]
    {
        func(tci_single, task, payload);
    });

    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
    {
        int ret = tci_task_set_visit(set, func, task, payload);
        if (ret != 0) return ret;
    }

    return 0;
}

#elif TCI_USE_OMPTASK_THREADS

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    (void)comm;
    (void)work;

    set->ntask = ntask;
    set->slots = (tci_slot*)malloc(sizeof(tci_slot)*ntask);
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);
}

void tci_task_set_destroy(tci_task_set* set)
{
    #pragma omp taskwait
    free((void*)set->slots);
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    #pragma omp task
    {
        func(tci_single, task, payload);
    }

    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
    {
        int ret = tci_task_set_visit(set, func, task, payload);
        if (ret != 0) return ret;
    }

    return 0;
}

#elif TCI_USE_DISPATCH_THREADS

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    (void)comm;
    (void)work;

    *(dispatch_group_t*)&set->comm = dispatch_group_create();
    *(dispatch_queue_t*)&set->subcomm =
        dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    set->ntask = ntask;
    set->slots = (tci_slot*)malloc(sizeof(tci_slot)*ntask);
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);
}

void tci_task_set_destroy(tci_task_set* set)
{
    dispatch_group_t group = *(dispatch_group_t*)&set->comm;
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
    free((void*)set->slots);
}

typedef struct tci_task_func_data
{
    tci_task_func func;
    unsigned task;
    void* payload;
} tci_task_func_data;

static void tci_task_launcher(void* data_)
{
    tci_task_func_data* data = (tci_task_func_data*)data_;
    data->func(tci_single, data->task, data->payload);
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    tci_task_func_data data = {func, task, payload};
    dispatch_group_t group = *(dispatch_group_t*)&set->comm;
    dispatch_queue_t queue = *(dispatch_queue_t*)&set->subcomm;

    dispatch_group_async_f(group, queue, &data, tci_task_launcher);

    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
    {
        int ret = tci_task_set_visit(set, func, task, payload);
        if (ret != 0) return ret;
    }

    return 0;
}

#elif TCI_USE_PPL_THREADS

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask, uint64_t work)
{
    (void)comm;
    (void)work;

    set->comm = (tci_comm*)new concurrency::task_group();
    set->ntask = ntask;
    set->slots = new tci_slot[ntask];
    for (unsigned task = 0;task < ntask;task++)
        tci_slot_init(set->slots+task, 0);
}

void tci_task_set_destroy(tci_task_set* set)
{
    ((concurrency::task_group*)set->comm)->wait();
    delete[] set->slots;
    delete (concurrency::task_group*)set->comm;
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    if (!tci_slot_try_fill(set->slots+task, 0, 1)) return EALREADY;

    ((concurrency::task_group*)set->comm)->run(
    [&,func,task,payload]
    {
        func(tci_single, task, payload);
    });

    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
    {
        int ret = tci_task_set_visit(set, func, task, payload);
        if (ret != 0) return ret;
    }

    return 0;
}

#else // single threaded

void tci_task_set_init(tci_task_set* set, tci_comm* comm, unsigned ntask,
                       uint64_t work)
{
    (void)comm;
    (void)work;
    set->ntask = ntask;
}

void tci_task_set_destroy(tci_task_set* set)
{
    (void)set;
}

int tci_task_set_visit(tci_task_set* set, tci_task_func func, unsigned task,
                       void* payload)
{
    if (task > set->ntask) return EINVAL;
    func(tci_single, task, payload);
    return 0;
}

int tci_task_set_visit_all(tci_task_set* set, tci_task_func func,
                           void* payload)
{
    for (unsigned task = 0;task < set->ntask;task++)
        func(tci_single, task, payload);
    return 0;
}

#endif

#ifdef __cplusplus
}
#endif
