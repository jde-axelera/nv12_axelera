#pragma once
#include <cstddef>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

struct DmaBuf {
    int    fd  = -1;
    void*  ptr = nullptr;
    size_t sz  = 0;

    static DmaBuf alloc(int heap_fd, size_t size) {
        DmaBuf b; b.sz = size;
        struct dma_heap_allocation_data d{};
        d.len = static_cast<__u64>(size);
        d.fd_flags = O_CLOEXEC | O_RDWR;
        if (::ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &d) < 0) return b;
        b.fd  = static_cast<int>(d.fd);
        b.ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, b.fd, 0);
        if (b.ptr == MAP_FAILED) { ::close(b.fd); b.fd = -1; b.ptr = nullptr; }
        return b;
    }

    void release() {
        if (ptr && ptr != MAP_FAILED) ::munmap(ptr, sz);
        if (fd >= 0) ::close(fd);
        fd = -1; ptr = nullptr;
    }

    bool valid() const { return fd >= 0 && ptr != nullptr; }
};
