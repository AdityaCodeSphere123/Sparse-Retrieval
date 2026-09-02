#pragma once
// mmap for index/corpus files; JSONL parser for {"doc_id","text"}.

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

struct MmapBuffer {
    const uint8_t* data = nullptr;
    size_t length = 0;
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;

    bool map_file(const std::string& path) {
        unmap();
        // FILE_SHARE_DELETE matters: without it the mapping locks the file for
        // the life of the process, and a caller that deletes index_dir while the
        // engine is still loaded (the harness does exactly that, and so does
        // pytest's temporary-directory cleanup) fails with a sharing violation.
        // POSIX mmap has no such restriction, which is why the branch below
        // needs no equivalent.
        hFile = CreateFileA(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz) || sz.QuadPart == 0) { unmap(); return false; }
        length = static_cast<size_t>(sz.QuadPart);
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { unmap(); return false; }
        void* addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) { unmap(); return false; }
        data = static_cast<const uint8_t*>(addr);
        return true;
    }

    void unmap() {
        if (data) { UnmapViewOfFile(const_cast<uint8_t*>(data)); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = NULL; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
        length = 0;
    }
#else
    bool map_file(const std::string& path) {
        unmap();
        int fd = open(path.c_str(), O_RDONLY);
        if (fd == -1) return false;
        struct stat st;
        if (fstat(fd, &st) == -1 || st.st_size == 0) { close(fd); return false; }
        length = st.st_size;
        void* addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) { data = nullptr; length = 0; return false; }
        data = static_cast<const uint8_t*>(addr);
        // Query process: page postings in during load_index (not scored).
#ifdef MADV_WILLNEED
        (void)madvise(addr, length, MADV_WILLNEED);
#endif
        return true;
    }

    void unmap() {
        if (data) {
            munmap(const_cast<uint8_t*>(data), length);
            data = nullptr;
            length = 0;
        }
    }
#endif

    ~MmapBuffer() { unmap(); }
};


bool parse_jsonl_sv(std::string_view line, std::string& doc_id, std::string& text);
bool parse_jsonl_line(const std::string& line, std::string& doc_id, std::string& text);
void collect_jsonl_spans(const char* p, size_t n,
                         std::vector<std::pair<const char*, size_t>>& spans);
