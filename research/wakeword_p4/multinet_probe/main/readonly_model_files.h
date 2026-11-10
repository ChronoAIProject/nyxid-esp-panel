// Standalone probe adapter for MultiNet's mandatory filesystem reader.
// All bytes remain in RAM/embedded app data. No filesystem partition is used.
#pragma once
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include "esp_vfs.h"

extern "C" void set_model_base_path(const char *base_path); // Pinned model_path.c.
extern const uint8_t default_commands[] asm("_binary_commands_en_txt_start");
extern const uint8_t default_commands_end[] asm("_binary_commands_en_txt_end");

namespace model_files {
struct File { char path[96]; const uint8_t *data; size_t size; };
struct Handle { const File *file; size_t at; };
static File files[7]{};
static Handle handles[8]{};
static unsigned count = 0;

static const File *find(const char *path) {
    for (unsigned i = 0; i < count; ++i) if (!strcmp(files[i].path, path)) return &files[i];
    return nullptr;
}
static int open(const char *path, int flags, int) {
    printf("MN_VFS open %s flags=%d\n", path, flags);
    if ((flags & O_ACCMODE) != O_RDONLY || (flags & (O_CREAT | O_TRUNC))) { errno = EROFS; return -1; }
    auto *file = find(path);
    if (!file) { errno = ENOENT; return -1; }
    for (unsigned i = 0; i < 8; ++i) if (!handles[i].file) { handles[i] = {file, 0}; return i; }
    errno = EMFILE;
    return -1;
}
static Handle *handle(int fd) {
    if (fd < 0 || fd >= 8 || !handles[fd].file) { errno = EBADF; return nullptr; }
    return &handles[fd];
}
static ssize_t read(int fd, void *output, size_t wanted) {
    auto *h = handle(fd);
    if (!h) return -1;
    const size_t n = std::min(wanted, h->file->size - h->at);
    memcpy(output, h->file->data + h->at, n);
    h->at += n;
    return n;
}
static int close(int fd) {
    auto *h = handle(fd);
    if (!h) return -1;
    *h = {};
    return 0;
}
static off_t seek(int fd, off_t offset, int whence) {
    auto *h = handle(fd);
    if (!h) return -1;
    const int64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? int64_t(h->at) : whence == SEEK_END ? int64_t(h->file->size) : -1;
    const int64_t next = base + offset;
    if (base < 0 || next < 0 || next > int64_t(h->file->size)) { errno = EINVAL; return -1; }
    h->at = size_t(next);
    return off_t(next);
}
static int fill_stat(const File *file, struct stat *st) {
    if (!file) { errno = ENOENT; return -1; }
    memset(st, 0, sizeof(*st));
    st->st_mode = S_IFREG | 0444;
    st->st_size = file->size;
    return 0;
}
static int fstat(int fd, struct stat *st) {
    auto *h = handle(fd);
    return h ? fill_stat(h->file, st) : -1;
}
static int stat(const char *path, struct stat *st) { return fill_stat(find(path), st); }

static bool install(srmodel_list_t *models) {
    auto *model = models->model_data[0];
    if (model->num != 4) return false;
    for (int i = 0; i < model->num; ++i) {
        auto &f = files[count++];
        snprintf(f.path, sizeof(f.path), "/mn7_en/%s", model->files[i]);
        f.data = reinterpret_cast<const uint8_t *>(model->data[i]);
        f.size = model->sizes[i];
    }
    for (const char *path : {"/mn7_en/commands_en.txt", "/fst/commands_en.txt", "/commands_en.txt"}) {
        auto &f = files[count++];
        snprintf(f.path, sizeof(f.path), "%s", path);
        f.data = default_commands;
        f.size = default_commands_end - default_commands;
    }
    esp_vfs_t vfs{};
    vfs.flags = ESP_VFS_FLAG_DEFAULT | ESP_VFS_FLAG_READONLY_FS;
    vfs.open = open;
    vfs.read = read;
    vfs.close = close;
    vfs.lseek = seek;
    vfs.fstat = fstat;
    vfs.stat = stat;
    if (esp_vfs_register("/mnprobe", &vfs, nullptr) != ESP_OK) return false;
    set_model_base_path("/mnprobe");
    return true;
}
} // namespace model_files
