#ifndef BITNET_LINUX_FF_H
#define BITNET_LINUX_FF_H

#include <stdint.h>
#include <stdio.h>

typedef unsigned int UINT;
typedef uint64_t FSIZE_t;
typedef int FRESULT;
typedef struct { int unused; } FATFS;
typedef struct {
    FILE *fp;
    FSIZE_t size;
} FIL;

#define FR_OK 0
#define FR_DISK_ERR 1
#define FR_NO_FILE 4
#define FR_EXIST 8
#define FA_READ 0x01u
#define FA_WRITE 0x02u
#define FA_CREATE_ALWAYS 0x08u
#define FM_FAT32 0x01u

FRESULT f_mount(FATFS *fs, const char *path, int mount_now);
FRESULT f_open(FIL *file, const char *path, unsigned mode);
FRESULT f_close(FIL *file);
FRESULT f_read(FIL *file, void *buffer, UINT bytes, UINT *read_bytes);
FRESULT f_write(FIL *file, const void *buffer, UINT bytes, UINT *written_bytes);
FRESULT f_lseek(FIL *file, FSIZE_t offset);
FRESULT f_mkdir(const char *path);
FRESULT f_mkfs(const char *path, unsigned option, unsigned allocation_unit,
               void *work, UINT work_bytes);

#define f_size(file) ((file)->size)

#endif
