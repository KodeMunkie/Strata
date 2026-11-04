"""Opt-in Linux file-cache pin owned by the server, not its unloadable engine."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import stat
import sys

GIB = 1 << 30
LOCK_CHUNK = 256 << 20


def settings(cfg):
    values = []
    for key, default in (("model_file_cache_gib", 0), ("model_file_cache_shard", 0),
                         ("model_file_cache_headroom_gib", 6)):
        value = cfg.get(key, default)
        if type(value) is not int or not 0 <= value <= 65536:
            raise ValueError(f"{key}: expected a whole number in 0..65536")
        values.append(value)
    return tuple(values)


def selected_file(cfg, args, shard):
    """Select by shard number, without tensor offsets or model-specific names. Zero means last."""
    names = [v for k, v in zip(args, args[1:]) if k in ("--native", "--model")]
    if not names:
        raise ValueError("model file cache needs --native or --model naming a GGUF file")
    path = Path(names[0])
    if not path.is_absolute():
        path = Path(cfg.get("cwd") or ".") / path
    if path.suffix.lower() != ".gguf":
        raise ValueError("model file cache supports GGUF files only")
    match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path.name)
    count = int(match[3]) if match else 1
    number = shard or count
    if not 1 <= number <= count:
        raise ValueError(f"model_file_cache_shard: model has {count} shard(s), requested {number}")
    if match:
        path = path.with_name(f"{match[1]}-{number:05d}-of-{count:05d}.gguf")
    return path.resolve(strict=True)


def available_bytes():
    with open('/proc/meminfo') as stream:
        available = next(int(s.split()[1]) * 1024 for s in stream if s.startswith('MemAvailable:'))
    # A container's allowance can be smaller than the host's MemAvailable.
    try:
        root = Path('/sys/fs/cgroup')
        limit = (root / 'memory.max').read_text().strip()
        if limit != 'max':
            available = min(available, max(0, int(limit) - int((root / 'memory.current').read_text())))
    except (OSError, ValueError):
        pass
    return available


class ModelFileCache:
    def __init__(self):
        self.address = None
        self.length = 0
        self.path = None
        self._libc = None

    def start(self, cfg, args):
        budget, shard, headroom = settings(cfg)
        if not budget:
            return
        if not sys.platform.startswith('linux'):
            raise ValueError('model file cache is currently supported on Linux only')
        path = selected_file(cfg, args, shard)
        self.pin(path, min(budget * GIB, path.stat().st_size), headroom * GIB)

    def pin(self, path, length, headroom):
        import resource
        if self.address is not None:
            raise ValueError('model file cache is already pinned')
        if length <= 0:
            raise ValueError('model file cache cannot pin an empty file')
        page = os.sysconf('SC_PAGE_SIZE')
        locked = (length + page - 1) // page * page
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < locked:
            raise ValueError(f'model file cache needs LimitMEMLOCK >= {locked} bytes (or a larger ulimit -l); '
                             'no memory was pinned')
        if available_bytes() < locked + headroom:
            raise ValueError('not enough available RAM for the model file cache plus its headroom')
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_longlong]
        for name in ('mlock', 'munlock', 'munmap'):
            getattr(libc, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            getattr(libc, name).restype = ctypes.c_int
        fd = os.open(path, os.O_RDONLY)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size < length:
                raise ValueError('model file cache needs a regular file containing the requested range')
            address = libc.mmap(None, length, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED: same page cache as fread
        finally:
            os.close(fd)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'model file cache mmap failed')
        self.address, self.length, self.path, self._libc = address, length, str(path), libc
        try:
            step = LOCK_CHUNK
            for offset in range(0, length, step):
                count = min(step, length - offset)
                if available_bytes() < headroom + count:
                    raise ValueError('model file cache headroom exhausted; releasing its pin')
                if libc.mlock(address + offset, count):
                    raise OSError(ctypes.get_errno(), 'model file cache mlock failed; releasing its pin')
                done = offset + count
                if done % GIB == 0:
                    print(f'[strata] model file cache: pinned {done / GIB:g}/{length / GIB:.2f} GiB', flush=True)
        except BaseException:
            self.close()
            raise
        print(f'[strata] model file cache: {length / GIB:.2f} GiB pinned from {path}; '
              'kept across engine unloads until the server exits', flush=True)

    def close(self):
        if self.address is not None:
            self._libc.munlock(self.address, self.length)
            self._libc.munmap(self.address, self.length)
            self.address = None
            self.length = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
