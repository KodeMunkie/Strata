"""Opt-in Linux shard-cache pin owned by the server, not its unloadable engine."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import stat
import sys
import threading

GIB = 1 << 30
LOCK_CHUNK = 256 << 20
DEFAULT_SHARD = 3  # Qwen3.8 Flash Next UD-IQ4_XS: third shard, or last available for smaller layouts
MIN_HEADROOM_GIB = 8


def budget_arg(value):
    """CLI convention: a whole GiB budget or the word auto."""
    return 'auto' if value.strip().lower() == 'auto' else int(value)


def settings(cfg):
    for key in cfg:
        if key.startswith('model_file_cache_'):
            raise ValueError(f"{key} was renamed to {key.replace('model_file_cache_', 'shard_cache_', 1)}")
    if type(cfg.get('shard_cache_hot_unload', False)) is not bool:
        raise ValueError('shard_cache_hot_unload: expected true or false')
    values = []
    for key, default in (("shard_cache_gib", 0), ("shard_cache_shard", None),
                         ("shard_cache_headroom_gib", MIN_HEADROOM_GIB)):
        value = cfg.get(key, default)
        if key == 'shard_cache_gib' and value == 'auto':
            values.append(value)
            continue
        if key == 'shard_cache_shard' and key not in cfg:
            values.append(None)  # default resolved against this model's shard count, never a numeric sentinel
            continue
        minimum = 1 if key == 'shard_cache_shard' else 0
        if type(value) is not int or not minimum <= value <= 65536:
            suffix = ' or auto' if key == 'shard_cache_gib' else ''
            raise ValueError(f"{key}: expected a whole number in {minimum}..65536{suffix}")
        values.append(value)
    return tuple(values)


def selected_file(cfg, args, shard):
    """Select exactly one file by its 1-based shard number, without model-specific tensor offsets."""
    names = [v for k, v in zip(args, args[1:]) if k in ("--native", "--model")]
    if not names:
        raise ValueError("shard cache needs --native or --model naming a GGUF file")
    path = Path(names[0])
    if not path.is_absolute():
        path = Path(cfg.get("cwd") or ".") / path
    if path.suffix.lower() != ".gguf":
        raise ValueError("shard cache supports GGUF files only")
    match = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", path.name)
    count = int(match[3]) if match else 1
    number = min(DEFAULT_SHARD, count) if shard is None else shard
    if not 1 <= number <= count:
        raise ValueError(f"shard_cache_shard: model has {count} shard(s), requested {number}")
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


class ShardCache:
    def __init__(self):
        self.address = None
        self.length = 0
        self.path = None
        self._libc = None
        self._deferred = None
        self._lock = threading.RLock()

    def configure(self, cfg, args):
        """Fill at startup by default; optionally defer until a normal engine unload."""
        budget, shard, _ = settings(cfg)
        if not budget:
            return
        if not sys.platform.startswith('linux'):
            raise ValueError('shard cache is currently supported on Linux only')
        selected_file(cfg, args, shard)  # bad filenames/shard choices are configuration errors, not RAM failures
        if cfg.get('shard_cache_hot_unload', False):
            self._deferred = (dict(cfg), list(args))
            print('[strata] shard cache hot unload: fill deferred until model unload', flush=True)
        else:
            self._fill(cfg, args)

    def _fill(self, cfg, args):
        try:
            self.start(cfg, args)
        except (OSError, ValueError, MemoryError) as exc:
            self._release()
            print(f'[strata] WARNING: shard cache unavailable; continuing without a pin: {exc}. '
                  + self._suggestion(cfg, args), flush=True)

    @staticmethod
    def _suggestion(cfg, args):
        """Estimate a usable budget after failure; never retry or claim to reserve future arena RAM."""
        try:
            import resource
            _, shard, headroom = settings(cfg)
            reserve = max(MIN_HEADROOM_GIB, headroom)
            room = max(0, available_bytes() - reserve * GIB)
            soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            if soft != resource.RLIM_INFINITY:
                room = min(room, max(0, soft))
            size = selected_file(cfg, args, shard).stat().st_size
            page = os.sysconf('SC_PAGE_SIZE')
            locked = (size + page - 1) // page * page
            budget = (size + GIB - 1) // GIB if room >= locked else room // GIB
            if budget:
                return (f'Current RAM and memory-lock limits suggest --shard-cache-gib {budget}, '
                        f'leaving at least {reserve} GiB available to the OS. This is a fill-time estimate; '
                        'also allow RAM for the model arena during reload.')
            return (f'No whole-GiB cache budget fits the current RAM/memory-lock allowance while leaving '
                    f'{reserve} GiB to the OS; use --shard-cache-gib 0 to disable the cache, '
                    'free RAM or increase the memory-lock allowance.')
        except (OSError, ValueError, MemoryError):
            return ('Could not estimate a smaller budget; use --shard-cache-gib 0 to disable the cache. '
                    'Leave at least 8 GiB available to the OS, plus RAM for the arena during reload.')

    def on_unload(self):
        """Called after engine/vision RAM is released, under the service's request lock."""
        with self._lock:
            if self._deferred is not None and self.address is None:
                cfg, args = self._deferred
                self._fill(cfg, args)

    def on_load(self):
        """The engine has reached READY, so its arena no longer needs the hot-unload pin."""
        with self._lock:
            if self._deferred is not None and self.address is not None:
                self._release()
                print('[strata] shard cache hot unload: pin released after model load; RAM is reclaimable again', flush=True)

    def start(self, cfg, args):
        budget, shard, headroom = settings(cfg)
        if not budget:
            return
        if not sys.platform.startswith('linux'):
            raise ValueError('shard cache is currently supported on Linux only')
        path = selected_file(cfg, args, shard)
        size = path.stat().st_size
        length = size if budget == 'auto' else min(budget * GIB, size)
        self.pin(path, length, max(MIN_HEADROOM_GIB, headroom) * GIB)

    def pin(self, path, length, headroom):
        import resource
        if self.address is not None:
            raise ValueError('shard cache is already pinned')
        if length <= 0:
            raise ValueError('shard cache cannot pin an empty file')
        page = os.sysconf('SC_PAGE_SIZE')
        locked = (length + page - 1) // page * page
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and soft < locked:
            raise ValueError(f'shard cache needs LimitMEMLOCK >= {locked} bytes (or a larger ulimit -l); '
                             'no memory was pinned')
        if available_bytes() < locked + headroom:
            raise ValueError('not enough available RAM for the shard cache plus its headroom')
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
                raise ValueError('shard cache needs a regular file containing the requested range')
            address = libc.mmap(None, length, 1, 1, fd, 0)  # PROT_READ, MAP_SHARED: same page cache as fread
        finally:
            os.close(fd)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'shard cache mmap failed')
        self.address, self.length, self.path, self._libc = address, length, str(path), libc
        try:
            step = LOCK_CHUNK
            for offset in range(0, length, step):
                count = min(step, length - offset)
                if available_bytes() < headroom + count:
                    raise ValueError('shard cache headroom exhausted; releasing its pin')
                if libc.mlock(address + offset, count):
                    raise OSError(ctypes.get_errno(), 'shard cache mlock failed; releasing its pin')
                done = offset + count
                if done % GIB == 0:
                    print(f'[strata] shard cache: pinned {done / GIB:g}/{length / GIB:.2f} GiB', flush=True)
        except BaseException:
            self._release()
            raise
        lifetime = 'until the next model load completes' if self._deferred else 'until the server exits'
        print(f'[strata] shard cache: {length / GIB:.2f} GiB pinned from {path}; kept {lifetime}', flush=True)

    def _release(self):
        if self.address is not None:
            self._libc.munlock(self.address, self.length)
            self._libc.munmap(self.address, self.length)
            self.address = None
            self.length = 0

    def close(self):
        with self._lock:
            self._deferred = None
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
