"""File selection, real small Linux locks, and server ownership without loading a model."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from serve import model_file_cache as cache
from serve import runconfig


class Configuration(unittest.TestCase):
    def test_default_off_performs_no_io(self):
        with cache.ModelFileCache() as pin, mock.patch.object(cache, 'selected_file') as select:
            pin.start({}, [])
            pin.start({'model_file_cache_gib': 0}, [])
            select.assert_not_called()
            self.assertIsNone(pin.address)

    def test_settings_roundtrip_and_rejection(self):
        cfg, _ = runconfig.apply({}, {'model_file_cache_gib': 30, 'model_file_cache_shard': 3})
        self.assertEqual(cache.settings(cfg), (30, 3, 6))
        cfg, _ = runconfig.apply(cfg, {'model_file_cache_gib': None})
        self.assertEqual(cache.settings(cfg)[0], 0)
        for value in (-1, True, 1.5, '30', float('nan'), float('inf'), 65537):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runconfig.apply({}, {'model_file_cache_gib': value})

    def test_shards_follow_model_and_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('one-00001-of-00003.gguf', 'one-00003-of-00003.gguf', 'single.gguf'):
                (root / name).write_bytes(b'model')
            cfg = {'cwd': tmp}
            args = ['--native', 'one-00001-of-00003.gguf']
            self.assertEqual(cache.selected_file(cfg, args, 0), root / 'one-00003-of-00003.gguf')
            self.assertEqual(cache.selected_file(cfg, args, 1), root / 'one-00001-of-00003.gguf')
            with self.assertRaises(ValueError):
                cache.selected_file(cfg, args, 4)
            with self.assertRaises(FileNotFoundError):
                cache.selected_file(cfg, args, 2)
            self.assertEqual(cache.selected_file(cfg, ['--model', 'single.gguf'], 0), root / 'single.gguf')

    def test_non_linux_refuses_enabled_feature(self):
        with mock.patch.object(cache.sys, 'platform', 'win32'), self.assertRaisesRegex(ValueError, 'Linux'):
            cache.ModelFileCache().start({'model_file_cache_gib': 1}, [])


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux mlock')
class LinuxPin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'weights.gguf'
        self.path.write_bytes(b'x' * 65536)

    def test_real_prefix_stays_locked_then_releases(self):
        import resource
        if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] < 8192:
            self.skipTest('needs 8 KiB memlock allowance')
        def locked():
            return next(int(s.split()[1]) for s in Path('/proc/self/status').read_text().splitlines()
                        if s.startswith('VmLck:'))
        before = locked()
        with cache.ModelFileCache() as pin:
            pin.pin(self.path, 8192, 0)
            self.assertEqual(locked(), before + 8)
            fd = os.open(self.path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            libc = pin._libc
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            vec = (ctypes.c_ubyte * 2)()
            self.assertEqual(libc.mincore(pin.address, 8192, vec), 0)
            self.assertEqual([v & 1 for v in vec], [1, 1])
        self.assertEqual(locked(), before)
        pin.close()  # repeated shutdown is safe

    def test_refuses_limit_and_headroom_before_mapping(self):
        with mock.patch('resource.getrlimit', return_value=(0, 0)), self.assertRaisesRegex(ValueError, 'LimitMEMLOCK'):
            cache.ModelFileCache().pin(self.path, 8192, 0)
        with mock.patch.object(cache, 'available_bytes', return_value=0), self.assertRaisesRegex(ValueError, 'RAM'):
            cache.ModelFileCache().pin(self.path, 8192, 0)

    def test_mlock_failure_unmaps_and_clears_state(self):
        libc = ctypes.CDLL(None, use_errno=True)
        real_lock = libc.mlock
        real_lock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        calls = 0
        def fail_second(address, count):
            nonlocal calls
            calls += 1
            return real_lock(address, count) if calls == 1 else -1
        libc.mlock = mock.Mock(side_effect=fail_second)
        pin = cache.ModelFileCache()
        with mock.patch.object(cache, 'LOCK_CHUNK', 4096), \
                mock.patch.object(cache.ctypes, 'CDLL', return_value=libc), self.assertRaises(OSError):
            pin.pin(self.path, 8192, 0)
        self.assertEqual(calls, 2)
        self.assertIsNone(pin.address)
        self.assertEqual(pin.length, 0)

    def test_server_owns_pin_until_exit_including_errors(self):
        from serve import server
        pin = cache.ModelFileCache()
        def run(*args, file_cache, **kwargs):
            file_cache.pin(self.path, 8192, 0)
            engine = mock.Mock()
            engine.unload()
            self.assertIsNotNone(file_cache.address)  # engine unload does not own or close the mapping
            raise RuntimeError('startup or server failure')
        with mock.patch.object(server.model_file_cache, 'ModelFileCache', return_value=pin), \
                mock.patch.object(server, '_main', side_effect=run), self.assertRaises(RuntimeError):
            server.main([])
        self.assertIsNone(pin.address)


if __name__ == '__main__':
    unittest.main()
