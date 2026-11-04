"""File selection, real small Linux locks, and server ownership without loading a model."""
import ctypes
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

from serve import shard_cache as cache
from serve import runconfig


class Configuration(unittest.TestCase):
    def test_old_config_keys_and_unprefixed_options_are_not_accepted(self):
        with self.assertRaisesRegex(ValueError, 'shard_cache_gib'):
            cache.settings({'model_file_cache_gib': 32})
        from serve import server
        for option in ('--hot-unload', '--model-file-cache-gib'):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as result:
                server.main([option])
            self.assertEqual(result.exception.code, 2)

    def test_default_off_performs_no_io(self):
        with cache.ShardCache() as pin, mock.patch.object(cache, 'selected_file') as select:
            pin.start({}, [])
            pin.start({'shard_cache_gib': 0}, [])
            select.assert_not_called()
            self.assertIsNone(pin.address)

    def test_settings_roundtrip_and_rejection(self):
        cfg, _ = runconfig.apply({}, {'shard_cache_gib': 30, 'shard_cache_shard': 3})
        self.assertEqual(cache.settings(cfg), (30, 3, 8))
        cfg, _ = runconfig.apply(cfg, {'shard_cache_gib': None})
        self.assertEqual(cache.settings(cfg)[0], 0)
        for value in (-1, True, 1.5, '30', float('nan'), float('inf'), 65537):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runconfig.apply({}, {'shard_cache_gib': value})

    def test_auto_budget_roundtrips_and_remains_opt_in(self):
        cfg, changed = runconfig.apply({}, {'shard_cache_gib': 'auto'})
        self.assertEqual(cache.settings(cfg), ('auto', None, 8))
        self.assertEqual(changed, ['shard_cache_gib'])
        self.assertEqual(cache.budget_arg('auto'), 'auto')
        self.assertEqual(cache.budget_arg('32'), 32)
        for value in ('all', 'auto:32', 'AUTO'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runconfig.apply({}, {'shard_cache_gib': value})

    def test_shards_follow_model_and_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('one-00001-of-00003.gguf', 'one-00003-of-00003.gguf', 'single.gguf',
                         'two-00002-of-00002.gguf'):
                (root / name).write_bytes(b'model')
            cfg = {'cwd': tmp}
            args = ['--native', 'one-00001-of-00003.gguf']
            self.assertEqual(cache.selected_file(cfg, args, None), root / 'one-00003-of-00003.gguf')
            self.assertEqual(cache.selected_file(cfg, args, 3), root / 'one-00003-of-00003.gguf')
            self.assertEqual(cache.selected_file(cfg, args, 1), root / 'one-00001-of-00003.gguf')
            with self.assertRaises(ValueError):
                cache.selected_file(cfg, args, 4)
            with self.assertRaises(ValueError):
                cache.selected_file(cfg, args, 0)
            with self.assertRaises(ValueError):
                runconfig.apply({}, {'shard_cache_shard': 0})
            with self.assertRaises(FileNotFoundError):
                cache.selected_file(cfg, args, 2)
            self.assertEqual(cache.selected_file(cfg, ['--model', 'single.gguf'], None), root / 'single.gguf')
            self.assertEqual(cache.selected_file(cfg, ['--model', 'two-00001-of-00002.gguf'], None),
                             root / 'two-00002-of-00002.gguf')

    def test_non_linux_refuses_enabled_feature(self):
        with mock.patch.object(cache.sys, 'platform', 'win32'), self.assertRaisesRegex(ValueError, 'Linux'):
            cache.ShardCache().start({'shard_cache_gib': 1}, [])

    def test_hot_unload_is_boolean_and_defaults_off(self):
        for value in ('true', 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'true or false'):
                cache.settings({'shard_cache_hot_unload': value})
        cfg, changed = runconfig.apply({}, {'shard_cache_hot_unload': True})
        self.assertIs(cfg['shard_cache_hot_unload'], True)
        self.assertEqual(changed, ['shard_cache_hot_unload'])
        cfg, _ = runconfig.apply(cfg, {'shard_cache_hot_unload': None})
        self.assertNotIn('shard_cache_hot_unload', cfg)
        with cache.ShardCache() as pin, mock.patch.object(pin, 'start') as start, \
                mock.patch.object(cache, 'selected_file'):
            pin.configure({'shard_cache_gib': 1}, ['--native', 'unused.gguf'])
            start.assert_called_once()

    def test_disabled_cache_does_not_defer_or_touch_files(self):
        with cache.ShardCache() as pin, mock.patch.object(cache, 'selected_file') as select:
            pin.configure({'shard_cache_hot_unload': True}, [])
            pin.on_unload()
            select.assert_not_called()

    def test_web_controls_and_platform_validation(self):
        fields = {k['key']: k for k in runconfig.view({}, 'model.json')['keys']}
        self.assertEqual(fields['shard_cache_hot_unload']['kind'], 'bool')
        self.assertTrue(fields['shard_cache_hot_unload']['linux_only'])
        self.assertEqual(fields['shard_cache_gib']['step'], 1)
        self.assertEqual(fields['shard_cache_gib']['kind'], 'cache_budget')
        with mock.patch.object(runconfig.sys, 'platform', 'darwin'):
            self.assertEqual(runconfig.view({}, 'model.json')['platform'], 'darwin')
            with self.assertRaisesRegex(ValueError, 'Linux'):
                runconfig.apply({}, {'shard_cache_gib': 32})


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux mlock')
class LinuxPin(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'weights.gguf'
        self.path.write_bytes(b'x' * 65536)
        # Real locks remain tiny; lifecycle tests must not depend on the machine's live RAM workload.
        available = mock.patch.object(cache, 'available_bytes', return_value=128 * cache.GIB)
        available.start()
        self.addCleanup(available.stop)

    def test_auto_pins_only_selected_file_and_enforces_eight_gib_headroom(self):
        with cache.ShardCache() as pin, mock.patch.object(pin, 'pin') as lock:
            pin.start({'shard_cache_gib': 'auto', 'shard_cache_headroom_gib': 0},
                      ['--native', str(self.path)])
            lock.assert_called_once_with(self.path, self.path.stat().st_size, 8 * cache.GIB)
        with cache.ShardCache() as pin, mock.patch.object(pin, 'pin') as lock:
            pin.start({'shard_cache_gib': 'auto', 'shard_cache_headroom_gib': 12},
                      ['--native', str(self.path)])
            lock.assert_called_once_with(self.path, self.path.stat().st_size, 12 * cache.GIB)

    def test_auto_failure_warns_with_smaller_budget_in_both_modes(self):
        import resource
        cfg = {'shard_cache_gib': 'auto'}
        for hot in (False, True):
            with self.subTest(hot=hot), cache.ShardCache() as pin:
                output = io.StringIO()
                with mock.patch.object(cache, 'available_bytes', return_value=40 * cache.GIB), \
                        mock.patch('resource.getrlimit', return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)), \
                        mock.patch.object(Path, 'stat', return_value=mock.Mock(st_size=48 * cache.GIB)), \
                        mock.patch.object(pin, 'pin', side_effect=MemoryError('allocation refused')), \
                        contextlib.redirect_stdout(output):
                    pin.configure({**cfg, 'shard_cache_hot_unload': hot}, ['--native', str(self.path)])
                    if hot:
                        pin.on_unload()
                self.assertIn('WARNING', output.getvalue())
                self.assertIn('--shard-cache-gib 32', output.getvalue())
                self.assertIn('at least 8 GiB', output.getvalue())
                self.assertIsNone(pin.address)

    def test_suggestion_respects_lock_limit_and_larger_headroom(self):
        args = ['--native', str(self.path)]
        with mock.patch.object(cache, 'available_bytes', return_value=40 * cache.GIB), \
                mock.patch('resource.getrlimit', return_value=(20 * cache.GIB, 20 * cache.GIB)), \
                mock.patch.object(Path, 'stat', return_value=mock.Mock(st_size=48 * cache.GIB)):
            suggestion = cache.ShardCache._suggestion({'shard_cache_gib': 'auto',
                                                           'shard_cache_headroom_gib': 24}, args)
            self.assertIn('--shard-cache-gib 16', suggestion)
            self.assertIn('24 GiB', suggestion)
        with mock.patch.object(cache, 'available_bytes', return_value=8 * cache.GIB):
            suggestion = cache.ShardCache._suggestion({'shard_cache_gib': 'auto'}, args)
            self.assertIn('--shard-cache-gib 0', suggestion)

    def test_auto_real_pin_is_released_after_hot_reload(self):
        svc = self.service()
        with cache.ShardCache() as pin:
            svc.shard_cache = pin
            pin.configure({'shard_cache_gib': 'auto', 'shard_cache_hot_unload': True},
                          ['--native', str(self.path)])
            svc.load()
            self.assertIsNone(pin.address)
            svc.unload()
            self.assertEqual(pin.length, self.path.stat().st_size)
            svc.load()
            self.assertIsNone(pin.address)

    def service(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, Service
        from serve.test_lifecycle import ResidentEngine
        tok = ByteTokenizer()
        return Service(ResidentEngine(tok), tok, ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))

    def test_hot_unload_fills_after_release_then_unlocks_after_arena_ready(self):
        svc = self.service()
        cfg = {'shard_cache_gib': 1, 'shard_cache_headroom_gib': 0,
               'shard_cache_hot_unload': True}
        with cache.ShardCache() as pin:
            svc.shard_cache = pin
            pin.configure(cfg, ['--native', str(self.path)])
            self.assertIsNone(pin.address)
            self.assertEqual(svc.unload(), 'not loaded')
            self.assertIsNone(pin.address)
            svc.engine.restart()
            svc.vision = mock.Mock()
            svc.vision.unload.side_effect = lambda: self.assertFalse(svc.engine.alive())
            vision = svc.vision
            real_start = pin.start
            def start(*args):
                self.assertFalse(svc.engine.alive())
                self.assertGreaterEqual(vision.unload.call_count, 1)
                self.assertFalse(svc.fifo.acquire(blocking=False))
                real_start(*args)
            with mock.patch.object(pin, 'start', side_effect=start) as fill:
                svc.status['busy'] = 1
                self.assertEqual(svc.unload(), 'busy')
                fill.assert_not_called()
                svc.status['busy'] = 0
                self.assertEqual(svc.unload(), 'unloaded')
                address = pin.address
                self.assertIsNotNone(address)
                svc.vision = None
                restart = svc.engine.restart
                def load_arena():
                    self.assertEqual(pin.address, address)  # still pinned while the arena reads it
                    restart()
                with mock.patch.object(svc.engine, 'restart', side_effect=load_arena):
                    svc.load()
                self.assertIsNone(pin.address)
                self.assertEqual(pin.length, 0)
                self.assertEqual(svc.unload(idle_for=0), 'unloaded')
                self.assertIsNotNone(pin.address)
                self.assertEqual(fill.call_count, 2)
        self.assertIsNone(pin.address)

    def test_startup_and_unload_allocation_failure_warn_and_continue(self):
        for hot in (False, True):
            with self.subTest(hot=hot), cache.ShardCache() as pin:
                svc = self.service()
                svc.shard_cache = pin
                svc.engine.restart()
                cfg = {'shard_cache_gib': 1, 'shard_cache_hot_unload': hot}
                output = io.StringIO()
                with mock.patch.object(cache, 'available_bytes', return_value=0), \
                        contextlib.redirect_stdout(output):
                    pin.configure(cfg, ['--native', str(self.path)])
                    self.assertEqual(svc.unload(), 'unloaded')
                self.assertIn('WARNING: shard cache unavailable', output.getvalue())
                self.assertIsNone(pin.address)
                self.assertEqual(pin.length, 0)
                svc.load()  # warning did not break the model lifecycle

    def test_failed_arena_load_keeps_pin_for_retry(self):
        svc = self.service()
        with cache.ShardCache() as pin:
            svc.shard_cache = pin
            pin.configure({'shard_cache_gib': 1, 'shard_cache_headroom_gib': 0,
                           'shard_cache_hot_unload': True}, ['--native', str(self.path)])
            svc.engine.restart()
            svc.unload()
            address = pin.address
            with mock.patch.object(svc.engine, 'restart', side_effect=RuntimeError('load failed')), \
                    self.assertRaises(RuntimeError):
                svc.load()
            self.assertEqual(pin.address, address)
            svc.load()
            self.assertIsNone(pin.address)

    def test_default_startup_pin_is_kept_after_load_and_unload(self):
        svc = self.service()
        with cache.ShardCache() as pin:
            svc.shard_cache = pin
            pin.configure({'shard_cache_gib': 1, 'shard_cache_headroom_gib': 0},
                          ['--native', str(self.path)])
            address = pin.address
            self.assertIsNotNone(address)
            svc.load()
            self.assertEqual(pin.address, address)
            svc.unload()
            self.assertEqual(pin.address, address)

    def test_hot_unload_failure_releases_partial_pin_and_can_retry(self):
        cfg = {'shard_cache_gib': 1, 'shard_cache_headroom_gib': 0,
               'shard_cache_hot_unload': True}
        with cache.ShardCache() as pin:
            pin.configure(cfg, ['--native', str(self.path)])
            with mock.patch.object(cache, 'available_bytes', return_value=0):
                pin.on_unload()
            self.assertIsNone(pin.address)
            self.assertEqual(pin.length, 0)
            pin.on_unload()
            self.assertIsNotNone(pin.address)

    def test_server_exit_cancels_deferred_fill(self):
        pin = cache.ShardCache()
        with pin, mock.patch.object(pin, 'start') as start:
            pin.configure({'shard_cache_gib': 1, 'shard_cache_hot_unload': True},
                          ['--native', str(self.path)])
        pin.on_unload()  # an idle worker after shutdown must not create a new pin
        start.assert_not_called()
        self.assertIsNone(pin.address)

    def test_shutdown_waits_for_an_in_progress_fill_then_unlocks(self):
        pin = cache.ShardCache()
        pin.configure({'shard_cache_gib': 1, 'shard_cache_headroom_gib': 0,
                       'shard_cache_hot_unload': True}, ['--native', str(self.path)])
        filling, release, closed = threading.Event(), threading.Event(), threading.Event()
        real_start = pin.start
        def start(*args):
            filling.set()
            if not release.wait(5):
                raise ValueError('test fill timed out')
            real_start(*args)
        def close():
            pin.close()
            closed.set()
        with mock.patch.object(pin, 'start', side_effect=start):
            worker = threading.Thread(target=pin.on_unload)
            closer = threading.Thread(target=close)
            worker.start()
            try:
                self.assertTrue(filling.wait(2))
                closer.start()
                self.assertFalse(closed.wait(0.05))
            finally:
                release.set()
                worker.join(5)
                if closer.ident is not None:
                    closer.join(5)
                pin.close()
            self.assertTrue(closed.is_set())
            self.assertIsNone(pin.address)

    def test_real_prefix_stays_locked_then_releases(self):
        import resource
        if resource.getrlimit(resource.RLIMIT_MEMLOCK)[0] < 8192:
            self.skipTest('needs 8 KiB memlock allowance')
        def locked():
            return next(int(s.split()[1]) for s in Path('/proc/self/status').read_text().splitlines()
                        if s.startswith('VmLck:'))
        before = locked()
        with cache.ShardCache() as pin:
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
            cache.ShardCache().pin(self.path, 8192, 0)
        with mock.patch.object(cache, 'available_bytes', return_value=0), self.assertRaisesRegex(ValueError, 'RAM'):
            cache.ShardCache().pin(self.path, 8192, 0)

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
        pin = cache.ShardCache()
        with mock.patch.object(cache, 'LOCK_CHUNK', 4096), \
                mock.patch.object(cache.ctypes, 'CDLL', return_value=libc), self.assertRaises(OSError):
            pin.pin(self.path, 8192, 0)
        self.assertEqual(calls, 2)
        self.assertIsNone(pin.address)
        self.assertEqual(pin.length, 0)

    def test_server_owns_pin_until_exit_including_errors(self):
        from serve import server
        pin = cache.ShardCache()
        def run(*args, cache_pin, **kwargs):
            cache_pin.pin(self.path, 8192, 0)
            engine = mock.Mock()
            engine.unload()
            self.assertIsNotNone(cache_pin.address)  # engine unload does not own or close the mapping
            raise RuntimeError('startup or server failure')
        with mock.patch.object(server.shard_cache, 'ShardCache', return_value=pin), \
                mock.patch.object(server, '_main', side_effect=run), self.assertRaises(RuntimeError):
            server.main([])
        self.assertIsNone(pin.address)


if __name__ == '__main__':
    unittest.main()
