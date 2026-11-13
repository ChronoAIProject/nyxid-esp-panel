"""Constructed metadata tests, not TFLite inference or candidate admission."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import prepare_complete_runtime as runtime


class CompletePreparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='complete-runtime-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'out'
        self.model = b'\0\0\0\0TFL3'+bytes(1016)  # Deliberately not a real model.
        self.model_sha = runtime.digest(self.model)
        reviewed = json.loads((Path(runtime.__file__).parent/'plans/gpu-v30-runtime-v1.json').read_text())
        self.training = dict(architecture='teacher60', architecture_args=reviewed['architecture_args'],
            context_frames=225, parameters=24301, steps=120000, selected_step=120000,
            checkpoint_policy='fixed-final', classifier_reset=False,
            batchnorm_initial={'snapshot': 'constructed'}, batchnorm_selected={'snapshot': 'constructed'},
            model_sha256=self.model_sha, model_bytes=len(self.model),
            batch_schedule=dict(train_batches=120000, ptq_batches=100, verified_batches=120100,
                all_actual_batch_tensors_match_precomputed=True))
        self.completion = dict(target='hey_nyx', steps=120000, model_sha256=self.model_sha,
            finite_weights=True, finite_logged_loss=True, frozen_bn_exact=True)
        self.model_id = 'constructed_nyx'
        identity = dict(sha256=self.model_sha, bytes=len(self.model), target_id='hey_nyx')
        self.execution = dict(schema='complete-development-stream-execution-v1', tail_policy=runtime.TAIL,
            authorizes='development-raw-collection-only', models={self.model_id: identity},
            materials={phase: {'role': 'new-development'} for phase in ('synthetic', 'background')})
        selected = dict(threshold=.98828125, eligible=True, registered_grid=True)
        self.calibration = dict(schema='complete-development-calibration-v1', model_id=self.model_id,
            target_id='hey_nyx', model=identity.copy(), materials=copy.deepcopy(self.execution['materials']), candidate_promoted=False,
            acceptance_authorized=False, threshold=selected['threshold'], selected=selected,
            grid=[copy.deepcopy(selected)])
        self.profile = dict(schema=runtime.SCHEMA, accepted=False, model_id=self.model_id,
            target_id='hey_nyx', input_frames=225, event_arithmetic=runtime.ARITHMETIC,
            tail_policy=runtime.TAIL, smoothing_steps=5, refractory_steps=67,
            threshold=selected['threshold'])
        self.fixture = self.make_fixture(50000)

    def make_fixture(self, samples, *, magic=b'NYXKWS02', outputs=None, uniform=False):
        count = (max(0, (samples+159)//160-2)+2)//3 if outputs is None else outputs
        raw = bytes(count) if uniform else bytes(i%256 for i in range(count))
        return magic+bytes.fromhex(self.model_sha)+struct.pack('<II', samples, count)+bytes(samples*2)+raw

    def bind(self):
        # Rebind hashes after semantic mutations so rejection exercises the
        # semantic check, rather than stopping at the first stale digest.
        data = {'model': self.model, 'selftest': self.fixture}
        data['training'] = json.dumps(self.training).encode()
        self.completion['training_report_sha256'] = runtime.digest(data['training'])
        data['completion'] = json.dumps(self.completion).encode()
        data['execution'] = json.dumps(self.execution).encode()
        self.calibration['execution_sha256'] = runtime.digest(data['execution'])
        data['calibration'] = json.dumps(self.calibration).encode()
        self.profile['artifacts'] = {}
        for name, value in data.items():
            path = self.root/(name+'.bin' if name in ('model', 'selftest') else name+'.json')
            path.write_bytes(value)
            self.profile['artifacts'][name] = dict(path=path.name, sha256=runtime.digest(value))
        path = self.root/'profile.json'
        path.write_text(json.dumps(self.profile))
        return path

    def prepare(self, bind=True, **kwargs):
        path = self.bind() if bind else self.root/'profile.json'
        return runtime.prepare(path, kwargs.get('model', self.root/'model.bin'),
            self.root/'calibration.json', self.root/'selftest.bin', self.output)

    def rejected(self):
        with self.assertRaises((ValueError, KeyError, TypeError)):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_complete_nonmultiple_fixture_exact_copy_precision_and_idempotence(self):
        self.output.mkdir()
        result = self.prepare()
        self.assertEqual(result['selftest_outputs'], 104)
        self.assertEqual((result['final_wave_padding'], result['final_feature_padding']), (80, 1))
        self.assertFalse(result['accepted'])
        self.assertEqual((self.output/'hey_nyx_model.bin').read_bytes(), self.model)
        self.assertEqual((self.output/'wake_selftest.bin').read_bytes(), self.fixture)
        config = (self.output/'wake_config.h').read_text()
        self.assertIn('#define PANEL_WAKE_THRESHOLD 0.98828125\n', config)
        self.assertIn('#define PANEL_WAKE_PRIME_SAMPLES 68000\n', config)
        self.assertIn('#define PANEL_WAKE_COMPLETE_INPUT 1\n', config)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        self.assertEqual(self.prepare(), result)
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_every_artifact_digest_is_bound_before_output(self):
        self.bind()
        for name, item in self.profile['artifacts'].items():
            with self.subTest(artifact=name):
                path = self.root/item['path']
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaisesRegex(ValueError, 'Changed profile artifact'):
                    self.prepare(bind=False)
                path.write_bytes(original)
                self.assertFalse(self.output.exists())

    def test_legacy_or_unknown_policy_and_context_are_rejected(self):
        original = copy.deepcopy(self.profile)
        for key, value in [('schema', 'legacy'), ('input_frames', 150), ('accepted', True),
                ('target_id', 'hey_willow'), ('event_arithmetic', 'float32'),
                ('tail_policy', 'truncate'), ('smoothing_steps', 7), ('refractory_steps', 68)]:
            with self.subTest(key=key):
                self.profile = {**original, key: value}
                self.rejected()

    def test_training_finality_topology_bn_and_actual_batches(self):
        original = copy.deepcopy(self.training)
        for key, value in [('context_frames', 150), ('parameters', 1), ('steps', 119999),
                ('selected_step', 70000), ('checkpoint_policy', 'best'), ('classifier_reset', True),
                ('batchnorm_selected', {}), ('architecture_args', []), ('model_sha256', '0'*64)]:
            with self.subTest(key=key):
                self.training = {**copy.deepcopy(original), key: value}
                self.rejected()
        for key, value in [('train_batches', 119999), ('ptq_batches', 99), ('verified_batches', 120099),
                ('all_actual_batch_tensors_match_precomputed', False)]:
            with self.subTest(batch=key):
                self.training = copy.deepcopy(original)
                self.training['batch_schedule'][key] = value
                self.rejected()

    def test_completion_identity_and_finite_guards(self):
        original = copy.deepcopy(self.completion)
        for key, value in [('target', 'hey_willow'), ('steps', 119999), ('model_sha256', '0'*64),
                ('finite_weights', False), ('finite_logged_loss', False), ('frozen_bn_exact', False)]:
            with self.subTest(key=key):
                self.completion = {**original, key: value}
                self.rejected()

    def test_null_ineligible_unregistered_or_foreign_selected_point(self):
        original = copy.deepcopy(self.calibration)
        for selected in [None, {}, {**original['selected'], 'eligible': False},
                {**original['selected'], 'registered_grid': False},
                {**original['selected'], 'threshold': .5}]:
            with self.subTest(selected=selected):
                self.calibration = {**copy.deepcopy(original), 'selected': selected}
                self.rejected()

    def test_execution_roles_and_model_identity(self):
        original = copy.deepcopy(self.execution)
        for key, value in [('schema', 'legacy'), ('tail_policy', 'truncate'), ('authorizes', 'acceptance')]:
            with self.subTest(key=key):
                self.execution = {**copy.deepcopy(original), key: value}
                self.rejected()
        for phase in ('synthetic', 'background'):
            self.execution = copy.deepcopy(original)
            self.execution['materials'][phase]['role'] = 'old-regression'
            self.rejected()
        self.execution = copy.deepcopy(original)
        self.execution['models'][self.model_id]['sha256'] = '0'*64
        self.rejected()

    def test_calibration_identity_schema_and_authority(self):
        original = copy.deepcopy(self.calibration)
        for key, value in [('schema', 'joint-development-v1'), ('model_id', 'foreign'),
                ('target_id', 'hey_willow'), ('materials', {}), ('candidate_promoted', True), ('acceptance_authorized', True)]:
            with self.subTest(key=key):
                self.calibration = {**copy.deepcopy(original), key: value}
                self.rejected()

    def test_threshold_precision_and_validity(self):
        original = self.profile['threshold']
        for value in [True, '0.99', 0, 1, -.1, float('nan'), float('inf'), .5]:
            with self.subTest(threshold=value):
                self.profile['threshold'] = value
                self.rejected()
        self.profile['threshold'] = original

    def test_v1_truncation_identity_count_length_uniform_short_are_rejected(self):
        original = self.fixture
        fixtures = [original[:47], original[:-1], original+b'\0',
            self.make_fixture(50000, magic=b'NYXKWS01'),
            original[:8]+bytes(32)+original[40:], self.make_fixture(50000, outputs=103),
            self.make_fixture(50000, uniform=True), self.make_fixture(960), self.make_fixture(256001)]
        for i, fixture in enumerate(fixtures):
            with self.subTest(fixture=i):
                self.fixture = fixture
                self.rejected()

    def test_build_arguments_and_partial_or_changed_output_are_not_overwritten(self):
        alternate = self.root/'different.bin'
        alternate.write_bytes(b'different')
        self.bind()
        with self.assertRaisesRegex(ValueError, 'Build argument'):
            self.prepare(bind=False, model=alternate)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        keep = self.output/'partial'
        keep.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(keep.read_bytes(), b'preserve')
        keep.unlink()
        self.prepare()
        (self.output/'wake_config.h').write_bytes(b'preserve changed output')
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.output/'wake_config.h').read_bytes(), b'preserve changed output')

    def cmake_project(self):
        source = self.root/'cmake-project'
        source.mkdir()
        component = Path(runtime.__file__).resolve().parent
        (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\n'
            'project(complete_profile_configure LANGUAGES CXX)\n'
            'macro(idf_component_register)\n'
            f'  add_library(runtime STATIC "{component}/panel_wake_runtime.cpp")\n'
            '  set(COMPONENT_LIB runtime)\nendmacro()\n'
            'function(target_add_binary_data target path kind)\n'
            '  if(NOT EXISTS "${path}")\n    message(FATAL_ERROR "Missing embedding")\n  endif()\n'
            '  file(APPEND "${CMAKE_BINARY_DIR}/embedded.txt" "${path}\\n")\nendfunction()\n'
            f'add_subdirectory("{component}" runtime)\n'
            f'get_property(deps DIRECTORY "{component}" PROPERTY CMAKE_CONFIGURE_DEPENDS)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/dependencies.txt" "${deps}")\n'
            'get_target_property(definitions runtime COMPILE_DEFINITIONS)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/definitions.txt" "${definitions}")\n')
        return source

    def test_actual_cmake_complete_selector_arguments_and_dependencies(self):
        profile = self.bind()
        source = self.cmake_project()
        build = self.root/'cmake-build'
        command = ['cmake', '-S', str(source), '-B', str(build), '-DPYTHON='+sys.executable,
            '-DNYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE='+str(profile),
            '-DNYX_PANEL_EXPERIMENTAL_WAKE_MODEL='+str(self.root/'model.bin'),
            '-DNYX_PANEL_EXPERIMENTAL_WAKE_CALIBRATION='+str(self.root/'calibration.json'),
            '-DNYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST='+str(self.root/'selftest.bin')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        manifest = json.loads((build/'runtime/generated/manifest.json').read_text())
        self.assertEqual(manifest['schema'], 'complete-development-p4-prepared-v1')
        self.assertIn('PANEL_WAKE_SELFTEST=1', (build/'definitions.txt').read_text())
        embedded = (build/'embedded.txt').read_text()
        self.assertIn('hey_nyx_model.bin', embedded)
        self.assertIn('wake_selftest.bin', embedded)
        dependencies = (build/'dependencies.txt').read_text()
        for item in self.profile['artifacts'].values():
            self.assertIn(str(self.root/item['path']), dependencies)
        self.assertIn(str(profile), dependencies)
        # Reconfigure the identical build without overwriting different output.
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_actual_cmake_conflicting_profiles_and_missing_selftest_are_rejected(self):
        profile = self.bind()
        source = self.cmake_project()
        for index, extra in enumerate([[],
                ['-DNYX_PANEL_EXPERIMENTAL_WAKE_SELFTEST=fixture', '-DNYX_PANEL_EXPERIMENTAL_WAKE_CANDIDATE_PROFILE=legacy'],
                ['-DNYX_PANEL_EXPERIMENTAL_EMBEDDING_BUNDLE=bundle']]):
            with self.subTest(case=index):
                result = subprocess.run(['cmake', '-S', str(source), '-B', str(self.root/('cmake-reject-'+str(index))),
                    '-DPYTHON='+sys.executable, '-DNYX_PANEL_EXPERIMENTAL_WAKE_COMPLETE_PROFILE='+str(profile),
                    *extra], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue('Complete profile requires' in result.stderr or
                    'cannot configure an embedding bundle' in result.stderr, result.stderr)


if __name__ == '__main__':
    unittest.main()
