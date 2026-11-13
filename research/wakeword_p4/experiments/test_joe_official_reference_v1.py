"""Test exact-development admission and tamper/final rejection, no detector calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('official_replay', ROOT/'qemu_esp32_replay/run_replay.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ReferencePolicy(unittest.TestCase):
    def setUp(self):
        path=ROOT/'results/4090-joe-development-replay-manifest-v1.json'
        self.manifest=json.loads(path.read_text())
        self.digest=hashlib.sha256(path.read_bytes()).hexdigest()

    def test_all_original_360_admitted(self):
        for row in self.manifest['records']:
            module.validate_development_paired(self.manifest,self.digest,row)

    def test_changed_bytes_do_not_fall_back_to_fake_annotation(self):
        self.manifest['voice_partition_sha256']='invented'
        self.manifest['endpoint_annotation_source_sha256']='invented'
        digest=hashlib.sha256(json.dumps(self.manifest).encode()).hexdigest()
        with self.assertRaises(ValueError):
            module.validate_development_paired(self.manifest,digest,self.manifest['records'][0])

    def test_wrong_endpoints_roles_source_and_rir_rejected(self):
        mutations={'voice':'mike','role':'independent-paired-final','split':'testing',
                   'generated_output_end_sample':1,'full_rir_end_sample':1,
                   'keyword_end':1,'source_samples':1,'samples':1,
                   'lead_samples':0,'tail_samples':0,'rir_tail_samples':0,'variant':9}
        for key,value in mutations.items():
            with self.subTest(key=key):
                m=copy.deepcopy(self.manifest);row=next(r for r in m['records'] if r['variant']==2)
                row[key]=value
                with self.assertRaises(ValueError):
                    module.validate_development_paired(m,self.digest,row)

    def test_incomplete_or_final_manifest_rejected(self):
        for key,value in [('records',self.manifest['records'][:-1]),
                          ('independent_final_audio_accessed',True),
                          ('paired_wakephrases',False),('synthesis_plan_sha256','bad'),
                          ('generation_reference_source_sha256','bad')]:
            with self.subTest(key=key):
                m=copy.deepcopy(self.manifest);m[key]=value
                with self.assertRaises(ValueError):
                    module.validate_development_paired(m,self.digest,m['records'][0])

    def test_historical_slr83_reference_remains_supported(self):
        m={'synthesis_protocol':'slr83-new-development-v1','paired_wakephrases':True,
           'independent_final_audio_accessed':False}
        row=dict(role='development-paired-validation',split='validation',keyword_end=100,samples=200)
        module.validate_development_paired(m,'88cfa446880e8a562927221d552a544bb72f204af071cd6867c7e1284db80415',row)
        with self.assertRaises(ValueError):module.validate_development_paired(m,'bad',row)


if __name__=='__main__':unittest.main()
