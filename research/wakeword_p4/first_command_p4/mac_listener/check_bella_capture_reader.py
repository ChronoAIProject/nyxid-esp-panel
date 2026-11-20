"""Verify the handed-off PCM24 files against a known lag and a silent receiver channel."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import wave
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import measure_command_capture as mod

mod_path = Path(mod.__file__)
results = mod_path.parents[2] / "results"
pack = results / "c161-bella-daily-missed-playback.zip"

def main():
    with TemporaryDirectory() as tmp, ZipFile(pack) as z:
        t = Path(tmp)
        targets = json.loads(z.read('manifest.json'))['records']
        for case in targets:
            wav = t / 'play.wav'
            wav.write_bytes(z.read(case['file']))
            at = case['command_at']
            with wave.open(str(wav),'rb') as src:
                raw = src.readframes(src.getnframes())
            expected,rate=sf.read(wav,dtype='int32')
            values,actual,width,rate = mod.read_channel(wav,0)
            assert rate==16000 and width==3 and np.array_equal(values,expected/65536) and actual==raw
            # Independent shifted stereo receive at 48 kHz. Selected channel is 1 and channel 0 is quiet.
            data=np.pad(values/32768,(4000,2000))*0.5
            replay=resample_poly(data,3,1)
            receive=t/'receiving.wav'
            sf.write(receive,np.stack([np.zeros_like(replay),replay],axis=1),48000,subtype='PCM_24')
            for ch in ([1,0] if case==targets[0] else [1]):
                output=subprocess.check_output([sys.executable,str(mod_path),str(wav),str(receive),'--command-at',str(at),'--channel',str(ch)])
                result=json.loads(output)
                assert result['command_pcm_sha256']==sha256(raw[at*3:]).hexdigest()
                assert result['recording_rate']==48000 and result['analysis_rate']==16000
                if ch==1:
                    assert result['status']=='three_windows_matched',result
                    assert all(abs(a['lag_samples']-at-4000)<=4 for a in result['anchors'].values())
                else:
                    assert result['status']=='inconclusive_matches'
        signed=np.array([-8388608,-8388607,-257,-256,-1,0,1,255,256,8388607],dtype=np.int32)
        sf.write(wav,signed*256,16000,subtype='PCM_24')
        np.testing.assert_array_equal(mod.read_channel(wav,0)[0],signed/256)
        receive=t/'16.wav'
        with ZipFile(pack.with_name('c161-bella-command-playback.zip')) as clean:
            case=targets[0]
            receive.write_bytes(clean.read(case['source']))
        args=[str(receive),str(receive),'--command-at',str(case['command_at'])]
        current=json.loads(subprocess.check_output([sys.executable,str(mod_path),*args]))
        assert current['status']=='three_windows_matched'
        assert current['command_pcm_sha256']=='08ff3c8ef479245e7e8c992c033e55097355a442f3d42799b50bade30d0c6f88'
        assert all(anchor['lag_samples']==case['command_at'] for anchor in current['anchors'].values())
        assert all(-1<=anchor['correlation']<=1.00000001 for anchor in current['anchors'].values())

    print('9 PCM24/48 kHz delayed stereo cases, silent channel, signed extrema, and existing PCM16 CLI passed')


if __name__ == "__main__":
    main()
