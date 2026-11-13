"""Real saved development witness plus missing-return negative control, no inference."""
import json,re,shutil,tempfile,zlib
from pathlib import Path
import soundfile as sf
from calibrate_development import audit_stream
from run_development_grid import PLAN,sha


def main():
    p=json.loads(PLAN.read_text());row=p['records'][0];root=Path(p['output']);stem='willow-00-000'
    values,rate=sf.read(row['path'],dtype='int16');assert rate==16000
    crc=zlib.crc32(values.astype('<i2').tobytes());runtime={k:sha(v) for k,v in p['runtime'].items()}
    audit_stream(root,stem,row,'willow','DEFAULT',runtime,crc)
    with tempfile.TemporaryDirectory(prefix='hey-missing-return-') as name:
        dest=Path(name)
        for suffix in ('report.json','serial.log','stage-monitor.txt','raw-returns.bin'):shutil.copyfile(root/(stem+'-'+suffix),dest/(stem+'-'+suffix))
        uart=dest/(stem+'-serial.log');before=uart.read_bytes()
        changed,n=re.subn(rb'HEY_REFERENCE_RETURN call=\d+ sample_end=\d+ result=-?\d+\r?\n',b'',before,count=1);assert n==1
        uart.write_bytes(changed)
        # Re-sign this copied UART's metadata so rejection must pass its simple hash check.
        report=dest/(stem+'-report.json');r=json.loads(report.read_text());r['serial_sha256']=sha(uart);report.write_text(json.dumps(r)+'\n')
        rejected=False
        try:audit_stream(dest,stem,row,'willow','DEFAULT',runtime,crc)
        except AssertionError:rejected=True
        assert rejected,'Missing raw return was accepted'
    result=dict(schema='hey-official-snapshot-corruption-check-v1',checker_sha256=sha(__file__),execution_sha256=sha(PLAN),original_report_sha256=sha(root/(stem+'-report.json')),original_uart_sha256=sha(root/(stem+'-serial.log')),original_witness_passed=True,removed_one_input_return=True,copied_claimed_uart_hash_updated=True,missing_return_rejected=True,original_evidence_modified=False,inference_run=False,limits='Single consumed saved-witness negative control for complete-return admission, not recognition/FAR or complete official grid result.')
    out=Path('research/wakeword_p4/results/4090-hey-official-snapshot-corruption-check-v1.json')
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
