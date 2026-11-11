"""Audit retained numeric P4 witnesses against immutable Mac/Linux evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'study', 'mac-native', 'mac-tables', 'linux-receipt', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--numeric-log', type=Path,
                   help='Use the public filtered markers instead of a private complete UART')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output file')
    root = args.artifacts
    board = load(root/'board/report.json')
    assert board['restored']['nyx_state'] == 'up' and board['restored']['wake_listening']
    manifest = load(root/'vectors-manifest.json')
    build = load(root/'build-receipt-private.json')
    audit = load(root/'fixture-parse-audit.json')
    assert board['vectors'] == manifest
    assert board['probe_bin_sha256'] == build['app_sha256']
    assert build['vectors_embedded_exactly_once'] and build['app_bytes'] <= build['slot_bytes']
    assert audit['all_headers_lengths_alignments_and_payload_hashes_match'] and audit['entire_blob_consumed']
    assert audit['vectors_sha256'] == manifest['vectors_sha256']
    for name, digest in manifest['pinned_inputs'].items():
        assert sha((args.study/name).read_bytes()) == digest
    records = load(args.study/'records.json')
    mac = load(args.mac_native)
    mac_by_index = {row['pack_index']: row for row in mac['details']}
    linux = load(args.linux_receipt)
    assert linux['all_complete_array_predictions_match']
    linux_by_index = {row['pack_index']: row for row in linux['details']}
    # Independently parse only the numeric markers from the complete private UART.
    parsed = defaultdict(list)
    marker_lines = []
    pattern = re.compile(r'^(P4_FRONTEND_TABLE|P4_FRONTEND_WITNESS|P4_FRONTEND_DELTA|P4_FRONTEND_DONE) (\{.*\})$')
    for line in (args.numeric_log or root/'board/uart-private.log').read_text().splitlines():
        match = pattern.fullmatch(line)
        if match:
            name, value = match.groups()
            parsed[name].append(json.loads(value)); marker_lines.append(line)
    assert parsed['P4_FRONTEND_TABLE'] == [board['P4_FRONTEND_TABLE']]
    assert parsed['P4_FRONTEND_DONE'] == [board['P4_FRONTEND_DONE']]
    assert parsed['P4_FRONTEND_WITNESS'] == board['frontend_witnesses']
    assert parsed['P4_FRONTEND_DELTA'] == board['frontend_deltas']
    assert len(parsed['P4_FRONTEND_WITNESS']) == len(manifest['witnesses']) == 30
    assert board['P4_FRONTEND_DONE'] == {'streams': 30, 'all_inputs_consumed': True}
    changes = defaultdict(list)
    for delta in board['frontend_deltas']:
        changes[delta['pack_index']].append(delta)
    details = []
    for actual, expected in zip(board['frontend_witnesses'], manifest['witnesses'], strict=True):
        index = actual['pack_index']; record = records[index]
        assert (index, actual['frames']) == (expected['pack_index'], expected['frames'])
        assert expected['mac_raw_sha256'] == record['raw_sha256']
        raw = np.load(args.study/record['raw'], allow_pickle=False)
        assert raw.dtype == np.uint16 and list(raw.shape) == [expected['frames'], 40]
        assert sha(raw.astype('<u2').tobytes()) == expected['mac_raw_sha256']
        differences = changes[index]
        # This dataset must retain every observed delta to reconstruct full bytes.
        assert len(differences) == actual['different_elements'] <= 8
        reconstructed = raw.copy()
        seen = set()
        for d in differences:
            key = (d['frame'], d['channel'])
            assert key not in seen and int(raw[key]) == d['expected']
            seen.add(key); reconstructed[key] = d['actual']
        assert max([abs(d['actual']-d['expected']) for d in differences], default=0) == actual['max_abs_delta']
        assert sha(reconstructed.astype('<u2').tobytes()) == actual['raw_sha256']
        e = record['feature_crop_end']
        five_equal = all(np.array_equal(raw[e+d-150:e+d], reconstructed[e+d-150:e+d])
                         for d in (0, 3, 6, 9, 12))
        details.append({**actual, 'mac_raw_sha256': expected['mac_raw_sha256'],
            'linux_raw_sha256': linux_by_index[index]['actual_full_linux_raw_sha256'],
            'mac_exact': actual['raw_sha256'] == expected['mac_raw_sha256'],
            'linux_exact': actual['raw_sha256'] == linux_by_index[index]['actual_full_linux_raw_sha256'],
            'mac_no_contract_exact': actual['raw_sha256'] == mac_by_index[index]['off_vs_saved']['actual_sha256'],
            'all_five_training_windows_equal_mac': five_equal, 'feature_crop_end': e,
            'all_differences_from_mac': differences})
    tables = board['P4_FRONTEND_TABLE']
    host_tables = load(args.mac_tables)['variants']
    table_matches = {}
    for mode, value in host_tables.items():
        table_matches[mode] = {key: tables[key+'_sha256'] == sha(np.asarray(value['tables'][field], dtype='<i2').tobytes())
                              for key, field in [('window', 'window_coefficients'), ('weights', 'weights'), ('unweights', 'unweights')]}
    result = {
        'scope': 'Standalone P4 numerical frontend diagnosis on30 fixed training witnesses; no model or recognition acceptance.',
        'auditor_sha256': sha(Path(__file__).read_bytes()),
        'protocol_commit': '2f7fcf8', 'vectors_manifest_sha256': sha((root/'vectors-manifest.json').read_bytes()),
        'vectors_sha256': manifest['vectors_sha256'], 'vectors_bytes': manifest['vectors_bytes'],
        'build': build, 'component': load(root/'compile-receipt-private.json')['dependency'],
        'mac_native_report_sha256': sha(args.mac_native.read_bytes()),
        'mac_table_report_sha256': sha(args.mac_tables.read_bytes()),
        'linux_receipt_sha256': sha(args.linux_receipt.read_bytes()),
        'fixture_parse_audit': audit, 'table': tables, 'table_matches': table_matches,
        'numeric_markers_sha256': sha(('\n'.join(marker_lines)+'\n').encode()),
        'witnesses': len(details), 'mac_exact_arrays': sum(r['mac_exact'] for r in details),
        'linux_exact_arrays': sum(r['linux_exact'] for r in details),
        'mac_no_contract_exact_arrays': sum(r['mac_no_contract_exact'] for r in details),
        'total_different_elements_from_mac': sum(r['different_elements'] for r in details),
        'maximum_raw_delta_from_mac': max(r['max_abs_delta'] for r in details),
        'all_five_training_windows_equal_mac': all(r['all_five_training_windows_equal_mac'] for r in details),
        'all_complete_arrays_reconstructed_and_hash_verified': True, 'details': details,
        'restored': board['restored'], 'audio_io': False, 'models_scored': False,
        'accepted_as_hi_esp_parity': False,
        'limits': ['Only these30 training inputs, fresh frontend per complete stream; not a universal numerical-equivalence claim.',
                   'P4 no-contract feature-output agreement does not mean its entire coefficient tables match that native build.',
                   'No acoustic/human recall, false-wake exposure, latency, CPU/heap or long-call/reconnect test in this probe.',
                   'v40 fixed data/protocol unchanged; no P4 coefficient overrides or production-board operation.']}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    (root/'numeric-markers.jsonl').write_text('\n'.join(marker_lines)+'\n')
    print(json.dumps({key: result[key] for key in ('witnesses', 'mac_exact_arrays', 'linux_exact_arrays',
        'mac_no_contract_exact_arrays', 'total_different_elements_from_mac', 'maximum_raw_delta_from_mac',
        'all_five_training_windows_equal_mac', 'accepted_as_hi_esp_parity')}))


if __name__ == '__main__':
    main()
