"""Measure background transcript coverage of AMI mixture foreground spans.

This is annotated word-interval coverage, not VAD, phonetic truth or wake score.
The insertion start is known to10ms in the feature index; report that limit.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('features', 'source-audit', 'annotations', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    audit = json.loads(a.source_audit.read_text())
    augmentation = json.loads((a.features/'dialogue-augmentation.json').read_text())
    if (augmentation['dialogue_source_kind'] != 'ami'
            or augmentation['source_dialogue_audit_sha256'] != sha(a.source_audit)
            or augmentation['combined_index_sha256'] != sha(a.features/'features.json')
            or audit['reserved_audio_accessed'] or audit['classifiers_scored']):
        raise ValueError('Invalid training mixture source')
    expected = [r for r in audit['metadata_sources'] if r['file'] == 'ami_public_manual_1.6.2.zip']
    if len(expected) != 1 or sha(a.annotations) != expected[0]['sha256']:
        raise ValueError('Changed pinned annotations')
    archive = zipfile.ZipFile(a.annotations)
    intervals = defaultdict(list)
    source_rows = {r['session']: r for r in audit['records']}
    parsed = []
    for row in audit['training_transcripts']:
        session = Path(row['path']).name.split('.')[0]
        if session not in source_rows:
            raise ValueError('Nonselected transcript')
        data = archive.read(row['path'])
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Changed training word annotations')
        parsed.append(row['path'])
        for word in ET.fromstring(data):
            if word.tag != 'w' or word.attrib.get('punc') == 'true':
                continue
            start = int(round(float(word.attrib['starttime'])*16000))
            end = int(round(float(word.attrib['endtime'])*16000))
            if not 0 <= start <= end <= source_rows[session]['samples']:
                raise ValueError('Word interval outside original recording')
            if start < end:
                intervals[session].append((start, end))
    unions = {}
    for session, items in intervals.items():
        merged = []
        for start, end in sorted(items):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        starts, ends = np.asarray(merged, dtype=np.int64).T
        sums = np.concatenate(([0], np.cumsum(ends-starts)))
        unions[session] = (starts, ends, sums)

    def integral(session, t):
        starts, ends, sums = unions[session]
        i = int(np.searchsorted(ends, t, side='right'))
        return int(sums[i]) + (max(0, int(t-starts[i])) if i < len(starts) else 0)

    groups = defaultdict(list)
    trace = hashlib.sha256()
    count = 0
    index = json.loads((a.features/'features.json').read_text())
    for row in index[augmentation['base_sequences']:]:
        if row['split'] != 'training' or row['background_session'] not in source_rows:
            raise ValueError('Nontraining mixture')
        session = row['background_session']
        start = row['background_start_sample'] + row['lead_frame']*160
        end = start + row['samples']
        overlap = integral(session, end)-integral(session, start)
        fraction = overlap/row['samples']
        item = dict(path=row['path'], variant=row['variant'], session=session,
                    foreground_span_start_approx_samples=start, foreground_samples=row['samples'],
                    annotated_word_overlap_samples=overlap)
        trace.update((json.dumps(item, sort_keys=True)+'\n').encode())
        for key in ('all', 'session:'+session, f"engine_label:{row['engine']}:{row['label']}"):
            groups[key].append((fraction, overlap, row['samples']))
        count += 1
    if count != augmentation['added_training_sequences']:
        raise ValueError('Incomplete augmentation coverage')
    summaries = {}
    for key, values in groups.items():
        fractions = np.asarray([v[0] for v in values])
        summaries[key] = dict(mixtures=len(values), mean_fraction=float(fractions.mean()),
            median_fraction=float(np.median(fractions)), p10_fraction=float(np.quantile(fractions, .1)),
            no_annotated_word_overlap=sum(v[1] == 0 for v in values),
            at_least_half_span_word_coverage=int(np.count_nonzero(fractions >= .5)))
    result = dict(role='Training background word-annotation overlap diagnostic, not speech/VAD or classifier evidence',
        source_sha256=sha(Path(__file__)), source_audit_sha256=sha(a.source_audit),
        augmentation_report_sha256=sha(a.features/'dialogue-augmentation.json'),
        annotations_sha256=sha(a.annotations), parsed_training_transcripts=parsed,
        per_mixture_trace_sha256=trace.hexdigest(), summaries=summaries,
        lead_quantization_samples=160, maximum_overlap_count_error_from_lead_rounding_samples=320,
        wake_scores_accessed=False, independent_final_audio_accessed=False,
        labels_or_features_changed=False,
        limits='Official training word intervals across all meeting speakers are unioned. Foreground support is the full inserted TTS clip, including its silence; not a foreground active-speech interval.10ms insertion-time rounding and word annotation/acoustic timing errors remain. Zero annotated overlap is not proof of silence; nonzero is not proof of simultaneous audible speech. No wake scores, model thresholds, clip filtering, new labels or independent exposure claim.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries['all']))


if __name__ == '__main__':
    main()
