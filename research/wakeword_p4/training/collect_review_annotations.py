"""Validate blind-review exports and describe agreement without creating gold labels."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pack(html_path, audit_path):
    audit = json.loads(audit_path.read_text())
    if sha(html_path) != audit['html_sha256']:
        raise ValueError('Review HTML differs from frozen input audit')
    html = html_path.read_text()
    marker = 'const DATA='
    if html.count(marker) != 1:
        raise ValueError('Ambiguous review payload')
    pack, _ = json.JSONDecoder().raw_decode(html.split(marker,1)[1])
    payload = {k:v for k,v in pack.items() if k!='pack_id'}
    digest = hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
    if (digest != pack['pack_id'] or digest != audit['pack_id']
            or len(pack['records']) != audit['clips']
            or len({r['id'] for r in pack['records']}) != audit['clips']
            or {r['id'] for r in pack['records']} != {r['id'] for r in audit['mapping']}):
        raise ValueError('Changed or incomplete frozen review population')
    return pack


def validate_export(export, pack):
    if (export.get('protocol') != pack['protocol'] or export.get('pack_id') != pack['pack_id']
            or export.get('parent_manifest_sha256') != pack['parent_manifest_sha256']
            or export.get('detector_outputs_shown') is not False
            or export.get('automatic_boundaries_shown') is not False):
        raise ValueError('Wrong pack, source or blinding declaration')
    who = export.get('annotator')
    if (not isinstance(who,str) or not who.strip()
            or any(token in who.upper() for token in ('AUTOMATED_UI_TEST','NOT_HUMAN_LABELS'))):
        raise ValueError('Missing annotator or known automated-test export')
    expected = {r['id']:r for r in pack['records']}
    rows = export.get('records')
    if (not isinstance(rows,list) or len(rows)!=len(expected)
            or any(not isinstance(r,dict) or not isinstance(r.get('id'),str) for r in rows)
            or len({r['id'] for r in rows}) != len(rows)
            or {r['id'] for r in rows} != set(expected)):
        raise ValueError('Missing, repeated or extra review records')
    validated = {}
    for row in rows:
        original = expected[row['id']]
        if any(row.get(k) != original[k] for k in ('wav_sha256','pcm_sha256','samples')):
            raise ValueError('Review waveform identity changed')
        status, notes = row.get('status',''), row.get('notes','')
        lower, upper = row.get('lower_sample'), row.get('upper_sample')
        if status not in ('','matches','mismatch','uncertain') or not isinstance(notes,str):
            raise ValueError('Invalid status or notes type')
        for value in (lower,upper):
            if value is not None and (type(value) is not int or not 0<=value<=original['samples']):
                raise ValueError('Invalid sample boundary type/range')
        if lower is not None and upper is not None and lower>upper:
            raise ValueError('Reversed sample interval')
        bounds = lower is not None and upper is not None
        absent = lower is None and upper is None
        complete = (bounds if status=='matches' else bool(notes.strip()) and (absent or bounds)
                    if status in ('mismatch','uncertain') else False)
        validated[row['id']] = dict(status=status,lower_sample=lower,upper_sample=upper,
                                    notes=notes,complete=complete)
    complete = all(r['complete'] for r in validated.values())
    if type(export.get('review_complete')) is not bool or export['review_complete'] != complete:
        raise ValueError('Incorrect complete/partial declaration')
    return dict(annotator=who.strip(),complete=complete,records=validated)


def agreement(pack, reviews):
    identities = [r['annotator'].casefold() for r in reviews]
    if len(set(identities)) != len(identities):
        raise ValueError('Repeated annotator identity cannot count as independent review')
    rows = []
    for original in pack['records']:
        usable = [dict(annotator=r['annotator'],**r['records'][original['id']]) for r in reviews
                  if r['records'][original['id']]['complete']]
        statuses = {r['status'] for r in usable}
        outcome = 'unreviewed' if not usable else 'single-review' if len(usable)==1 else next(iter(statuses)) if len(statuses)==1 else 'disagreement'
        interval = None
        if len(usable)>=2 and statuses=={'matches'}:
            # Union preserves both annotators' uncertainty; intersection would
            # manufacture precision or silently discard disjoint opinions.
            interval = dict(lower_sample=min(r['lower_sample'] for r in usable),
                            upper_sample=max(r['upper_sample'] for r in usable),
                            intervals_overlap=max(r['lower_sample'] for r in usable)<=min(r['upper_sample'] for r in usable),
                            separation_samples=max(0,max(r['lower_sample'] for r in usable)-min(r['upper_sample'] for r in usable)))
        rows.append(dict(id=original['id'],wav_sha256=original['wav_sha256'],
                         pcm_sha256=original['pcm_sha256'],validated_reviews=usable,
                         descriptive_agreement=outcome,conservative_interval_union=interval))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('html','input-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--annotations',type=Path,action='append',required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    pack = load_pack(a.html,a.input_audit)
    reviews = [validate_export(json.loads(path.read_text()),pack) for path in a.annotations]
    rows = agreement(pack,reviews)
    summary = {status:sum(r['descriptive_agreement']==status for r in rows)
               for status in ('unreviewed','single-review','matches','mismatch','uncertain','disagreement')}
    result = dict(role='Review export integrity and descriptive agreement; not gold labels',
                  source_sha256=sha(Path(__file__)),html_sha256=sha(a.html),input_audit_sha256=sha(a.input_audit),
                  annotation_export_sha256=[sha(path) for path in a.annotations],pack_id=pack['pack_id'],
                  exports=len(reviews),complete_exports=sum(r['complete'] for r in reviews),
                  summary=summary,records=rows,human_origin_independently_verified=False,
                  independent_annotator_identities_verified=False,gold_endpoints_created=False,
                  labels_or_training_inputs_changed=False,models_scored=False,
                  limits='Valid JSON, claimed identity and agreement do not establish actual independent human listening, phonetic accuracy or inter-rater reliability. Partial/dissent/uncertainty retained. No automatic label replacement, candidate selection, final recall or latency acceptance.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
