"""Validate opt-in training anchors without mutating rows or PTQ sampling."""
import math


def endpoint_map(rows, records, minimum_end):
    if not records:
        return {}
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Ambiguous training feature identity')
    result = {}
    for record in records:
        identity = record['id']
        if identity in result or identity not in by_id:
            raise ValueError('Duplicate or unknown endpoint feature identity')
        row = by_id[identity]
        if (row['split'], row['kind'], row['label']) != ('training', 'tts', 1):
            raise ValueError('Endpoint overrides require training-positive TTS')
        for key in ('source_sha256', 'features_sha256', 'engine', 'variant'):
            if record[key] != row[key]:
                raise ValueError('Endpoint source/feature identity differs')
        if record['feature_directory'] != row['_feature_directory'] or record['old_keyword_end'] != row['keyword_end']:
            raise ValueError('Endpoint original directory or marker differs')
        value = record['new_keyword_end']
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Endpoint must be a finite number')
        if not minimum_end * .08 <= value < row['keyword_end'] or math.ceil((value + .32) / .08) > row['frames']:
            raise ValueError('Endpoint lies outside bounded warm positive sampling')
        result[identity] = value
    return result
