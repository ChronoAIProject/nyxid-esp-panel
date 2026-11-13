"""Registered v2 positive jitter cap; no source selection or RNG calls."""
import ast
import hashlib
import json
from pathlib import Path


def legal_max(row):
    maximum=min(27,150-(row['keyword_frame']-row['lead_frame']))
    if type(maximum) is not int or maximum<3:
        raise ValueError('Complete generated prefix cannot fit fixed150 context')
    if row.get('positive_jitter_max',maximum)!=maximum:
        raise ValueError('Changed positive jitter sidecar')
    return maximum


def effective_jitter(row,draw):
    if type(draw) is not int or not 3<=draw<=27:raise ValueError('Invalid original jitter draw')
    return min(draw,legal_max(row))


class CropRule:
    def __init__(self,trace_path=None):
        self.path=trace_path;self.log=None;self.count=0;self.capped=0;self.first=[]

    def __call__(self,row,draw):
        effective=effective_jitter(row,draw)
        item=dict(index=self.count+1,source_id=row['source_id'],variant=row['variant'],
                  original_jitter=draw,effective_jitter=effective,legal_max=legal_max(row))
        self.count+=1;self.capped+=int(effective!=draw)
        if len(self.first)<200:self.first.append(item)
        if self.path is not None:
            if self.log is None:self.log=Path(self.path).open('x')
            self.log.write(json.dumps(item,sort_keys=True)+'\n')
        return effective

    def finish(self):
        if self.log:self.log.close()
        result=dict(positive_draws=self.count,capped_draws=self.capped,first200=self.first)
        if self.path is not None:
            h=hashlib.sha256()
            with Path(self.path).open('rb') as f:
                for b in iter(lambda:f.read(4194304),b''):h.update(b)
            result.update(trace_path=str(self.path),trace_sha256=h.hexdigest())
        return result


def apply_crop_rule(tree):
    original=ast.parse("end = int(row['keyword_frame'] + rng.integers(3, 28))").body[0]
    replacement=ast.parse("end = int(row['keyword_frame'] + HEY_PREFIX_CROP(row, int(rng.integers(3, 28))))").body[0]
    matches=0
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal matches
            if ast.dump(node,include_attributes=False)==ast.dump(original,include_attributes=False):
                matches+=1;return ast.copy_location(replacement,node)
            return self.generic_visit(node)
    tree=Rewrite().visit(tree)
    if matches!=1:raise ValueError('Original positive sampler changed')
    return ast.fix_missing_locations(tree)
