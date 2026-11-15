"""English-only pinned upstream Melo inference; no audio loading or sentence split.

Load unchanged upstream acoustic model and English frontend. The English module
imports one pure helper from the Japanese frontend; copy that exact AST function
instead of importing Japanese models/dictionaries. Replace both BERT model IDs
with one authenticated local snapshot. No fake torchaudio module, model edits,
normalization, audio repair, sentence splitting or engine waveform trimming.
"""
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['HF_HUB_OFFLINE']='1'
import ast
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'.local/wakeword/melo-source-v1'
BERT=ROOT/'.local/wakeword/melo-bert-pinned-v1'
MODEL=ROOT/'.local/wakeword/melo-english-v2-pinned-v1'

def upstream_module(filename,qualname,copy_phone_helper=False):
 tree=ast.parse(filename.read_text())
 helper=next(n for n in ast.parse((SOURCE/'melo/text/japanese.py').read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name=='distribute_phone') if copy_phone_helper else None
 class Adapter(ast.NodeTransformer):
  def visit_ImportFrom(self,node):
   if copy_phone_helper and node.module=='japanese' and any(a.name=='distribute_phone' for a in node.names):return helper
   return node
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Name) and t.id=='model_id' for t in node.targets):node.value=ast.Constant(str(BERT))
   return self.generic_visit(node)
 module=types.ModuleType(qualname);module.__file__=str(filename);module.__package__='melo.text';sys.modules[qualname]=module
 exec(compile(ast.fix_missing_locations(Adapter().visit(tree)),str(filename),'exec'),module.__dict__)
 return module

class EnglishFrontend:
 def __init__(self):
  sys.path.insert(0,str(SOURCE))
  self.english=upstream_module(SOURCE/'melo/text/english.py','melo.text.english_research_v1',True)
  self.bert_module=None
 def phones(self,text):
  normalized=self.english.text_normalize(text)
  phones,tones,word2ph=self.english.g2p(normalized)
  return dict(normalized=normalized,phones=phones,tones=tones,word2ph=word2ph,
              bert_token_ids=self.english.tokenizer(normalized)['input_ids'])
 def tensors(self,text,config,device):
  import torch
  from melo import commons
  from melo.text import cleaned_text_to_sequence
  info=self.phones(text)
  phones,tones,languages=cleaned_text_to_sequence(info['phones'],info['tones'],'EN',{s:i for i,s in enumerate(config['symbols'])})
  word2ph=list(info['word2ph'])
  if config['data']['add_blank']:
   phones=commons.intersperse(phones,0);tones=commons.intersperse(tones,0);languages=commons.intersperse(languages,0)
   word2ph=[n*2 for n in word2ph];word2ph[0]+=1
  if self.bert_module is None:self.bert_module=upstream_module(SOURCE/'melo/text/english_bert.py','melo.text.english_bert_research_v1')
  ja_bert=self.bert_module.get_bert_feature(info['normalized'],word2ph,device=device)
  assert ja_bert.shape==(768,len(phones))
  bert=torch.zeros(1024,len(phones))
  info.update(input_phone_ids=phones,input_tone_ids=tones,input_language_ids=languages,expanded_word2ph=word2ph)
  return info,(torch.tensor(phones,device=device).unsqueeze(0),torch.tensor(tones,device=device).unsqueeze(0),torch.tensor(languages,device=device).unsqueeze(0),bert.to(device).unsqueeze(0),ja_bert.to(device).unsqueeze(0))

def load_acoustic_model(config,device='cuda:0'):
 import torch
 from melo.models import SynthesizerTrn
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0) and device=='cuda:0'
 model=SynthesizerTrn(len(config['symbols']),config['data']['filter_length']//2+1,
                       config['train']['segment_size']//config['data']['hop_length'],
                       n_speakers=config['data']['n_speakers'],num_tones=config['num_tones'],
                       num_languages=config['num_languages'],**config['model']).to(device).eval()
 checkpoint=torch.load(MODEL/'checkpoint.pth',map_location='cpu',weights_only=False)
 model.load_state_dict(checkpoint['model'],strict=True)
 return model
