"""Metadata only; no models loaded, candidate outcomes or credentials inspected."""
from pathlib import Path
import json
import sys
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'tools/posttrain'))
from pref_artifact_io_v1 import sha
paths={'checkpoint':'/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1.safetensors','base_spec':'/mnt/c/ml_artifacts/general_chat/sft/sft_spec.json','vocab':'/mnt/c/ml_artifacts/general_chat/sft/pk_sft_v1.gguf','val_split':'/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/data/smol_smoltalk_qa_v1/dialogues_val.jsonl','train_split':'/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/corpus/data/smol_smoltalk_qa_v1/dialogues_train.jsonl'}
report={name:{'path':path,'sha256':sha(path)} for name,path in paths.items()}
expected=json.loads(Path('/mnt/c/ml_artifacts/pref_full/sample/sample_summary.json').read_text())['gguf_sha256']
assert report['vocab']['sha256']==expected
report['same_winner_GGUF_as_sampled_candidates']=True
report['before_dpo_outcomes']=True
report['provider_or_gpu_calls']=0
(Path(__file__).parent/'existing_input_pins.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
