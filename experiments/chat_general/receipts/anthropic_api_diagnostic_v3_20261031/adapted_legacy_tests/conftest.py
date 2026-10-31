import sys
sys.path.insert(0,'/home/jonat/pk_api_transport_diagnostic_20261031/tools/posttrain')
import chat_judge_api_diagnostic_v3 as diagnostic
sys.modules["chat_judge_api"] = diagnostic
