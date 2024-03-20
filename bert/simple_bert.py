from transformers import BertTokenizer, BertModel
import torch
from torch.profiler import ExecutionTraceObserver, profile

def trace_handler(prof):
    prof.export_chrome_trace("bert_kt.json")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# BERT 토크나이저 및 모델 초기화, 모델을 GPU로 이동
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 토크나이징
input_text = "Hello, my dog is cute"
tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 토큰 텐서를 GPU로 이동
tokens = {k: v.to(device) for k, v in tokens.items()}

# 프로파일링을 위한 콜백 함수 정의
et = ExecutionTraceObserver()
et.register_callback("bert_et.json")
with profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
    on_trace_ready=trace_handler
) as prof:
    # 특징 추출
    et.start()
    prof.step()
    with torch.no_grad():
        outputs = model(**tokens)
        last_hidden_states = outputs.last_hidden_state
    et.stop()

et.unregister_callback()
# `last_hidden_states`는 모델의 마지막 레이어에서의 특징을 담고 있습니다.
# `.cpu()`를 호출하여 CPU로 데이터를 이동시키고 `.numpy()`를 호출하여 numpy 배열로 변환할 수 있습니다.
print(last_hidden_states.shape)  # (batch_size, sequence_length, hidden_size)
