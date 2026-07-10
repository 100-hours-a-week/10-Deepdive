> QLoRA가 양자화된 LLM 위에서 LoRA Adapter를 학습하는 구조를 설명하고, 전체 파인튜닝 대비 학습 메모리와 비용을 줄일 수 있는 원리를 분석하시오.
> 

## Full fine-tuning의 문제점

Full fine-tuning은 사전 학습된 모델의 모든 파라미터를 전부 다시 학습하는 방식이다.

새로운 데이터에 맞춰 조정할 수 있지만 많은 파라미터를 메모리에 올려야하므로 많은 메모리와 비용이 필요하다.

## LoRA

LoRA는 기반 모델을 고정한 뒤 작은 LoRA Adapter만 추가하여 학습하는 파인튜닝 기법이다.

Adapter만 학습시켜서 작은 파라미터로도 파인튜닝이 가능하도록 해준다.

!image.png

#### 파인 튜닝이 없는 경우

$h = x * W$

여기서 $x$는 입력, $W$는 기존 모델의 가중치, $h$는 출력이다.

#### LoRA 적용

$h = (x * W) + (x * A * B)$

LoRA를 적용하면 기존 가중치 $W$는 고정하고 추가적인 변화량을 작은 행렬인 $A$와 $B$로 학습한다.

이때 A와 B를 쓰는 이유는 작은 파라미터를 사용하기 위해 저랭크 r로 축소하고 복원하기 위하여 A와 B를 사용한다.

- $W$ = 4096 * 4096

만약 이 공식에서 가중치 W의 차원이 4096 * 4096이라면 16,777,216개의 파라미터를 가지게 된다.

랭크 r = 16을 사용한다고 가정하면

- A = 4096 * 16
- B = 16 * 4096

A 차원을 축소하고 다시 B 으로 기존 출력 차원으로 복원한다. 

이렇게 할 경우 Adapter는 2 * *16 ** 4096 = 131,072개의 파라미터만 사용해 기존 대비 0.8% 파라미터만으로 학습이 가능해진다. 

## QLoRA

QLoRA는 LoRA의 구조를 유지하면서, 기반 모델을 4-bit로 양자화하여 사용하는 파인튜닝 기법이다.

LoRA에서는 기반 모델을 고정하고 Adapter만 학습한다.

QLoRA는 여기서 한 단계 더 나아가 고정된 기반 모델을 4-bit로 양자화하여 GPU 메모리 사용량을 줄인다.

QLoRA의 구조는 다음과 같다.

1. 사전 학습된 기반 모델을 4-bit로 양자화한다.
2. 양자화된 기반 모델은 학습하지 않고 고정한다.
3. 기반 모델 위에 Adapter를 추가한다.
4. Adapter만 FP16 또는 BF16 정밀도로 학습한다

기반 모델은 4-bit로 저장되어 메모리를 적게 사용하고, Adapter는 실제 학습되는 파라미터이므로 안정적인 학습을 위해 16-bit 정밀도로 유지된다.

## QLoRA가 메모리와 비용을 줄이는 원리

QLoRA는 Full fine-tuning에 비해 두 가지 방식으로 메모리와 비용을 줄인다.

1. 기반 모델을 학습하지 않는다.
    
    기반 모델의 가중치는 고정하고 작은 Adapter만 학습한다.
    
2. 기반 모델을 4-bit로 양자화한다.
    
    기반 모델은 학습되지 않고 고정되어 있으므로 높은 정밀도로 저장하지 않고 낮은 정밀도로 저장하여 메모리 사용량을 줄인다.
    

위 두 가지 방법으로 대형 모델도 제한된 환경에서 파인튜닝 할 수 있다.

## 예시 코드

#### 1. 모델을 양자화하여 로드하는 과정

```python
# 필요한 라이브러리를 설치합니다.
!pip install -q -U transformers accelerate peft trl bitsandbytes datasets
!pip uninstall -q -y torchao

# PyTorch 라이브러리를 불러옵니다.
import torch

# Hugging Face Transformers에서 모델 로딩, 토크나이저, 양자화 설정 클래스를 가져옵니다.
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 실습용 모델 ID
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# NF4 + Double Quantization 양자화 설정을 만듭니다.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 기반 모델을 4-bit로 로드하여 GPU 메모리 사용량을 줄입니다.
    bnb_4bit_quant_type="nf4",               # QLoRA에서 사용하는 NF4 양자화 타입을 지정합니다.
    bnb_4bit_compute_dtype=torch.bfloat16,   # 4-bit로 저장된 가중치를 실제 연산 시 bfloat16으로 계산합니다.
    bnb_4bit_use_double_quant=True,          # 양자화 상수까지 한 번 더 양자화해 추가 메모리를 줄입니다.
)

# 4-bit 양자화 설정과 함께 기반 모델을 로드합니다.
model = AutoModelForCausalLM.from_pretrained(
    model_id,                                # 위에서 정의한 Qwen2.5-1.5B-Instruct 모델 ID
    quantization_config=bnb_config,          # NF4 + Double Quantization 설정 전달
    device_map="auto",                       # 사용 가능한 GPU에 모델을 자동 배치
)

# 모델에 맞는 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Causal LM 학습에서 padding을 위해 pad_token을 eos_token으로 설정합니다.
tokenizer.pad_token = tokenizer.eos_token
```

#### 2. LoRA Adapter 부착

```python
# PEFT 라이브러리에서 QLoRA 학습에 필요한 기능을 가져옵니다.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# k-bit 학습 준비를 수행합니다.
model = prepare_model_for_kbit_training(model)

# LoRA 어댑터 설정을 만듭니다.
lora_config = LoraConfig(
    r=16,                         # LoRA rank. 어댑터가 표현할 수 있는 변화의 크기를 정합니다.
    lora_alpha=32,                # LoRA 출력의 영향력을 조절하는 스케일링 값입니다.
    target_modules="all-linear",  # 모든 선형층에 LoRA를 적용합니다. QLoRA 논문 흐름에서 자주 쓰는 설정입니다.
    lora_dropout=0.05,            # 어댑터 경로에 dropout을 적용해 과적합을 줄입니다.
    bias="none",                  # bias는 학습하지 않습니다. LoRA 어댑터만 학습 대상으로 둡니다.
    task_type="CAUSAL_LM",        # 다음 토큰을 예측하는 텍스트 생성 모델 학습임을 지정합니다.
)

# 4-bit 기반 모델 위에 LoRA 어댑터를 추가합니다.
model = get_peft_model(model, lora_config)
```

#### 3. 학습 진행

```python
# 최신 TRL에서 SFT 학습에 필요한 클래스를 가져옵니다.
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# 데모용 공개 instruction 데이터셋을 작게 불러옵니다.
dataset = load_dataset("trl-lib/Capybara", split="train[:200]")

# SFTTrainer로 학습기를 만듭니다.
trainer = SFTTrainer(
    model=model,                                  # LoRA 어댑터가 붙은 QLoRA 모델
    args=SFTConfig(
        output_dir="./qlora-output",              # 학습 로그와 중간 결과가 저장될 디렉터리
        max_steps=30,                             # 데모용 짧은 학습 스텝 수입니다. 실제 학습에서는 더 크게 설정합니다.
        per_device_train_batch_size=2,            # GPU 한 장에 한 번에 올릴 샘플 수입니다.
        gradient_accumulation_steps=4,            # 4번의 작은 배치를 누적해 더 큰 배치처럼 학습합니다.
        learning_rate=2e-4,                       # LoRA 어댑터 학습률입니다.
        warmup_ratio=0.03,                        # 초반 학습률을 천천히 올리는 워밍업 비율입니다.
        logging_steps=5,                          # 5스텝마다 학습 로그를 출력합니다.
        bf16=True,                                # BF16 mixed precision으로 학습합니다.
        max_length=1024,                          # 한 샘플에서 사용할 최대 토큰 길이입니다.
        gradient_checkpointing=True,              # 중간 활성화 저장을 줄여 GPU 메모리를 절약합니다.
        gradient_checkpointing_kwargs={"use_reentrant": False},  # k-bit 학습과 함께 사용할 때 권장되는 설정입니다.
        optim="paged_adamw_8bit",                 # Paged Optimizer를 사용해 GPU 메모리 부담을 줄입니다.
    ),
    train_dataset=dataset,                        # 학습에 사용할 데이터셋
    processing_class=tokenizer,                   # 텍스트를 토큰으로 바꾸는 토크나이저 전달
)

# 학습을 시작합니다.
trainer.train()

# 학습된 LoRA 어댑터만 저장합니다.
model.save_pretrained("qwen2-5-1-5b-qlora")
```
