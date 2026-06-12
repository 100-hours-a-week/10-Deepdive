# AdamW vs Adam
 
---
 
## 01. Optimizer란?

모델 학습 = 손실(Loss)을 줄이는 방향으로 가중치(W)를 업데이트하는 과정

기본적인 경사하강법(Gradient Descent)은 다음과 같다.

```
W ← W - lr × gradient
```

Optimizer는 **얼마나, 어느 방향으로, 학습률을 어떻게 조절할지** 결정하는 알고리즘

---
 
## 02. ADAM 이란?

> **Ada**ptive **M**oment Estimation

**Momentum + RMSProp의 장점을 결합**한 최적화 알고리즘
 
| | 역할 |
|---|---|
| Momentum | 이전 기울기를 기억해 진동을 줄이고 빠르게 수렴 |
| RMSProp | 파라미터마다 다른 학습률 적용 |
| **Adam** | 이 두 가지를 동시에 사용 |

### 엄데이트 과정

```
m_t = β1 * m_(t-1) + (1 - β1) * g_t        # 1차 모멘트 (방향, 관성)
v_t = β2 * v_(t-1) + (1 - β2) * g_t²       # 2차 모멘트 (크기, 분산)
 
m̂_t = m_t / (1 - β1^t)                      # 편향 보정
v̂_t = v_t / (1 - β2^t)
 
W ← W - lr × m̂_t / (√v̂_t + ε)
```
 
| 하이퍼파라미터 | 기본값 | 역할 |
|---|---|---|
| β1 | 0.9 | 방향 평활화 |
| β2 | 0.999 | 크기 평활화 |
| ε | 1e-8 | 0 나눗셈 방지 |
 
### ADAM의 장점

- 학습 초반 수렴 속도가 빠름
- 하이퍼파라미터에 덜 민감
- Sparse Gradient에 강함
- 대부분의 딥러닝 문제에서 안정적으로 동작

---
 
## 03. ADAM의 문제점

Adam은 학습은 잘하지만 **일반화 성능(Generalization)** 이 기대보다 떨어지는 경우가 있다.
 
> 일반화 = 학습 데이터가 아닌 새로운 데이터에서도 잘 동작하는 능력
 
이를 해결하기 위해 **Weight Decay(L2 Regularization)** 를 적용하는데,

Adam에서는 이게 제대로 작동하지 않는다.
 
### Weight Decay(L2 Regularization)란?
 
가중치가 너무 커지는 것을 막아 과적합을 줄이는 기법
 
```
Loss_total = Loss + λ × Σ(W²)
```
 
→ gradient에 `λ × W` 가 추가됨
 
### 왜 Adam에서 문제가 생기나?
 
SGD에서는 L2 Regularization과 Weight Decay가 사실상 동일하게 동작하지만,
**Adam에서는 다름**
 
- L2 패널티가 gradient에 포함되면 → **v(2차 모멘트) 계산에도 영향**을 줌
- 자주 업데이트되는 파라미터 → v가 크게 누적 → 패널티 효과 희석
- 덜 업데이트되는 파라미터 → 상대적으로 패널티를 과하게 받음
**결과: 정규화 효과가 파라미터마다 불균등 → 의도한 Weight Decay가 작동하지 않음**
 
---

## 4. AdamW — Weight Decay를 분리
 
> **2017, Loshchilov & Hutter 제안**  
> *Decoupled Weight Decay Regularization*
 
### 핵심 아이디어
 
> L2 패널티를 gradient에 섞지 말고, **가중치 업데이트 단계에서 직접 감쇠**시키자
 
### 업데이트 과정
 
```
# gradient 계산은 Adam과 동일
m_t = β1 * m_(t-1) + (1 - β1) * g_t
v_t = β2 * v_(t-1) + (1 - β2) * g_t²
 
# Weight Decay를 별도로 적용 ← 핵심 차이
W ← W - lr × m̂_t / (√v̂_t + ε)  -  lr × λ × W
         ↑ Adam과 동일                 ↑ 여기서 직접 감쇠
```
 
마지막 항이 Weight Decay.
Adam의 적응형 학습률은 그대로 사용하면서, Weight Decay는 **독립적으로** 적용된다.
 
---
 
## 5. Adam vs AdamW 비교
 
| 항목 | Adam + L2 | AdamW |
|---|---|---|
| 정규화 방식 | Gradient에 패널티 추가 | Weight 직접 감소 |
| Adaptive LR 영향 | 정규화도 영향 받음 | 받지 않음 |
| Weight Decay 효과 | 파라미터마다 불균등, 왜곡 | 모든 파라미터에 균등 |
| 일반화 성능 | 상대적으로 낮음 | 더 우수 |
| 최신 모델 사용 | 감소 추세 | **사실상 표준** |
 
---
 
## 6. PyTorch 코드 비교
 
```python
import torch
import torch.nn as nn
 
model = nn.Linear(10, 1)
 
# Adam + L2 (weight_decay가 내부적으로 gradient에 합산됨)
optimizer_adam = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2   # 사실상 L2 regularization
)
 
# AdamW (weight decay를 gradient와 분리해서 적용)
optimizer_adamw = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2   # 진짜 weight decay
)
```
 
> **실무 팁**: `Adam(weight_decay=...)` 은 L2와 동일하게 동작  
> 진짜 Weight Decay를 원하면 **AdamW를 써야 함**
 
---
 
## 7. 언제 AdamW를 쓸까?
 
| 상황 | 추천 |
|---|---|
| Transformer 계열 (BERT, GPT, ViT 등) | **AdamW** (거의 표준) |
| 정규화가 중요한 대형 모델 | **AdamW** |
| 간단한 분류/회귀 태스크 | Adam도 무방 |
| 빠른 프로토타이핑 | Adam |
