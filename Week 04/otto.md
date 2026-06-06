# Gradient Descent와 Adam Optimizer 비교

## 1. 주제

딥러닝 모델은 처음부터 정답을 잘 맞히는 것이 아니라, 학습을 통해 점점 더 나은 예측을 하게 된다.
이때 모델이 학습한다는 것은 결국 **loss를 줄이는 방향으로 parameter를 업데이트하는 과정**이라고 볼 수 있다.

여기서 parameter는 모델이 학습하면서 바꾸는 값이다. 대표적으로 weight와 bias가 있다.

```text
parameter = weight + bias
```

모델 학습의 전체 흐름은 다음과 같다.

```text
입력 데이터
→ 모델 예측
→ 손실 함수로 loss 계산
→ 역전파로 gradient 계산
→ optimizer가 parameter 업데이트
→ 반복
```

Gradient Descent와 Adam은 이 중에서 **optimizer가 parameter를 어떻게 업데이트할 것인가**와 관련된 알고리즘이다.

---

## 2. Gradient Descent란?

Gradient Descent, 즉 경사 하강법은 손실 함수 값을 줄이기 위해 gradient의 반대 방향으로 parameter를 업데이트하는 가장 기본적인 최적화 방법이다.

기본 공식은 다음과 같다.

```text
새 parameter = 현재 parameter - learning_rate × gradient
```

수식으로 표현하면 다음과 같다.

```text
θ = θ - α × ∇J(θ)
```

각 기호의 의미는 다음과 같다.

```text
θ = parameter
α = learning rate
∇J(θ) = loss function의 gradient
```

여기서 중요한 점은 gradient를 더하는 것이 아니라 **뺀다**는 점이다.

gradient는 loss가 가장 빠르게 증가하는 방향을 의미한다.
하지만 모델 학습의 목표는 loss를 증가시키는 것이 아니라 줄이는 것이다.
따라서 gradient 방향이 아니라, gradient의 반대 방향으로 이동해야 한다.

```text
gradient 방향 = loss가 증가하는 방향
-gradient 방향 = loss가 감소하는 방향
```

즉, Gradient Descent는 현재 위치에서 기울기를 보고, loss가 줄어드는 방향으로 조금씩 이동하는 방법이다.

---

## 3. Gradient Descent를 쉽게 이해하기

Gradient Descent는 산을 내려가는 과정으로 비유할 수 있다.

```text
산의 높이 = loss 값
현재 위치 = 현재 parameter 상태
가장 낮은 지점 = loss가 최소가 되는 지점
gradient = 가장 가파르게 올라가는 방향
-gradient = 가장 가파르게 내려가는 방향
learning rate = 한 번에 움직이는 보폭
```

모델은 처음에 랜덤한 parameter에서 시작한다.
처음에는 예측이 부정확하기 때문에 loss가 클 수 있다.

이때 gradient를 계산하고, loss가 줄어드는 방향으로 parameter를 조금씩 수정한다.
이 과정을 반복하면 모델은 점점 더 나은 예측을 하게 된다.

---

## 4. Gradient Descent의 장점

Gradient Descent의 장점은 다음과 같다.

### 1. 원리가 단순하다

Gradient Descent는 “gradient의 반대 방향으로 이동한다”는 하나의 핵심 원리로 이해할 수 있다.
그래서 딥러닝에서 optimizer가 왜 필요한지 이해하기 좋다.

### 2. 구현이 쉽다

기본 업데이트 공식이 단순하다.

```text
parameter = parameter - learning_rate × gradient
```

그래서 직접 구현하거나 학습 원리를 설명하기에 좋다.

### 3. 다른 optimizer를 이해하는 기초가 된다

Adam, RMSProp, AdaGrad 같은 optimizer도 결국 Gradient Descent의 아이디어에서 발전한 것이다.
따라서 Gradient Descent를 이해하면 Adam이 왜 등장했는지도 이해하기 쉬워진다.

---

## 5. Gradient Descent의 단점

Gradient Descent에는 한계도 있다.

### 1. Learning Rate에 민감하다

learning rate가 너무 크면 최저점을 지나쳐 loss가 불안정하게 변할 수 있다.

```text
learning rate가 너무 큼
→ 너무 크게 이동
→ 최저점을 지나칠 수 있음
→ 학습 불안정
```

반대로 learning rate가 너무 작으면 학습이 너무 느리다.

```text
learning rate가 너무 작음
→ 조금씩 이동
→ 학습 시간이 오래 걸림
```

### 2. 모든 parameter에 같은 learning rate를 적용한다

기본 Gradient Descent는 보통 모든 parameter에 같은 learning rate를 적용한다.
하지만 실제 딥러닝 모델에서는 어떤 parameter는 크게 움직여야 하고, 어떤 parameter는 작게 움직여야 할 수 있다.

모든 parameter에 같은 보폭을 적용하는 것은 비효율적일 수 있다.

### 3. 복잡한 loss 지형에서 느리거나 불안정할 수 있다

딥러닝의 loss surface는 단순한 U자 곡선이 아니다.
local minima, saddle point, 좁고 긴 골짜기 같은 복잡한 지형이 존재할 수 있다.

이런 상황에서 기본 Gradient Descent는 지그재그로 움직이거나 수렴이 느려질 수 있다.

---

## 6. Adam Optimizer란?

Adam은 **Adaptive Moment Estimation**의 줄임말이다.
Gradient Descent의 한계를 보완하기 위해 많이 사용되는 optimizer다.

Gradient Descent는 현재 gradient만 보고 parameter를 업데이트한다.

반면 Adam은 두 가지 정보를 함께 사용한다.

```text
1. gradient의 이동 평균
2. gradient 제곱의 이동 평균
```

쉽게 말하면 Adam은 현재 gradient만 보는 것이 아니라, 이전 gradient의 흐름과 gradient 크기의 변화까지 기억하면서 parameter를 업데이트한다.

비유하면 다음과 같다.

```text
Gradient Descent
= 지금 발밑 경사만 보고 내려가는 방법

Adam
= 지금 경사도 보고,
  이전에 어느 방향으로 움직였는지도 보고,
  길이 얼마나 흔들리는지도 보면서 내려가는 방법
```

---

## 7. Adam의 핵심 원리

Adam의 핵심은 크게 두 가지다.

### 1. 1차 모멘트: gradient의 이동 평균

Adam은 최근 gradient들이 대체로 어느 방향을 가리키는지 기억한다.

```text
1차 모멘트
= gradient의 이동 평균
= 최근 gradient 방향의 흐름
```

이것은 Momentum과 비슷한 역할을 한다.
gradient가 조금씩 흔들리더라도, 전체적으로 일관된 방향을 보고 더 안정적으로 이동할 수 있게 도와준다.

### 2. 2차 모멘트: gradient 제곱의 이동 평균

Adam은 gradient의 크기 변화도 함께 기억한다.

```text
2차 모멘트
= gradient 제곱의 이동 평균
= gradient가 얼마나 크게 흔들리는지에 대한 정보
```

이 정보를 사용하면 parameter마다 업데이트 크기를 다르게 조절할 수 있다.

```text
gradient가 큰 parameter
→ 너무 크게 움직이지 않도록 조심스럽게 업데이트

gradient가 작은 parameter
→ 상대적으로 더 움직일 수 있도록 조정
```

즉, Adam은 모든 parameter에 같은 보폭을 적용하는 것이 아니라, parameter별 상황에 맞게 보폭을 조절하는 optimizer라고 볼 수 있다.

---

## 8. Adam의 업데이트 흐름

Adam의 업데이트 흐름을 단순화하면 다음과 같다.

```text
1. 현재 gradient를 계산한다.
2. gradient의 이동 평균을 계산한다.
3. gradient 제곱의 이동 평균을 계산한다.
4. 학습 초반의 편향을 보정한다.
5. 보정된 값을 이용해 parameter를 업데이트한다.
```

Gradient Descent가 단순히 다음과 같이 업데이트한다면,

```text
parameter = parameter - learning_rate × gradient
```

Adam은 gradient의 방향과 크기 변화를 함께 고려하여 더 적응적으로 parameter를 업데이트한다.

---

## 9. Adam의 장점

Adam의 장점은 다음과 같다.

### 1. 학습이 빠르게 안정되는 경우가 많다

Adam은 이전 gradient의 흐름과 gradient 크기 변화를 함께 고려하기 때문에, 단순 Gradient Descent보다 빠르게 loss를 줄이는 경우가 많다.

### 2. Learning Rate 설정 부담이 상대적으로 적다

Gradient Descent는 learning rate에 매우 민감하다.
반면 Adam은 parameter마다 업데이트 크기를 조절하기 때문에 기본값으로도 잘 작동하는 경우가 많다.

Adam에서 자주 사용하는 기본 learning rate는 다음과 같다.

```text
lr = 0.001
```

### 3. 복잡한 딥러닝 모델에서 자주 사용된다

CNN, MLP 같은 딥러닝 모델에서는 parameter가 많고 loss 지형이 복잡하다.
Adam은 이런 상황에서 안정적으로 학습되는 경우가 많아 실습과 실제 모델 개발에서 자주 사용된다.

### 4. Sparse data나 noisy gradient에 비교적 강하다

Adam은 parameter별로 업데이트 크기를 조절하기 때문에, 일부 parameter만 자주 업데이트되는 상황에서도 유리할 수 있다.

---

## 10. Adam의 단점

Adam도 항상 좋은 것은 아니다.

### 1. 내부 원리가 Gradient Descent보다 복잡하다

Gradient Descent는 단순히 gradient 반대 방향으로 이동하면 된다.
하지만 Adam은 이동 평균, 제곱 이동 평균, bias correction 등 추가 개념이 들어간다.

그래서 처음 배우는 입장에서는 Gradient Descent보다 이해하기 어렵다.

### 2. 항상 최고의 일반화 성능을 보장하지 않는다

Adam은 빠르게 loss를 줄이는 경우가 많지만, 항상 test 성능이 가장 좋은 것은 아니다.
어떤 문제에서는 SGD나 SGD with Momentum이 더 좋은 일반화 성능을 보이기도 한다.

즉, Adam을 쓴다고 무조건 가장 좋은 모델이 되는 것은 아니다.

### 3. 추가 hyperparameter가 있다

Adam은 learning rate 외에도 beta1, beta2, epsilon 같은 값을 사용한다.
대부분 기본값을 사용하지만, 상황에 따라 조정이 필요할 수도 있다.

---

## 11. Gradient Descent와 Adam의 알고리즘적 차이

| 구분             | Gradient Descent         | Adam                                             |
| -------------- | ------------------------ | ------------------------------------------------ |
| 기본 아이디어        | gradient 반대 방향으로 이동      | gradient의 방향과 크기 변화를 함께 고려                       |
| 사용하는 정보        | 현재 gradient              | 현재 gradient + gradient 이동 평균 + gradient 제곱 이동 평균 |
| learning rate  | 보통 모든 parameter에 동일하게 적용 | parameter마다 적응적으로 조절                             |
| 과거 gradient 반영 | 거의 없음                    | 이동 평균으로 반영                                       |
| 수렴 속도          | 느릴 수 있음                  | 빠른 경우가 많음                                        |
| 안정성            | learning rate에 민감        | 비교적 안정적                                          |
| 구조             | 단순함                      | 상대적으로 복잡함                                        |
| 장점             | 이해하기 쉽고 기본 원리 학습에 좋음     | 실전 딥러닝에서 빠르고 안정적인 경우가 많음                         |
| 단점             | 느리거나 불안정할 수 있음           | 항상 최고의 일반화 성능을 보장하지 않음                           |

---

## 12. PyTorch 코드에서의 차이

PyTorch에서는 optimizer를 다음과 같이 설정할 수 있다.

SGD를 사용하는 경우:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Adam을 사용하는 경우:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

학습 루프는 거의 비슷하다.

```python
optimizer.zero_grad()

outputs = model(inputs)
loss = criterion(outputs, labels)

loss.backward()
optimizer.step()
```

하지만 `optimizer.step()` 안에서 일어나는 업데이트 방식이 다르다.

```text
SGD / Gradient Descent 계열
= 현재 gradient를 중심으로 parameter 업데이트

Adam
= gradient의 이동 평균과 gradient 제곱의 이동 평균을 이용해 parameter 업데이트
```

즉, 코드상으로는 한 줄 차이처럼 보이지만, 내부 알고리즘은 다르다.

---

## 13. 결론

Gradient Descent와 Adam은 모두 딥러닝 모델의 parameter를 업데이트하기 위한 optimizer다.

Gradient Descent는 가장 기본적인 최적화 방법으로, gradient의 반대 방향으로 learning rate만큼 이동한다.
원리가 단순하고 직관적이기 때문에 딥러닝 학습 구조를 이해하는 데 매우 중요하다.

하지만 Gradient Descent는 learning rate에 민감하고, 모든 parameter에 같은 보폭을 적용하기 때문에 복잡한 딥러닝 모델에서는 학습이 느리거나 불안정할 수 있다.

Adam은 이러한 한계를 보완하기 위해 gradient의 이동 평균과 gradient 제곱의 이동 평균을 사용한다.
이를 통해 과거 gradient의 방향과 크기 변화를 반영하고, parameter마다 적응적으로 업데이트할 수 있다.

따라서 Adam은 실제 딥러닝 실습에서 빠르고 안정적인 optimizer로 자주 사용된다.

하지만 Adam이 항상 정답은 아니다.
어떤 문제에서는 SGD 계열이 더 좋은 일반화 성능을 보일 수 있고, Adam도 learning rate나 beta 값에 따라 결과가 달라질 수 있다.

따라서 optimizer를 선택할 때는 단순히 “Adam이 좋다”라고 외우기보다, 모델의 loss 변화, validation 성능, 학습 안정성을 함께 확인해야 한다.

---

## 15. 한 줄 요약

Gradient Descent는 현재 gradient만 보고 고정된 보폭으로 loss를 줄이는 기본 최적화 방법이고, Adam은 gradient의 방향과 크기 변화를 기억하여 parameter마다 보폭을 조절하는 더 적응적인 optimizer다.
