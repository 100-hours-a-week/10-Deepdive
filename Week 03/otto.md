# Pandas GroupBy 한눈에 정리

## 1. GroupBy란?

`groupby()`는 데이터를 특정 기준으로 묶고, 각 그룹별로 계산을 적용하는 Pandas 기능이다.

예를 들어 학생 성적 데이터가 있을 때, 전체 평균만 보는 것이 아니라 **학생별 평균 점수**, **과목별 평균 점수**처럼 기준을 나누어 분석할 수 있다.

```python
df.groupby("Name")["Score"].mean()
```

위 코드는 `Name`을 기준으로 데이터를 묶고, 각 학생의 평균 점수를 계산한다.

---

## 2. GroupBy 동작 흐름

GroupBy는 크게 3단계로 동작한다.

```text
전체 데이터
   ↓
Split: 기준에 따라 그룹으로 나누기
   ↓
Apply: 각 그룹에 함수 적용하기
   ↓
Combine: 결과를 하나로 합치기
```

| 단계      | 의미           | 예시         |
| ------- | ------------ | ---------- |
| Split   | 데이터를 기준별로 나눔 | 학생별로 나누기   |
| Apply   | 각 그룹에 계산 적용  | 평균 점수 계산   |
| Combine | 결과를 다시 합침    | 학생별 평균표 생성 |

즉, GroupBy는 데이터를 **나누고 → 계산하고 → 다시 정리하는 기능**이다.

---

## 3. 기본 예시

```python
import pandas as pd

data = {
    "Name": ["John", "Anna", "John", "Anna"],
    "Subject": ["Math", "Math", "Science", "Science"],
    "Score": [85, 90, 78, 88]
}

df = pd.DataFrame(data)

result = df.groupby("Name")["Score"].mean()
print(result)
```

### 결과 해석

| Name | 평균 점수 |
| ---- | ----: |
| Anna |    89 |
| John |  81.5 |

전체 평균만 보면 학생별 차이를 알기 어렵다.
하지만 GroupBy를 사용하면 학생별 차이를 쉽게 확인할 수 있다.

---

## 4. 자주 쓰는 집계 함수

| 함수          | 의미     |
| ----------- | ------ |
| `sum()`     | 합계     |
| `mean()`    | 평균     |
| `count()`   | 개수     |
| `max()`     | 최댓값    |
| `min()`     | 최솟값    |
| `nunique()` | 고유값 개수 |

예시:

```python
df.groupby("Name")["Score"].sum()
```

학생별 점수 합계를 계산한다.

---

## 5. agg()로 여러 통계량 한 번에 보기

`agg()`는 여러 집계 함수를 한 번에 적용할 때 사용한다.

```python
df.groupby("Name")["Score"].agg(["sum", "mean", "max", "min"])
```

| Name | sum | mean | max | min |
| ---- | --: | ---: | --: | --: |
| Anna | 178 |   89 |  90 |  88 |
| John | 163 | 81.5 |  85 |  78 |

`.agg()`를 사용하면 합계, 평균, 최댓값, 최솟값을 한 번에 비교할 수 있다.

---

## 6. 여러 열 기준 GroupBy

GroupBy는 하나의 열뿐만 아니라 여러 열을 기준으로도 사용할 수 있다.

```python
df.groupby(["Name", "Subject"])["Score"].mean()
```

이 코드는 학생 이름과 과목을 함께 기준으로 묶는다.

```text
Name 기준
  └── Subject 기준
        └── Score 평균 계산
```

여러 기준을 사용하면 데이터를 더 세부적으로 나누어 분석할 수 있다.

---

## 7. apply()로 직접 만든 계산 적용하기

기본 함수만으로 부족할 때는 `apply()`를 사용할 수 있다.

```python
df.groupby("Name")["Score"].apply(lambda x: x.max() - x.min())
```

이 코드는 학생별로 최고 점수와 최저 점수의 차이를 계산한다.

| 기능        | 사용 목적               |
| --------- | ------------------- |
| 기본 집계 함수  | 평균, 합계, 개수처럼 정해진 계산 |
| `apply()` | 내가 직접 만든 계산 방식 적용   |

---

## 8. GroupBy와 Pivot Table 차이

| 기능          | 핵심 목적         | 예시          |
| ----------- | ------------- | ----------- |
| GroupBy     | 기준별로 묶고 계산    | 학생별 평균 점수   |
| Pivot Table | 행과 열 기준으로 재배치 | 학생 × 과목 점수표 |

GroupBy는 **계산 중심**이고, Pivot Table은 **표 구조 변경 중심**이다.

```python
pd.pivot_table(
    df,
    values="Score",
    index="Name",
    columns="Subject",
    aggfunc="mean"
)
```

---

## 9. 핵심 정리

Pandas의 `groupby()`는 데이터를 특정 기준으로 나누고, 각 그룹별로 계산을 적용하는 기능이다.

GroupBy는 **Split → Apply → Combine** 구조로 동작한다.
즉, 데이터를 나누고, 각 그룹에 함수를 적용하고, 결과를 다시 하나로 합친다.

`mean()`, `sum()`, `count()` 같은 기본 집계 함수로 그룹별 통계값을 구할 수 있고, `.agg()`를 사용하면 여러 통계량을 한 번에 볼 수 있다.
또한 `apply()`를 사용하면 직접 만든 계산 방식도 그룹별로 적용할 수 있다.

결국 GroupBy는 전체 데이터를 기준별로 나누어 비교하고, 데이터 안의 차이와 패턴을 쉽게 파악하게 해주는 Pandas의 핵심 기능이다.

---

## 한 줄 요약

`groupby()`는 데이터를 기준별로 나누고 계산해서, 그룹별 차이와 패턴을 쉽게 확인할 수 있게 해주는 기능이다.
