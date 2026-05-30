# SciPy의 선형대수 기능을 분석 및 계산 가속화

## Scipy란?

Scipy는 과학적, 기술적 계산을 위한 Python 라이브러리이다.

Numpy 라이브러리를 기반으로 하여 더 복잡한 수학적 연산, 최적화, 통계 분석, 신호 처리 등을 수행할 수 있도록 확장된 기능을 포함합니다.

`scipy.linalg` 모듈을 통해서 행렬 계산, 선형 연립방정식, 행렬 분해 등등 다양한 선형대수 기능을 제공한다.

| 분류 | 대표 함수 | 하는 일 |
| --- | --- | --- |
| 연립방정식 | `solve` | `Ax = b` 형태의 방정식을 풀기 |
| 기본 연산 | `inv`, `det` | 역행렬, 행렬식 계산 |
| 고유값 | `eig` | 고유값·고유벡터 계산 |
| 행렬 분해 | `lu`, `qr`, `cholesky`, `svd` | 행렬을 쪼개서 계산 효율을 높임 |

> 특히 행렬 분해 기능은 Numpy에 없는 다양한 분해 함수를 제공한다.
> 

## Python은 왜 느린가?

같은 행렬 연산을 순수 Pyhton의 반복문으로 작성하면 매우 느리다.

- 인터프리터 방식 - 반복문의 모든 연산을 한 줄씩 해석하며 실행
- 타입 체크 - 매 연산마다 자료형을 확인
- 반복문 자체 - 계산 하나하나를 Pyhton이 직접 처리

행렬이 커질수록 이 비용이 엄청나게 늘어난다.

## Scipy는 어떻게 가속화하는가?

계산을 Python이 직접 하지 않는다.

1. BLAS / LAPACK 활용
    - BLAS: 기본 벡터,행렬 연산 라이브러리
    - LAPACK: 연립방정식, 행렬 분해 등 고급 연산 라이브러리
    - C/Fortran으로 작성되어 최적화된 코드들이다.
    - Numpy와 Scipy는 내부적으로 이 라이브러리를 호출한다.
2. 벡터화 
    - 반복문을 직접 쓰지 않고, 배열 전체를 한번의 명령으로 처리
    - 내부적으로 컴파일된 C 코드가 한번에 실행
3. 컴파일된 코드
    - 핵심 연산이 미리 컴파일된 저수준 코드로 동작
    - Pyhton의 인터프리터 방식을 쓰지 않는다.

### 속도 비교

```jsx
import time
import numpy as np
from scipy import linalg

# 비교할 행렬 크기
N = 100

# 행렬 두 개 준비
A = np.random.rand(N, N)
B = np.random.rand(N, N)

# Python용 리스트
A_list = A.tolist()
B_list = B.tolist()

# Python: 삼중 반복문으로 행렬 곱
def matmul_python(A, B, n):
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

start = time.time()
matmul_python(A_list, B_list, N)
t_python = time.time() - start

# 2) NumPy: @ 연산자
start = time.time()
A @ B
t_numpy = time.time() - start

# 3) SciPy: BLAS 함수 직접 호출
start = time.time()
linalg.blas.dgemm(1.0, A, B)
t_scipy = time.time() - start

# 결과 출력
print(f"Python : {t_python:.6f}초")
print(f"NumPy       : {t_numpy:.6f}초")
print(f"SciPy       : {t_scipy:.6f}초")
```

<img width="386" height="124" alt="image" src="https://github.com/user-attachments/assets/88e29715-9048-40c2-9d9c-2ab09dc06c78" />
