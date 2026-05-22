# FastAPI 예외 처리와 AI API 안정성 전략

---

## 1. 왜 예외 처리가 중요한가?

AI API는 일반 API보다 실패 가능성이 훨씬 많다.

### 사용자 쪽 문제 — 잘못된 입력, 없는 모델 요청, 텍스트가 너무 긴 경우

- 존재하지 않는 모델 ID 요청
- 잘못된 입력 데이터 (타입 오류, 범위 초과, 빈 값)
- 입력 텍스트가 너무 김

### 모델 쪽 문제 — 서버 내부에서 터지는 경우

- AI 모델 추론 실패
- 모델 추론 중 GPU 메모리 부족

예외 처리가 없으면:

```text
500 Internal Server Error
```

- 사용자에게 원인 모를 500 에러만 반환
- 내부 오류 메시지가 그대로 노출 → 보안 위협
- 서비스 신뢰도 하락

> "AI API는 언제든 실패할 수 있기 때문에, 예외 처리는 선택이 아니라 필수입니다."

---

## 2. 기본 예외 처리 — HTTPException

FastAPI가 기본 제공하는 예외. 상태 코드와 메시지를 함께 반환한다.

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/model/{model_id}")
def get_model(model_id: str):

    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=404,
            detail=f"모델 '{model_id}'을 찾을 수 없습니다."
        )

    return {"model": model_id}
```

### 클라이언트 응답

```json
HTTP 404 Not Found

{
  "detail": "모델 'gpt-99'을 찾을 수 없습니다."
}
```

### 주요 상태 코드

| 코드 | 의미 | AI 서빙 예시 |
|---|---|---|
| 400 | Bad Request | 입력값 범위 초과 |
| 404 | Not Found | 요청한 모델 ID 없음 |
| 422 | Unprocessable Entity | 타입 오류 |
| 500 | Internal Server Error | 모델 추론 실패, GPU OOM |

---

## 3. Pydantic 자동 검증 (422 에러)

FastAPI는 Pydantic 기반 입력 검증을 자동으로 처리한다.

```python
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):

    age: int = Field(gt=0, lt=120)

    salary: float = Field(gt=0)

    text: str
```

### 잘못된 요청 예시

```json
{
  "age": "스물",
  "salary": "많이"
}
```

자동으로 422 에러 반환:

```json
HTTP 422 Unprocessable Entity

{
  "detail": [
    {
      "loc": ["body", "age"],
      "msg": "value is not a valid integer"
    }
  ]
}
```

### 장점

- 잘못된 데이터를 모델에 전달하기 전에 차단 가능
- 서버 오류 예방
- 데이터 신뢰성 확보

---

## 4. 커스텀 예외 클래스

`HTTPException` 하나로 모든 에러를 처리하면 관리가 어렵다.

AI 서비스 전용 예외 계층을 만든다.

```python
# exceptions.py

class AIServiceException(Exception):

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code


class ModelNotFoundException(AIServiceException):

    def __init__(self, model_id: str):
        super().__init__(
            f"모델 '{model_id}'을 찾을 수 없습니다.",
            404
        )


class InputTooLongException(AIServiceException):

    def __init__(self, max_length: int, actual_length: int):
        super().__init__(
            f"입력이 너무 깁니다. 최대 {max_length}자",
            400
        )


class ModelInferenceException(AIServiceException):

    def __init__(self, reason: str):
        super().__init__(
            "AI 모델 예측 중 오류가 발생했습니다.",
            500
        )

        # 상세 원인은 로그에만 기록
        self.internal_reason = reason
```

---

## 5. 예외 핸들러 등록

```python
# main.py

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from exceptions import AIServiceException

app = FastAPI()

logger = logging.getLogger(__name__)


@app.exception_handler(AIServiceException)
async def ai_exception_handler(
    request: Request,
    exc: AIServiceException
):

    # internal_reason 속성이 있으면
    # 상세 원인을 서버 로그에만 기록
    if hasattr(exc, "internal_reason"):
        logger.error(
            f"Inference error: {exc.internal_reason}"
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message
        }
    )
```

### 클라이언트 응답

```json
HTTP 500 Internal Server Error

{
  "error": "ModelInferenceException",
  "message": "AI 모델 예측 중 오류가 발생했습니다."
}
```

> 핵심:
>
> `Tensor dimension mismatch` 같은 내부 오류는 클라이언트에 직접 노출하지 않는다.
>
> 사용자에게는 친화적인 메시지만 제공하고,
> 상세 원인은 서버 로그에만 기록한다.

---

## 6. 실제 시나리오 — AI 예측 API

```python
@app.post("/predict")
async def predict(request: PredictRequest):

    # 1단계: 모델 존재 확인
    if request.model_id not in AVAILABLE_MODELS:
        raise ModelNotFoundException(request.model_id)

    # 2단계: 추가 입력 검증
    # (Pydantic이 잡지 못하는 비즈니스 로직 검증)

    # 너무 짧은 입력 방어
    if len(request.text) < 5:
        raise HTTPException(
            status_code=400,
            detail="입력 문장이 너무 짧습니다."
        )

    # 너무 긴 입력 방어
    if len(request.text) > 512:
        raise InputTooLongException(
            max_length=512,
            actual_length=len(request.text)
        )

    # 3단계: AI 모델 추론
    try:

        result = model.predict(request)

        return {
            "label": result["label"],
            "score": result["score"]
        }

    # 모델 추론 실패 대응
    except Exception as e:

        raise ModelInferenceException(
            reason=str(e)
        )
```

---

## 7. 3단계 방어선

```text
1단계 — Pydantic 자동 검증
         타입 오류, 범위 초과를 코드 없이 차단

2단계 — 커스텀 예외 처리
         비즈니스 로직 오류를 명확하게 분류

3단계 — 전역 예외 핸들러
         예상 못한 모든 에러가 날것으로 나가지 않게 차단
```

```python
@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    # 상세 로그 기록
    logger.error(
        f"Unexpected error: {exc}",
        exc_info=True
    )

    return JSONResponse(
        status_code=500,
        content={
            "message": (
                "서버 오류가 발생했습니다. "
                "잠시 후 다시 시도해주세요."
            )
        }
    )
```

---

## 8. 결론

FastAPI 예외 처리 = 서버 보호 + 클라이언트 배려

### 서비스 안정성

- 서버가 죽지 않고 계속 응답 가능

### 보안

- 내부 오류를 외부에 노출하지 않음

### 사용자 경험

- 원인을 알 수 있는 명확한 에러 메시지 제공

특히 AI API는 일반 CRUD API보다 실패 경로가 많기 때문에,
이 구조가 더욱 중요하다.