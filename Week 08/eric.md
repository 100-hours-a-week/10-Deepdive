# 주제

LangChain에서 새로 도입된 Agent를 조사하여 설명하시오.

- 참고 문서: [LangChain Agents 공식 문서](https://docs.langchain.com/oss/python/langchain/agents)

## 1. LangChain Agent란?

Agent는 주어진 작업이 완료될 때까지 루프 안에서 작업을 완료하는 모델이다.

LangChain에서 Agent는 주어진 작업이 완료될 때까지 모델이 반복적으로 판단하고, 필요한 도구를 호출하면서 문제를 해결하는 구조이다.

공식 문서에서는 Agent를 다음과 같이 설명한다.

> Agent = Model + Harness

즉, 에이전트는 단순히 LLM 모델만 의미하는 것이 아니라, 모델이 작업을 수행할 수 있도록 감싸는 실행 구조까지 포함한다.

### Model

Model은 사용자의 요청을 이해하고 다음 행동을 결정하는 핵심 요소이다. 예를 들어 사용자의 질문에 바로 답할지, 검색 도구를 호출할지, 외부 API를 사용할지 등을 판단한다.

### Harness

Harness는 에이전트 루프를 둘러싼 전체 실행 환경을 의미한다.

Harness에는 다음 요소들이 포함된다.

- 모델
- 프롬프트
- 도구
- 미들웨어
- 도구 호출 방식
- 응답 형식
- 실행 중 상태 관리

Harness의 역할은 모델이 주어진 작업을 수행할 때 적절한 시점에 적절한 맥락을 제공하는 것이다.

## 2. 기본 에이전트 생성

LangChain에서는 `create_agent()`를 사용하여 에이전트를 생성할 수 있다.

```python
from langchain.agents import create_agent

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=tools,
)
```

가장 기본적인 에이전트는 모델과 도구 목록을 전달하여 만들 수 있다. 모델은 사용자의 입력을 보고 도구를 사용할지, 직접 답변할지 결정한다.

## 3. Tools

Tools는 에이전트가 외부 기능을 사용할 수 있도록 해주는 도구이다.

#### LangChain Agent에서 사용할 수 있는 도구
- Python 호출 가능 함수
- LangChain Tool
- Tool dictionary

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"
```

Tools를 사용하면 에이전트는 단순히 텍스트만 생성하는 것이 아니라, 실시간 데이터를 가져오거나, 코드를 실행하거나, 외부 데이터베이스를 조회하거나, 실제 서비스의 기능을 수행할 수 있다.

모델은 대화 맥락을 바탕으로 다음을 스스로 결정한다.

- 어떤 도구를 호출할지
- 도구를 언제 호출할지
- 도구에 어떤 인자를 전달할지
- 도구 결과를 바탕으로 어떻게 최종 답변할지

## 4. System Prompt

System prompt는 에이전트의 역할, 말투, 답변 방식, 제한 조건 등을 지정하는 데 사용한다.

LangChain의 `create_agent`에서는 문자열 또는 시스템 메시지를 전달할 수 있다.

```python
agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate.",
)
```

## 5. Structured Output

LangChain Agent는 `response_format=`을 사용하여 구조화된 출력을 만들 수 있다.

```python
from pydantic import BaseModel
from langchain.agents import create_agent

class Answer(BaseModel):
    summary: str
    confidence: float

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=tools,
    response_format=Answer,
)
```

이렇게 설정하면 에이전트의 응답을 단순 문자열이 아니라 `summary`, `confidence` 같은 정해진 필드를 가진 형태로 받을 수 있다.

구조화된 출력은 다음과 같은 상황에서 유용하다.

- API 응답 형식을 고정해야 할 때
- 프론트엔드에서 특정 필드를 바로 사용해야 할 때
- 추천 결과, 분석 결과, 평가 결과 등을 JSON 형태로 내려줘야 할 때
- LLM 응답을 후처리하기 쉽게 만들고 싶을 때

## 6. Invocation

Invocation은 에이전트를 실제로 호출하는 과정이다.

에이전트는 메시지를 입력받아 실행되며, 내부적으로 메시지는 에이전트의 상태를 업데이트한다.

```python
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?",
            }
        ]
    }
)
```

대화 기록을 유지하려면 `thread_id`와 `checkpointer`를 함께 사용해야 한다.

- `thread_id`: 하나의 대화 흐름을 구분하는 ID
- `checkpointer`: 대화 상태를 저장하고 다시 불러오기 위한 저장소

LangSmith에 배포하면 체크포인터가 자동으로 생성된다. 로컬 환경에서는 직접 체크포인터를 설정해야 한다.

```python
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[],
    context_schema=Context,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in San Francisco?",
            }
        ]
    },
    config={"configurable": {"thread_id": str(uuid7())}},
    context=Context(user_id="user-123"),
)
```

여기서 `context`는 도구나 미들웨어가 실행 중에 참고할 수 있는 추가 실행 정보이다. 예를 들어 사용자 ID, API 키, 기능 플래그 등을 전달할 수 있다.

## 7. Middleware

Middleware는 에이전트 내부에서 발생하는 동작을 더 세밀하게 제어할 수 있도록 해준다.

미들웨어를 사용하면 다음과 같은 작업을 할 수 있다.

- 로깅, 분석, 디버깅을 통해 에이전트 동작 추적
- 프롬프트 변경
- 도구 선택 방식 변경
- 출력 형식 변경
- 대체 모델 또는 대체 도구 설정


즉, Middleware는 에이전트가 실행되는 과정에 개입하여 더 안정적이고 통제 가능한 구조를 만들기 위한 기능이다.

## 8. Execution Environment

Agent는 텍스트만 생성하는 것보다 실제 행동을 수행할 때 더 유용하다. Execution environment는 에이전트가 작업할 수 있는 실행 환경을 제공한다.

예를 들어 다음과 같은 기능을 포함할 수 있다.

- 파일 읽기 및 쓰기
- 코드 실행
- 쉘 명령 실행
- 외부 도구 호출
- 샌드박스 환경 사용

이를 통해 에이전트는 단순 답변을 넘어서 파일을 수정하거나, 데이터를 분석하거나, 코드를 실행하는 작업까지 수행할 수 있다.


## 9. Planning and Delegation

Planning and delegation은 복잡한 작업을 여러 하위 작업으로 나누어 처리하는 방식이다.

메인 에이전트는 전체 작업을 조율하고, 실제 세부 작업은 하위 에이전트에게 위임할 수 있다.

이 방식의 장점은 다음과 같다.

- 복잡한 작업을 작은 단위로 나눌 수 있다.
- 하위 에이전트가 독립적인 컨텍스트에서 작업할 수 있다.
- 여러 작업을 병렬로 실행할 수 있다.
- 메인 에이전트의 컨텍스트를 깔끔하게 유지할 수 있다.
- 연구, 코딩, 분석처럼 긴 작업에 적합하다.

예시는 다음과 같다.

```python
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for a query and return a short summary."""
    return f"Search results for: {query}"

backend = StateBackend()

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": "Use the search tool to research the question and summarize key points.",
                    "tools": [search],
                    "model": "anthropic:claude-sonnet-4-6",
                    "middleware": [],
                }
            ],
        ),
    ],
)
```

이 예시에서는 메인 에이전트가 전체 흐름을 관리하고, `researcher`라는 하위 에이전트가 검색과 요약 작업을 담당한다.

## 10. Fault Tolerance

실제 운영 환경에서는 모델 호출 실패, API 제한, 네트워크 오류, 도구 실행 실패 등이 발생할 수 있다.

Fault tolerance는 이러한 오류를 인프라 수준에서 처리하는 기능이다.

예를 들어 다음과 같은 방식으로 사용할 수 있다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        ModelRetryMiddleware(max_retries=3),
        ToolRetryMiddleware(max_retries=2),
    ],
)
```

이렇게 하면 도구나 비즈니스 로직마다 직접 `try/catch`를 작성하지 않아도, 미들웨어가 재시도나 오류 처리를 담당할 수 있다.

## 11. Steering

Steering은 에이전트가 완전히 자율적으로 실행되지 않도록, 특정 결정 지점에 사람을 개입시키는 기능이다.

예를 들어 다음과 같은 작업은 사람의 승인을 받은 뒤 실행하는 것이 안전하다.

- 파일 삭제
- 데이터베이스 수정
- 비용이 큰 API 호출
- 외부 서비스에 쓰기 작업 수행
- 사용자에게 중요한 영향을 주는 결정

예시는 다음과 같다.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="google_genai:gemini-3.5-flash",
    tools=[search],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"write_file": True}
        )
    ],
)
```

이렇게 설정하면 `write_file`과 같은 중요한 도구가 실행되기 전에 에이전트가 잠시 멈추고 사람의 승인을 기다린다.

## 12. Agent가 필요한 경우

Agent는 모든 상황에서 필요한 것은 아니다. 단순히 한 번의 질문에 답하거나 정해진 순서대로 실행되는 작업이라면 Chain이나 일반 LLM 호출만으로도 충분할 수 있다.

Agent가 적합한 경우는 다음과 같다.

- 어떤 도구를 사용할지 모델이 직접 판단해야 하는 경우
- 작업 단계가 고정되어 있지 않은 경우
- 외부 API, 검색, DB 조회 등 여러 도구를 조합해야 하는 경우
- 사용자의 요청에 따라 실행 흐름이 달라지는 경우
- 긴 작업을 여러 단계로 나누어 처리해야 하는 경우
- 중간 결과를 보고 다음 행동을 결정해야 하는 경우

반대로 실행 순서가 항상 정해져 있다면 Agent보다 일반적인 Pipeline 구조가 더 단순하고 안정적일 수 있다.

## 13. Agent와 Chain의 차이

| 구분 | Chain | Agent |
|---|---|---|
| 실행 흐름 | 미리 정해진 순서대로 실행 | 모델이 다음 행동을 판단 |
| 도구 사용 | 정해진 도구를 순서대로 사용 | 필요한 도구를 선택적으로 호출 |
| 유연성 | 낮음 | 높음 |
| 예측 가능성 | 높음 | 상대적으로 낮음 |
| 적합한 작업 | 고정된 파이프라인 | 동적으로 변하는 복잡한 작업 |

Chain은 정해진 흐름을 안정적으로 실행하는 데 적합하고, Agent는 상황에 따라 스스로 판단하며 도구를 사용하는 작업에 적합하다.

## 14. 정리

LangChain Agent는 모델이 단순히 텍스트를 생성하는 것을 넘어, 필요한 도구를 선택하고 반복적으로 호출하면서 주어진 작업을 해결하는 구조이다.

핵심은 `Model + Harness`이다. 모델은 판단을 담당하고, Harness는 모델이 올바른 맥락과 도구를 사용해 작업할 수 있도록 실행 환경을 제공한다.


따라서 Agent는 검색 기반 챗봇, 코드 실행 도우미, 데이터 분석 자동화, 고객 상담 자동화, 복잡한 업무 자동화 시스템처럼 여러 도구를 조합하고 상황에 따라 실행 흐름이 달라지는 서비스에 적합하다.
