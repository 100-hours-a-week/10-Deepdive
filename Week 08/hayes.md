> LangGraph의 StateGraph가 상태 기반으로 워크플로우를 구성하는 이유를 설명하고, 그래프 실행 중 상태 값이 어떻게 노드 간에 전달·갱신되는지 구체적으로 서술하시오.
> 

## LangGraph

state를 가진 LLM 애플리케이션의 워크플로와 에이전트를 그래프 구조로 설계하고 실행하는 프레임워크

단순한 LLM 호출은 입력 → 프롬프트 → 모델 → 출력 순서로 한 방향으로 끝난다.

LangGraph는 이러한 실행 흐름을 State, Node, Edge로 구성된 그래프 구조로 설계하고 실행할 수 있게 해준다.

## StateGraph

StateGraph는 상태 스키마를 기반으로 Node와 Edge를 구성하여 실행 가능한 그래프를 구축하는 클래스이다.

StateGraph(schema) → add_node() → add_edge() → compile() → invoke()

위 5단계 흐름으로 구성되고 실행된다.

- StateGraph(schema) : 상태 스키마를 인자로 받아 빌더 객체를 생성
- add_node() : 노드를 등록
- add_edge() : 노드 사이의 실행 순서를 정의
- compile() : 빌더를 실행 가능한 그래프로 변환
- invoke() : 초기 상태를 전달해 그래프를 실행

StateGraph가 상태 기반으로 워크플로우를 구성하는 이유는 여러 노드가 공유하는 상태와 실행 흐름을 구조적으로 관리해서 복잡한 LLM 워크플로를 예측 가능하고 관리하기 쉬운 그래프로 만들기 위해서이다.

각 노드는 state에서 필요한 키만 읽고 갱신할 키만 반환하기 때문에 다른 노드의 입출력 형식을 알 필요가 없다.

가장 중요한 것은 Reducer개념이다.

전체 state를 다시 만드는 것이 아니라 업데이트만 반환하고 각 reducer 규칙에 따라 그 업데이트를 병합한다.

## 예시코드

### 상태 스키마 정의

```python
from typing_extensions import TypedDict                                       # 상태 스키마 정의용 TypedDict 임포트

# === 그래프 전체에서 공유되는 상태 구조 정의 ===
class MyState(TypedDict):
    count: int                                                                # 정수 상태 키: 카운트 값을 저장
    message: str     
```

그래프에서 공유할 상태 구조를 정의

### 노드 함수 정의

```python
# === 증가 노드: count를 1 증가 ===
def increment(state: MyState) -> dict:
    return {"count": state["count"] + 1}                                      # count 키만 부분 업데이트로 반환 (message는 유지)

# === 인사 노드: 인사 메시지 생성 ===
def greet(state: MyState) -> dict:
    return {"message": f"안녕하세요! 카운트: {state['count']}"}                  # message 키만 부분 업데이트로 반환
```

작업을 수행할 노드 함수를 정의

### 그래프 구성

```python
from langgraph.graph import StateGraph, START, END                            # 그래프 빌더와 시작·종료 노드 임포트

# === StateGraph 생성 및 구성 ===
builder = StateGraph(MyState)                                                 # 상태 스키마를 기반으로 그래프 빌더 생성
builder.add_node("increment", increment)                                      # 증가 노드 등록
builder.add_node("greet", greet)                                              # 인사 노드 등록
builder.add_edge(START, "increment")                                          # 시작 지점에서 증가 노드로 연결
builder.add_edge("increment", "greet")                                        # 증가 노드에서 인사 노드로 연결
builder.add_edge("greet", END)                                                # 인사 노드에서 종료 지점으로 연결
```

그래프 빌더를 생성하고 노드들을 등록

실행순서를 START → increment → greet → END로 구성

### 컴파일 및 실행

```python
# === 컴파일 및 실행 ===
graph = builder.compile()                                                     # 빌더를 실행 가능한 그래프로 변환
result = graph.invoke({"count": 0, "message": ""})                            # 초기 상태를 전달하여 그래프 실행
print(result)
```

그래프 빌더를 실행가능한 객체로 바꾸고 초기 state를 넣어 실행한다.

위에서 정의한 실행 순서에 따라 Increment노드에서 count가 0 → 1로 변화하고 greet 노드에서 count = 1을 읽어 message를 생성한다.

이런 방식으로는 기본 동작인 덮어쓰기가 되어서 새 값만 계속 저장된다.

이럴 때 Reducer를 사용하여 규칙이나 직접 정의한 리듀서 함수 정의에 따라 State를 수정한다. 

```python
# === 커스텀 Reducer: 합산 ===
def add_count(left: int, right: int) -> int:
    return left + right

# === Reducer가 포함된 상태 스키마 정의 ===
class State(TypedDict):
    query: str
    results: Annotated[list[str], operator.add]
    count: Annotated[int, add_count]
```

- query : 새 값이 들어오면 기존 값을 덮어쓴다.
- results : operator.add 기존 리스트 뒤에 새 리스트를 이어 붙인다.
- count : add_count 기존 숫자와 새 숫자를 더한다.
