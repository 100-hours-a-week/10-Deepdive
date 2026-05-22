>이터레이터와 제너레이터가 메모리 공간 효율성을 개선하는 방식을 설명하고, 대규모 데이터 처리(예: 로그 파일 분석)에서 이를 활용하는 구체적인 시나리오를 제시하시오.


## 이터레이터란?

이터레이터는 next() 함수 호출 시 계속 그 다음 값을 반환하는 객체이다.

더이상 출력할 값이 없다면 StopIteration 예외를 발생시킨다.

```python
a = [1,2,3]

print(type(a)) # <class 'list'>
a = iter(a)
print(type(a)) # <class 'list_iterator'>

print(a.__next__()) # 1
print(a.__next__()) # 2
print(a.__next__()) # 3
print(a.__next__()) # StopIteration Exception
```

## 제너레이터란?

제너레이터는 yield를 사용하여 데이터를 생성하는 함수이다.

yield는 함수의 실행을 중단하고 해당 데이터를 반환한다.

다시 호출된다면 처음부터 실행되지 않고 중단된 지점에서부터 다시 실행된다.

더이상 출력할 값이 없다면 이터레이터와 동일하게 StopIteration 예외를 발생시킨다.

```python
def generator():
    yield 1
    yield 2
    yield 3

# 제너레이터 객체 생성
a = generator()
print(type(a))  # <class 'generator'>

print(a.__next__())  # 1
print(a.__next__())  # 2
print(a.__next__())  # 3
print(a.__next__())  # StopIteration Exception
```

## 메모리 공간 효율성을 개선하는 방식: 지연 평가

1. 조급한 평가 (Eager Evaluation - 리스트 방식)
    
    데이터가 1000개이던 1억개이던 당장 메모리(RAM)에 전부 올려 버리는 방식.
    
    데이터가 너무 크다면 시작하기전에 동작을 멈춘다.
    
2. 지연 평가 (Lazy Evaluation - 이터레이터, 제너레이터 방식)
    
    데이터를 전부 메모리(RAM)에 올리지 않고 next()가 호출될 때 필요한 데이터만 메모리에 올린다.
    
    값을 바깥으로 전달하고 나면 그 데이터는 메모리에서 지워진다.
    

## 대규모 데이터 처리 예시

```python
import os

# 제너레이터: 100GB 파일에서 한 줄씩만 꺼내는 함수
def read_log_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line  # 파일을 한 줄씩만 읽어 메모리에 올리고 바깥으로 던짐

# 제너레이터: 받은 줄에서 "[ERROR]"이 포함된 line만 바깥으로 던짐
def error_logs(log_lines):
    for line in log_lines:
        if "[ERROR]" in line:
            yield line  # 필터링된 한 줄만 바깥으로 던짐
            
----------------------------------------------------
# 실행
log_path = "server.log"

# 제너레이터 객체만을 생성 (현 시점에서는 메모리 사용x)
all_lines = read_log_file(log_path)
error_lines = error_logs(all_lines)

# error_logs가 호출될 때마다 read_log_file에서 한 줄씩 데이터를 가져와서,
# 해당 줄에 [ERROR]이 포함 되었을 경우에만 바깥으로 전달
for log in error_lines:
    # 원하는 로직 처리 
    # ex)출력 및 저장
    print(log.strip())
```

코드 작동 흐름

1. all_lines = read_log_file(log_path)을 실행할 때. 100GB 파일을 전부 읽지 않고 함수 객체만을 생성
2. for log in error_lines: 루프가 시작되면서 error_lines에 next()가 실행된다.
3. 앞 단계인 read_log_file에게 한 줄을 요청한다.
4. read_log_file은 한 줄을 반환하고 그 자리에 정지한다.(yield)
5. [ERROR] 이 포함되어 있다면 반환되어서 print(log.strip())이 실행된다.
6. 없다면 다시 read_log_file이 한 줄 읽어서 반환한다. 이때 그전에 반환했던 줄은 메모리에서 삭제되어있다.
