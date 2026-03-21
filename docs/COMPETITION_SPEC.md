# Matrix Multiplication Competition Spec (No Accelerate)

## 1. 목적
대회 참가자는 동일한 인터페이스의 `matmul` 함수를 제출하고, 벤치마크 프로그램으로 성능(GFLOPS)과 정확도(상대오차)를 평가받습니다.

## 2. 타깃 환경
- OS: macOS (Apple Silicon)
- 기준 하드웨어: MacBook Pro M3 Pro
- 컴파일러: `clang`
- 데이터 타입: `float` (FP32)

## 3. 제출 함수 인터페이스 (고정)
```c
void matmul(const float* A, const float* B, float* C, int M, int N, int K);
```

## 4. 입력/출력 포맷 (고정)
- `A`: `M x K` 행렬, row-major, 연속 메모리
- `B`: `K x N` 행렬, row-major, 연속 메모리
- `C`: `M x N` 행렬, row-major, 연속 메모리
- 연산 정의: `C = A x B`
- `C`는 항상 overwrite 해야 하며, 기존 값에 누적하면 안 됩니다.

인덱스 예시:
- `A[i][k]` -> `A[i*K + k]`
- `B[k][j]` -> `B[k*N + j]`
- `C[i][j]` -> `C[i*N + j]`

## 5. 라이브러리 제한 사항
### 허용
- C 표준 라이브러리
- POSIX 스레드(`pthread`)
- ARM NEON intrinsic (`<arm_neon.h>`)

### 금지
- Apple Accelerate/vecLib (`<Accelerate/Accelerate.h>`, CBLAS/LAPACK 포함)
- 외부 BLAS/수치 라이브러리 (OpenBLAS, MKL, BLIS 등)
- GPU API 사용 (Metal/CUDA/OpenCL)
- 네트워크 I/O 또는 외부 프로세스 의존 실행

## 6. 채점/검증 규칙
### 기본 성능 케이스
- 고정 크기: `512, 1024, 2048` (정방 행렬: `M=N=K=size`)
- 기본 반복: `warmup=3`, `repeat=10`

### 성능 지표
- `GFLOPS = (2*M*N*K) / elapsed_seconds / 1e9`
- 각 크기의 평균 시간 기반 GFLOPS를 계산
- 최종 점수: 크기별 평균 GFLOPS의 산술 평균

### 정확도 기준
- 참조 구현(`matmul_ref`) 대비 최대 상대오차:

```text
max_relative_error = max(|C_test - C_ref| / max(|C_ref|, 1e-6))
```

- 통과 기준: `max_relative_error <= 1e-4`

## 7. 빌드/실행
### 빌드
```bash
make
```

### 제출 함수 벤치마크
```bash
./benchmark
```

### 정확도 검증 포함
```bash
./benchmark --validate
```

### 스레드 수 지정 예시
```bash
./benchmark --threads 8
```

### submissions 폴더 전체 제출물 자동 벤치마크
```bash
make bench-all
```
- 기본값: `--validate --sizes 128,256 --repeat 2 --warmup 1`
- 사용자 지정 예시:
```bash
make bench-all BENCHMARK_ALL_ARGS="--validate --threads 11"
```

### 제출 파일 목록 확인
```bash
make list-submissions
```

## 8. 참가자 제출 방법
1. `submissions/*.c` 파일에 `matmul` 함수를 구현
2. 단일 파일 테스트 시:

```bash
make SUBMISSION=path/to/your_submission.c
./benchmark --validate
```

## 9. 출력 형식
결과는 사람 친화 텍스트로 출력됩니다.
- 실행 환경 요약
- 크기별 `avg_ms`, `best_ms`, `avg_GFLOPS`, `best_GFLOPS`
- 검증 시 `PASS/FAIL`, `max_rel_err`
- 최종 점수(평균 GFLOPS)

## 10. 실격 조건
- 함수 시그니처 불일치
- out-of-bounds 메모리 접근/크래시
- 금지 라이브러리 사용 (Accelerate/BLAS 계열)
- NaN/Inf를 유발하는 비정상 결과 다발
- 정확도 기준 미달
