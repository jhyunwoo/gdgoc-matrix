# Matrix Multiplication Participant Guide (No Accelerate)

이 문서는 참가자가 `matmul` 함수를 작성할 때 고려해야 할 핵심 사항과 공식 벤치마크 실행 환경을 정리한 가이드입니다.

## 1. 제출 함수 규격

공식 제출 함수 시그니처:

```c
void matmul(const float* A, const float* B, float* C, int M, int N, int K);
```

입출력 메모리 포맷:
- `A`: `M x K`, row-major
- `B`: `K x N`, row-major
- `C`: `M x N`, row-major
- 의미: `C = A x B`
- `C`는 overwrite 해야 하며, 기존 값을 누적하면 안 됩니다.

## 2. 함수 구현 시 필수 고려사항

### 2.1 금지 API 확인
- `<Accelerate/Accelerate.h>` 및 CBLAS/BLAS/LAPACK 계열 호출은 금지입니다.
- 실수로 링크되지 않게 빌드 스크립트와 include를 반드시 점검하세요.

### 2.2 정확도
- 검증은 참조 구현 대비 최대 상대오차로 판정합니다.

```text
max_relative_error = max(|C_test - C_ref| / max(|C_ref|, 1e-6))
```

- 통과 기준: `max_relative_error <= 1e-4`
- 벡터화/루프 재배치 후에는 반드시 `--validate`로 확인하세요.

### 2.3 메모리 접근
- row-major 연속 접근 패턴을 유지해야 캐시 효율이 올라갑니다.
- 루프 순서와 타일링(BM/BN/BK)을 함께 튜닝하세요.
- 호출마다 불필요한 할당/복사는 피하세요.

### 2.4 벡터화
- Apple Silicon에서는 NEON intrinsic(`<arm_neon.h>`)이 성능에 직접적인 영향을 줍니다.
- scalar tail 처리 분기까지 포함해 구현해야 정확/성능이 안정적입니다.

### 2.5 멀티스레딩
- `pthread` 기반 분할(예: row block 분할)이 일반적으로 구현이 단순하고 효과적입니다.
- 너무 작은 문제 크기에서는 스레드 생성 오버헤드가 커질 수 있으니 단일 스레드 fallback을 두세요.
- 벤치마크의 `--threads N`은 `MATRIX_THREADS`/`OMP_NUM_THREADS`를 설정합니다.

### 2.6 측정 안정성
- 워밍업 후 반복 측정 평균으로 비교하세요.
- 발열/백그라운드 작업/전원 모드가 점수 변동을 유발합니다.

## 3. 공식 벤치마크 실행 환경 정보

현재 제공된 머신 기준:

### 3.1 시스템
- OS: macOS 26.3 (Build 25D125)
- CPU: Apple M3 Pro
- 논리 코어 수: 11
- 메모리: 19,327,352,832 bytes (약 18 GB)

### 3.2 컴파일러
- Apple clang version 17.0.0 (`clang-1700.6.4.2`)
- Target: `arm64-apple-darwin25.3.0`

### 3.3 빌드 설정 (현재 저장소 기본)
- 표준/경고: `-std=c11 -Wall -Wextra -Wpedantic`
- 최적화: `-O3 -DNDEBUG -funroll-loops -mcpu=native`
- 링크: `-lm`

### 3.4 벤치마크 기본 규칙
- 기본 크기: `512,1024,2048`
- 기본 반복: `warmup=3`, `repeat=10`
- 점수: 크기별 평균 GFLOPS의 산술 평균
- 정확도 임계값: `1e-4`

## 4. 참가자가 바로 쓰는 점검 절차

1. 단일 제출물 기능/정확도 확인
```bash
make SUBMISSION=path/to/your_submission.c
./benchmark --sizes 128,256 --repeat 2 --warmup 1 --validate
```

2. 공식 크기 성능 확인
```bash
./benchmark
```

3. submissions 폴더 전체 자동 벤치마크
```bash
make bench-all
```

4. 전체 테스트 시 스레드 고정
```bash
make bench-all BENCHMARK_ALL_ARGS="--validate --threads 11"
```

## 5. 성능 개선 체크리스트

- 루프 순서가 row-major 연속 접근을 보장하는가?
- NEON 벡터 구간과 tail 구간이 모두 최적화되어 있는가?
- 타일링/블로킹 파라미터를 측정 기반으로 튜닝했는가?
- 스레드 분할이 균등하고 race가 없는가?
- `--validate`에서 정확도 통과를 유지하는가?
