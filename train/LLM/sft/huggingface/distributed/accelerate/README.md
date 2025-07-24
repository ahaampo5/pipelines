## Deepspeed vs. FSDP vs. Megatron-LM
1. 분산·병렬화 가능한 “대상” 6가지
Data (데이터 병렬)

Optimizer state (옵티마이저 모멘텀·스케줄러 상태)

Gradient (그래디언트)

Module / Parameters (모델 파라미터)

Tensor (텐서 차원 단위 분할 → tensor parallelism)

Pipeline (레이어·블록 단위 분할 → pipeline parallelism)

| 축            | DeepSpeed (ZeRO)               | FSDP                     | Megatron-LM            |
| ------------ | ------------------------------ | ------------------------ | ---------------------- |
| Data         | ✅                              | ✅                        | ✅                      |
| Optimizer    | ✅ (ZeRO‐2/3 sharding)          | ✅ (state sharding)       | ✅ (optimizer sharding) |
| Gradient     | ✅ (ZeRO‐1/2/3)                 | ✅ (grad sharding)        | ✅                      |
| Module/Param | ✅ (ZeRO‐3)                     | ✅ (full\_param sharding) | ✅                      |
| Tensor       | ❌ (별도 구성 필요)                   | ❌ (별도 구성 필요)             | ✅ (native 지원)          |
| Pipeline     | ✅\* (DeepSpeed PipelineEngine) | ❌ (별도 구현 필요)             | ✅ (native 지원)          |
| KV Cache⁺    | ✅\* (DeepSpeed Inference)      | ❌                        | ✅ (decoder KV 분산)      |

⁺KV Cache 분산은 주로 추론(inference) 단계의 디코더 캐시(키·값 텐서)를 말합니다.

## Type of Launcher
| GPU 통신에 사용되는 도구
* pdsh(경량 분산 쉘 도구) - 소규모 테스트용(ssh로 간단히 실행)
* openmpi(오픈소스 MPI 구현체) - 노드 수가 많아도 빠르게 배포
* mvapich(InfiniBand,RoCE 등 네트워크 고성능 MPI 구현체) - HPC/고성능 네트워크 (InfiniBand 가속)
* mpich(오리지널 MPI 구현체) - 범용 MPI 환경 (경량, 견고)
* nossh(SSH 없이 로컬 머신에서 실행) - 보안 정책상 ssh 불가시
* slurm(클러스터 관리 및 작업 스케줄링 시스템) - 대규모 자원 관리, 예약