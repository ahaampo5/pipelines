accelerate config
# Machine: ["This Machine", "AWS"]
# Hardware setting: ["No disributed", "multi-CPU", "multi-XPU", "multi-HPU", "multi-GPU", "multi-NPU", "multi-MLU", "multi-SDAA", "multi-MUSA", "TPU"]
# Num Nodes: ["1", "2", "4", "8", "16", "32"]
# Rank of this node: ["0", "1", "2", "3", "4", "5", "6", "7"]
# Host IP: 192.168.100.12 - {hostname -I}
# Port of main process: 29500
# Same Local Network: ["YES", "no"]
# distributed operations: ["yes", "NO"]
# optimize with dynamo: ["yes", "NO"] - 사용자가 작성한 PyTorch 코드를 FX 그래프로 자동 전환 후 컴파일, 최적화 파이프라인(AOTAutograd, Inductor 등)에 전달
#    model = model.compile(model) > CNN/RNN Transformer 는 mode="default" > 동적 분기가 많으면 mode="reduce-overhead"
# use deepspeed: ["yes", "NO"] - ZeRO 기법으로 분산, CPU/GPU/NVMe offload / 설정 복잡도, Offload Bottleneck 가능, ZeRO Sync 비용
#    if deepspeed -> 
#         generate json file? where? - /root/.cache/huggingface/accelerate/default_config.yaml
#         deepspeed.zero.Init when using zero-3? - 전체 모델이 메모리에 로드된 뒤에 분산이 아닌 초기화 시점부터 분산
#         train MoE? ["yes", "NO"] 
#         if MoE == "yes" ->
#              MoE class name: [`MixtralSparseMoeBlock`, `Qwen2MoeSparseMoeBlock`, `JetMoEAttention,JetMoEBlock`]
#              Type of Launcher: ["pdsh", "standard", "openmpi", "mvapich", "mpich", "nossh", "slurm"] - 
# use FullySharedDataParallel: ["yes", "NO"] - PyTorch 1.12+ 에 내장된 분산 학습 API / Tensor Parallel 미지원, ZeRO 3 대비 메모리 절감 함계, 미성숙 offload 기능
# use Megatron-LM: ["yes", "NO"] - NVIDIA에서 개발 / Accelerate와 통합 제한, 유연성 부족, 높은 진입장벽
# num GPUs: ["1", "2", "4", "8", "16", "32"]

# use FSDP? ["yes", "NO"] 
# what should be your FSDP version? ["1", "2"]
# Do you want to enable resharding after forward? ["YES", "no"] # 메모리 여유가 충분한 경우(속도가 더 중요한 경우) NO, 메모리가 중요한 경우 YES
