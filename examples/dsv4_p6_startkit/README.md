# P6 T01 start kit：可离线运行的开发稿

状态：`p6-t01-recorded.draft.v1`。供 NVIDIA H100 团队测试、审阅和继续开发。

交付 CPU/FP32 oracle、固定 golden inputs/intermediates、saved-forward 合同、mock return / gradient dispatch、recorded provider stub、独立测试与可选 GPU reference smoke。
本包没有注册到 production 路径，也不是原 member 未提供的 T01 PR 的副本。

## 1. 范围与当前限制

- P5 已加权，P4 逐 slot return，P6 slot 0→5 本地 FP32 累加。
- 先加 shared，再加 residual，P6 最后一次 BF16 RNE 转换。
- backward 复用 forward metadata，归并属于同一 `gradient_boundary` 的 routed/shared dX。
- 原始 residual 梯度留给 P1 的正确 fork join；P3 gate 梯度也须按真实计算图接线。
- CPU 命令只需 Python >=3.10，不导入 Torch、不编译扩展、不下载模型。
- GPU smoke 需要现有 PyTorch/NumPy/CUDA 环境，一张 H100 即可。

**尚未冻结的部分不会宣称完成。** 没有获取到可验证的 Foundation 具体 ABI 实现。
`Plan` 仅为本包 recorded-fixture DTO，通过 opaque schema/fingerprint 引用 producer。
`foundation_compatibility=UNVERIFIED`、`production_certified=false` 始终保留。
正式 P6-S0 需要维护者接到真实 Foundation contract tests，并审定下表中的 draft 选择。

## 2. 获取远程开发分支

本 PR 基于 `dsv4-p6-dev`，分支名为 `p6-t01-start-kit`，只新增独立 start kit。在 RL-Kernel checkout 中：

```bash
git fetch origin p6-t01-start-kit
git switch --track origin/p6-t01-start-kit
python examples/dsv4_p6_startkit/run.py selftest
```

如果已有同名本地分支，直接 `git switch p6-t01-start-kit`，再检查是否需要更新。
离线交付的 patch/zip 仍可用于其他 checkout；已有同名目录时先检查冲突，勿强制覆盖。

## 3. CPU 独立验收

在仓库根目录或代码包根目录执行：

```bash
python examples/dsv4_p6_startkit/run.py contract
python examples/dsv4_p6_startkit/run.py selftest
python examples/dsv4_p6_startkit/run.py conformance \
  --device cpu --hidden-size 4096 --output /tmp/p6-t01-cpu-attempt-01
python examples/dsv4_p6_startkit/run.py verify /tmp/p6-t01-cpu-attempt-01
```

每次使用新的 output 目录，已有目录拒绝覆盖。`selftest` 通过 unittest 运行，不经过仓库 pytest 的全局 conftest。
CPU PASS 只表示本地标准答案与合同测试通过，不能当作 GPU 通过。

## 4. H100 测试

进入已配置 PyTorch/CUDA 的开发环境：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
python examples/dsv4_p6_startkit/run.py conformance \
  --device cuda:0 --require-h100 --hidden-size 4096 \
  --output /tmp/p6-t01-h100-eager-attempt-01
python examples/dsv4_p6_startkit/run.py conformance \
  --device cuda:0 --require-h100 --graph --hidden-size 4096 \
  --output /tmp/p6-t01-h100-graph-attempt-01
python examples/dsv4_p6_startkit/run.py verify /tmp/p6-t01-h100-graph-attempt-01
```

检查 PyTorch reference 的 forward、逐 slot intermediate、BF16 output、backward FP32 fan-in、P4-style dy gather，与独立 scalar CPU oracle 逐字节对照。
覆盖 contiguous / non-contiguous；Graph 测试含重复重放与固定地址输入修改。
没有 H100、CUDA 或 PyTorch 时命令失败退出，不回退到 CPU PASS。

这不是 T05/T06 Triton kernel 认证，也不是 EP/TP/DP 或 NCCL 通信验收。
本环境没有运行 GPU 分支。需要逐卡检查时，在资源空闲时执行：

```bash
for gpu in 0 1 2 3 4 5 6 7; do
  python examples/dsv4_p6_startkit/run.py conformance \
    --device "cuda:$gpu" --require-h100 --graph --hidden-size 4096 \
    --output "/tmp/p6-t01-h100-gpu-${gpu}-attempt-01" || break
done
```

每张卡分别跑本地 reference，卡之间不交换数据。

## 5. 文件与数学合同

| 文件 | 用途 |
| --- | --- |
| `contract.json`、`p6_startkit/contract.py` | draft policy、opaque producer refs、logical token/slot、saved-forward fingerprint |
| `p6_startkit/oracle.py` | 每次加法显式 FP32 舍入、BF16 ties-to-even、正反向 CPU oracle |
| `fixtures/golden.json` | 9 个 seeded/hand-crafted case、完整逐阶段 bytes、saved forward |
| `p6_startkit/fixtures.py` | fixture generator、input-bound recorded stub |
| `p6_startkit/torch_reference.py` | 可选 PyTorch CPU/CUDA reference 与 CUDA Graph smoke |
| `tests/test_startkit.py` | 正负测试、算术反例、metadata、artifact 完整性 |
| `run.py` | 独立 CLI、conformance、seal、离线 verify |

Golden 覆盖非连续 global token IDs、乱序 packed rows、invalid/overflow 排除、padding、empty batch、zero routes、奇数 H、抵消/量级差、BF16 tie、signed zero。
`slot_partials_fp32` 保存每个 slot 后的 accumulator。bytes 为 little-endian hex，dtype/shape 由 plan/boundary 名定义。
日常验收只读取 committed goldens，不自动刷新。合同更改必须升 schema 并审阅差异，再由维护者显式 `freeze-goldens --output 新文件`。

| 项目 | 需 owner 审定的 draft 选择 |
| --- | --- |
| forward row/shared/residual | 精确 BF16 值，读入后 widen 到 FP32 |
| 有效 slot | 0→5；第一份有效 slot 直接复制，后续逐次 FP32 相加 |
| invalid | 不加载、不加入；all-invalid routed accumulator 为正零 |
| final merge | `(routed + shared) + residual`，然后 BF16 RNE |
| backward | FP32 row/shared 输入和 FP32 输出；待对齐 P4/P5 最终 payload ABI |
| gradient boundary | 必须显式同名；本包使用 synthetic expert-input boundary |
| active nonfinite/overflow | fail-closed；padding payload 不进树 |
| subnormal | 明确 unsupported；未认证 CPU/GPU FTZ 一致性 |
| scope | recorded/reference；不含生产 core、真实跨模块接线、通信 |

SavedForward 保存整个 immutable 本地 plan，含 run/microbatch/forward/checkpoint、producer refs、mask、inverse map、policy hash。
反向必须提交受信的原 fingerprint 和 context。checksum 校验一致性/完整性，不是签名。

## 6. T02–T09 独立开发入口

| 任务 | 直接消费 T01 golden |
| --- | --- |
| T02 | plan/inverse map、canonical rows |
| T03 | 六 slot rows、每步 FP32 partial、抵消反例 |
| T04 | routed/after-shared/precast/BF16 output、分支 boundary |
| T05 | recorded 输入、完整 forward golden；实现生产融合 core |
| T06 | saved forward、dx_rows/dx_shared、backward intermediate/output |
| T07 | immutable snapshot、input-bound stub、unsupported production gate |
| T08 | 负向测试、recorded cases |
| T09 | local trace、sealed artifact、CPU/GPU reference 结果 |

P6-R0 仍需要 Foundation compatibility、T05/T06 kernel、T07 adapter 和目标 GPU 认证。
本包不导入 P3/P4/P5/P7 live 实现。

## 7. 发回结果

发回 `selftest` 摘要和 H100 output 中的 `report.json`、`completion.json`。
失败时保留完整错误以及首个 case/boundary。`verify` 只在 CPU 重放并检查 seal，不表示重新执行 GPU。
