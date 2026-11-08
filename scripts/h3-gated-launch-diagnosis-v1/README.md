# 门控残差主机提交与单次设备图执行诊断 v3

用途：调查 2026-10-28 首轮性能回传中融合路径慢于原生 mul→add 的原因，使用已通过正确性回归的 rounding-v3 资产。本修订仅测单次运算图 K=1。新包的完整 DCU 结果仍待服务器运行，本地检查不能替代设备验证。

## 本次调整的依据

用户粘贴的 `20261028_102452_2a69c989` 日志显示：节点类型交叉校准和 FP32 7×64 的 K=1 阶段已返回，K=16 baseline 捕获却仍只暴露两个类型为200的节点和一条边。该轮 ZIP/原始计时数据尚未回传，本次不据此填写耗时或收益。

这否定了旧脚本“每次提交都对应新增可枚举节点”的假设，但不能证明设备只执行了一次；节点聚合与捕获不完整仍未区分。新包在进入 GPU 调用前拒绝 K>1，计时函数也拒绝按未经证明的次数作除法。K=16/64仍是未验证项，不通过放宽节点数检查将其记为成功。

## 运行

上传 `h3-gated-launch-diagnosis-v3-single-op.tar.gz` 到 `/root/private_data/scripts/`。兼容之前的服务器路径，解压后仍使用 `h3-gated-launch-diagnosis-v1/` 目录。在已分配的单卡 Pod 运行：

```bash
source /root/private_data/setup_env.sh
tar -xzf "$WORK_HOME/scripts/h3-gated-launch-diagnosis-v3-single-op.tar.gz" -C "$WORK_HOME/scripts" &&
run_logged h3_gated_launch_single_op_quick.log bash "$WORK_HOME/scripts/h3-gated-launch-diagnosis-v1/run.sh" --quick
```

`--quick` 测三种dtype的7×64用例，预热2次、采样6轮。只有出现 `SINGLE-OP SMOKE PASSED` 才运行完整的三精度×三尺寸矩阵：

```bash
run_logged h3_gated_launch_single_op_full.log bash "$WORK_HOME/scripts/h3-gated-launch-diagnosis-v1/run.sh"
```

完整矩阵每条路径预热10次、采样40轮，成功消息为 `SINGLE-OP DIAGNOSIS COMPLETE`。两次都保留 `EVIDENCE ZIP:` 指向的 ZIP 和对应的 `$WORK_HOME/logs/*.log`。小尺寸通过不计为全尺寸性能证据。`summary.json` 的 `measurement_scope=single_operation_graph`，`full_matrix_measured` 仅在完整矩阵成功时为真，`batch_graph_measured` 始终为假。

新包不安装依赖、不编译内核或框架、不修改 setup_env.sh、源代码、已安装库或活动资产路径。沿用 `logs/h3_gated_active_bundle.txt` 指向的 v3。三项已加载原生库哈希必须与上轮实测一致。所有临时实验、日志、NPZ、JSON、CSV 和 ZIP 均在 `$WORK_HOME/logs/h3_gated_launch_<时间>_<编号>/`。

## 测量边界

固定三种 dtype：FP32/FP16/BF16；固定 6H gate 切片布局，三个合成形状：7×64、257×513、4096×2048。种子与首轮相应输入一致；不是正式模型尺寸。全部预分配输出，固定热输入，不测分配开销或完整模型。

每条主机路径预热10次，再记录40轮：

- `baseline`：公开 InfiniCore mul→add。
- `fused`：公开 InfiniCore gated_residual。
- `direct`：相同已验收宿主 DSO 与 HSACO，模块和参数在计时前准备，通过 ctypes 调用宿主启动器。它是诊断对照，不接入正式 H3，也不计作新的核心算子。

每轮初始流同步在计时外。记录 `submit_ns`（调用前到返回）、`remaining_wait_ns`（返回到流同步完成）、`total_ns`，原始整数严格满足前两项之和等于第三项。提交时可能已经执行设备工作，也可能有运行库阻塞；等待包含同步开销，**不是设备总计算耗时**。direct 与公开接口的 Python 边界也不同，不能用两个中位数直接相减断言某个 C++ 检查占用多少时间。

另外对 baseline/fused 各捕获只含一次门控残差运算的原始 HIP 图：`start event → 单次计算 → end event`。baseline 包含mul→add，fused包含门控残差；计时事件显式放在图内，每次回放只提交一次图。保存原始 `event_ms`，仅作毫秒到微秒换算，保留字段名 `device_graph_us_per_op`；`operations=1`。每个图同样预热10次、40轮 AB/BA 交替测量，不筛除慢样本。

图回放避免重新通过Python逐个提交算子；事件间隔仍包含图调度、节点间开销及事件自身开销，**不是独立 kernel 指令执行时间**。K=1没有批量摊销，特别小的张量可能主要测到调度和事件成本，不作纯内核结论。不要计算“CPU耗时=总耗时−GPU耗时”，二者可能重叠。若需要纯内核时间线，后续根据 `profilers` 清单选择当前 DTK 实际可用的 profiler；本包仅盘点工具。

## 捕获与正确性门禁

- 从 InfiniCore 的当前显式流取得 opaque pointer；HIP 库必须与实际已加载的库一致，避免再次猜测设备属性结构 ABI。
- 先显式创建两个已知事件节点，仅校准事件节点枚举（已知兼容布局为 event=7 或 event=8），检查真实节点和线性依赖；不再用事件枚举推导计算节点枚举。
- 再分别捕获一次 baseline 与 fused：baseline 必须是两个同类型串行计算节点，fused 必须是一个相同类型的计算节点。2026-10-28 的海光 DTK 探针观察到该类型为不透明值 `200`；只有两条受控路径相互印证时才接受，不能把 `200` 写死或推广到其他运行库。
- 普通 `hipEventRecord` 在部分运行库捕获期间只建立依赖，并不生成实际计时节点。本包捕获内核后，通过 `hipGraphAddEventRecordNode` 与依赖 API 显式包围计算，不依赖这种版本相关的默认行为。
- 单次baseline图要求2个已校准类型的节点，fused要求1个，另各有2个计时事件节点。不同类型、额外节点、类型变化或非串行依赖均拒绝。类型200是不透明节点类型，不能据此将节点数等同于设备kernel数；K>1已禁用。
- 每条路径计时前后将输出置NaN；baseline同时将乘法中间结果置NaN，再回放并同步，防止预热残留的中间值掩盖漏执行。对完整输出进行CPU参考逐位检查、有限性与输入未修改检查，保存全量结果哈希和确定性抽样数组。
- 原生计数器表示宿主提交次数：普通fused调用增加1，单次捕获增加1；回放不增加。回放的计算证据来自单次捕获、图结构与污染后的检查，不将宿主计数当作设备执行计数。
- 模块、输入输出、临时乘积和事件覆盖整个图生命周期，销毁前同步。捕获或校验失败会停止并归档已有证据，进程崩溃/超时也由外层收集日志；不会回退到普通事件计时后仍宣称已分离。
- 这是显式诊断实验。既有 `is_device_graph_capture_safe=false` 不变，不宣称已支持正式模型的设备图、动态形状或图更新。

## 读结果

`summary.json` 汇总，`results.csv` 分路径/阶段/指标列出P50/P95；逐项 `host.json` 与 `graph_1.json` 保留全部原始样本。`worker.log` 实时输出并收进ZIP。`batch_capture_status`记录此前失败与当前限制。

若fused公开接口提交明显高于direct，提示检查框架包装、计划创建、文件路径与锁等宿主成本，但还不能只归因于某一函数。图内间隔用于比较同一调度方式下的单次运算；即使它较快，也不能覆盖实际公开接口总耗时较慢的结论。收到证据后按原始数据判断。

## 参考及兼容范围

HIP capture、instantiate、replay与event elapsed接口依据[HIP图API](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.2/doxygen/html/group___graph.html)和[事件API](https://rocm.docs.amd.com/projects/HIP/en/docs-6.1.2/doxygen/html/group___event.html)。[HIP 6.0通用节点枚举](https://rocm.docs.amd.com/projects/HIP/en/docs-6.0.0/doxygen/html/group___global_defs.html)没有此次观察到的200，不能将通用枚举直接当作DTK语义或兼容保证；当前以服务器证据约束测量范围。
