# 九齿门控残差单卡性能补测 v1

只测已验收的 rounding-v3，不构建、不安装、不修改框架，不切换 active bundle。
要求已完成 v3 六阶段回归，`$WORK_HOME/logs/h3_gated_active_bundle.txt` 指向成功资产。
在正常初始化的 Python 3.11 下运行；不要进入九齿编译 venv。

## 上传与执行

上传 `h3-gated-benchmark-v1.tar.gz` 到服务器 `$WORK_HOME/scripts/`。

```bash
source /root/private_data/setup_env.sh
tar -xzf "$WORK_HOME/scripts/h3-gated-benchmark-v1.tar.gz" -C "$WORK_HOME/scripts"
run_logged h3_gated_benchmark.log bash "$WORK_HOME/scripts/h3-gated-benchmark-v1/run.sh"
```

运行期间保持该卡空闲，不同时运行其他测试、模型推理或构建。脚本不修改频率、功耗或其他进程。
若想先快速排查环境，可在上述 run.sh 后添加 `--quick`；快速模式只作冒烟，不能填成完整性能实验。
通常不需重新运行 v3 导出或构建。环境校验失败时保留日志，先核对错误，不重新安装全部依赖。

## 测量内容

- 3种dtype：FP32、FP16、BF16。
- 每种dtype的8个合成尺寸/布局：7×64连续与6H；257×513连续、6H、步长2、batch1的6H；4096×2048连续与6H。共24组输入。
- 每组输入两种分配口径，共48行用例：
  - `preallocated`：基线乘积/输出和融合输出均提前分配，使用双方公开接口的 `out=`；原生乘积中间缓冲区仍要读写。
  - `allocating`：`ic.add(r, ic.mul(g,b))` 对比 `ic.gated_residual(r,g,b)`，两边都由接口自行分配输出。它复现H3计算表达式，但包括Python接口成本，不等于模型C++前向耗时。
- 默认每条路径预热25次；60个采样轮，基线/融合按AB/BA交替，每组起始次序轮换；输入固定且不会原地累加。所有样本保留，不截断慢样本。
- `device`：InfiniCore当前流上DeviceEvent记录50次调用的区间，除以50，单位微秒。P50/P95是60个**批均值**的分位数，不是单kernel或单请求P95；主机提交间隙可能计入，不声称纯GPU内核耗时。
- `host`：独立一次接口调用并同步完成的耗时，前置同步在计时外，末尾同步在计时内。P50/P95是60个**独立单次同步接口**样本的分位数。
- 输入产生、H2D、Python张量包装、参考计算、结果读取、事件创建、校验和日志不在计时区间。双方首次调用和模块加载通过正确性检查/预热提前完成，不计稳态。
- 同一输入重复运行，保留热缓存；不清空缓存、不使用GPU graph。峰值有效带宽、官方模型加速比、8卡加速比均不在本次测量范围。
- 加速比=基线P50/融合P50；低于1x时照实报告。正确性通过但性能变慢仍属于一次有效测量，不能为了得到加速比改容差、换参考或筛选结果。

## 正确性与证据

每条用例计时前后全量检查：基线/融合/独立CPU参考逐位一致（包括符号零）、有限值、完整输入底层存储不变。每批核对真实原生九齿调用数：基线0次、融合每次调用1次。CPU Torch/NumPy仅用于创建测试数据、存储、传输和独立对照；两条参赛计算路径都通过InfiniCore接口运行。

检查 bundle 所有资产哈希、v3九齿源身份、框架补丁、当前HYGON/gfx906设备及实际已加载库。运行结束重新检查内核资产未变。保存Git版本/差异、xmake配置、Python/运行库、设备、可见卡数及计时协议。

输出统一在 `$WORK_HOME/logs/h3_gated_benchmark_<时间>_<编号>/`：

- `summary.json`：环境、全部用例、全部原始样本、比较结果和失败状态。
- `results.csv`、`report.md`：可以填入报告的统计表。两种分配口径和计时口径分列。
- 每个用例的 `result.json`、`before/after_correctness.json` 及 `*_sample.npz`。
- NPZ只保存固定抽样（完整小样本或最多约4160位置），**全量验证不依赖抽样**，完整输入底层存储、输出和参考的哈希在JSON中。大数组未完整存档，需按固定种子和版本重建。
- `run.json`、`files.json`：子进程退出状态、脚本和证据哈希。
- 相邻同名 `.zip`：自动归档。异常/超时也尽量保留已有证据；默认超时3600秒。

运行成功必须同时满足 `BENCHMARK PASSED`、外层exit code 0、`run.json/summary.json`的passed=true、48行完成、每行前后正确性与调用数检查通过。`--quick`是12行冒烟，不能混为48行完整结果。

请回传 `h3_gated_benchmark.log` 与本轮自动生成ZIP。建议截取开始环境/协议信息和最后性能汇总/成功行；详细数字仍以CSV/JSON为准。
收到真实DCU数据之前，技术报告中的性能栏继续标注待测。单卡合成算子实验不替代正式模型、完整视频或初赛8卡适配。
