# 九齿安装 v1.2：Triton 缓存重命名兼容

本次日志确认：系统NumPy1.25.0保持原样，九齿候选使用私有NumPy1.26.4，Torch2.9.0/HIP身份保持一致。新的首个错误是海光Triton初始化HIP工具时保存缓存失败：`OSError: [Errno18] Invalid cross-device link`。

出错位置为Triton `FileCacheManager.put` 把临时子目录中的 `hip_utils...so` 重命名到缓存父目录。v1.2仅在本次工作区的海光Triton副本中改为同目录的唯一临时文件，保留原子 `os.replace`。不做非原子复制回退，不改全局os模块，也不改九齿表达或数值内核。补丁检查原始文件SHA，保存修改前后源码和哈希。

## 更新一个文件后重跑

1. Windows本地解压 `ninetoothed-gfx936-install-upload-v1.2.zip`。
2. 项目中仅需替换 **`install_ninetoothed_gfx936.py`**。
3. 其余三个运行文件已放入包内便于核对，内容与v1.1相同，可沿用。

四个依赖wheel和原海光Triton继续保留，**不需要重新上传依赖**。原失败候选目录无需删除；脚本会创建新编号目录。

启动入口仍为 **`install_ninetoothed_gfx936.sh`**，沿用 `scnet-dcu` 八卡批处理，**参数留空，28分钟**。

完成后返回 `ninetoothed-install.json`、`ninetoothed-probe.json` 和完整日志。日志目录新增 `triton-cache-patch.json` 与 `compat/`，保留补丁证据。真实平台上的同目录重命名及后续gfx936目标识别/JIT仍需这次运行验证；本地测试不代表已经在服务器修复成功。
