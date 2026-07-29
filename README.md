# Open Duck Runtime

本文档面向当前 `mjlab` runtime 部署版本，说明仓库结构、主运行脚本
`scripts/v2_rl_walk_mjlab.py` 的参数，以及唯一保留的关节限位记录调试脚本。

完整观测到电机下发链路可参考 [RUNTIME_PIPELINE.md](RUNTIME_PIPELINE.md)。

## 1. 目录

```text
duck_runtime/

|-- README.md                                      # 当前 runtime 使用说明
|-- RUNTIME_PIPELINE.md                            # observation -> policy -> motor pipeline 详细说明
|-- duck_config.json                               # 机器人配置: IMU 方向、关节 offset、功能开关
|-- pyproject.toml                                 # Python 包配置
|-- setup.cfg                                      # Python 包安装配置
|-- run.sh                                        # 常用运行入口封装
|-- dry.sh                                        # dry-run 调试入口封装
|-- command.sh                                    # 命令行运行辅助脚本
|-- run_duck_on_boot.sh                            # 开机启动脚本
|-- open-duck-mini.service.txt                     # systemd service 示例
|
|-- mini_bdx_runtime/                              # runtime Python package
|   |-- mini_bdx_runtime/
|   |   |-- duck_config.py                         # 读取 duck_config.json
|   |   |-- rustypot_position_hwi.py               # Feetech 舵机硬件接口: 读状态、写目标
|   |   |-- raw_imu.py                             # BNO055 IMU 读取线程
|   |   |-- feet_contacts.py                       # 左右脚 GPIO 触点读取
|   |   |-- onnx_infer.py                          # ONNX Runtime policy 推理封装
|   |   |-- rl_utils.py                            # action dict、滤波器、关节顺序工具
|   |   |-- xbox_controller.py                     # 手柄输入工具
|   |   |-- sounds.py / eyes.py / camera.py         # 表情、声音、相机等扩展功能
|   |-- assets/                                    # 声音等资源
|
|-- scripts/                                       # runtime 和部署脚本
|   |-- v2_rl_walk_mjlab.py                        # 当前 mjlab tracking policy 实机入口
|   |-- v2_rl_walk_mujoco.py                       # 旧版 runtime 入口, 观测格式不同, 不用于当前 mjlab policy
|   |-- verify_mjlab_deploy.py                     # 检查 policy/motion/manifest 是否匹配
|   |-- check_mjlab_initial_pose.py                # 检查 motion_start 或 zero 初始姿态
|   |-- update_neutral_from_config.py              # 从配置更新 neutral/offset 的辅助脚本
|   |-- control_motor.py                           # 单关节/多关节手动控制工具
|   |-- turn_on.py / turn_off.py                   # 电机上电、下电工具
|   |-- sway_t2_full.onnx                          # 当前部署 policy
|   |-- A2_-_Sway_t2_stageii.npz                   # 当前部署 reference motion
|   |-- mjlab_sway_t2_full_manifest.json           # policy/motion/order/scale manifest
|   |-- onnx/                                      # 按动作类型整理的 policy 和 motion
|
|-- checker/                                       # 实机调试工具
|   |-- hardware_joint_limit_recorder.py            # 交互式记录真实关节逻辑限位
|   |-- joint_checker_common.py                    # joint order、默认限位和通用函数
|   |-- logs/                                      # 默认输出 hardware_joint_limits.json
|
|-- cam/                                           # owner/camera 相关实验脚本
|-- mic/                                           # 麦克风、唤醒词相关脚本和资源
|-- rsd/                                           # 其他实验性工具
```

当前主线运行方式：

```bash
cd duck_runtime/scripts
python v2_rl_walk_mjlab.py \
  --onnx_model_path sway_t2_full.onnx \
  --motion_path A2_-_Sway_t2_stageii.npz \
  --duck_config_path ../duck_config.json \
  --control_freq 50 \
  --max_target_step 0.08 \
  --initial_pose motion_start \
  --debug
```

## 2. `v2_rl_walk_mjlab.py` 说明

`scripts/v2_rl_walk_mjlab.py` 是当前 mjlab tracking policy 的实机入口。它每帧完成：

```text
reference motion + IMU + motor state + action history + feet contacts
  -> 114D observation
  -> ONNX policy
  -> 14D action
  -> clip / scale / rate limit / safety limit / optional low-pass filter
  -> 14D target joint positions
  -> HWI adds joints_offsets
  -> rustypot.write_goal_position()
  -> Feetech motors
```

### 必填参数

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--onnx_model_path` | 必填 | ONNX policy 路径 | 换 policy 时指向新的 `.onnx` | policy 必须和 observation schema、action order、action scale 匹配 |
| `--motion_path` | 必填 | reference motion `.npz` 路径 | 换动作时指向新的 `.npz` | motion 的 `joint_pos/joint_vel` 必须能映射到 16D `JOINT_ORDER_16` |

### 硬件和配置参数

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--duck_config_path` | `~/duck_config.json` | 读取 `imu_upside_down` 和 `joints_offsets` | 实机通常传 `../duck_config.json` 或机器人 HOME 下配置 | offset 错会导致观测和下发目标整体偏移 |
| `--serial_port` | `/dev/ttyACM0` | 舵机控制板串口 | 控制板枚举变化时改为 `/dev/ttyUSB0` 等 | 串口错误会无法读取/写入电机 |
| `-p` | `30` | 电机位置环 KP | 初次调试用小值, 例如 `2~10`; 稳定后提高 | 越大越硬, 响应快但更容易抖动或冲击 |
| `-i` | `0` | 电机位置环 KI | 通常保持 0 | 非 0 可能积累误差并引入不可预期动作 |
| `-d` | `0` | 电机位置环 KD | 有速度震荡时可小幅增加 | 阻尼更强, 过大可能响应迟钝 |

### 控制频率和启动姿态

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `-c`, `--control_freq` | `50` | runtime 主循环频率, Hz | 应与训练/导出的 policy 频率一致 | 改变频率会改变动作历史和 reference 播放节奏 |
| `--start_frame` | `0` | reference motion 起始帧 | 想从动作中间开始时调整 | 会改变启动姿态和第一帧 reference |
| `--initial_pose` | `motion_start` | 上电后 policy 开始前保持的姿态 | `motion_start` 或 `zero` | `motion_start` 更适合 tracking policy; `zero` 用于零位检查 |
| `--pitch_bias` | `0.0` | IMU pitch 偏置参数 | 只有确认机身安装角存在固定偏差时再调 | 当前主观测主要用 gyro/accelero, 调整需谨慎 |

### action 后处理参数

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--action_gain` | `1.0` | 全局 action 增益 | 调小如 `0.3~0.8` 可降低动作幅度 | 小: 保守稳定但动作弱; 大: 动作强但更容易越界/抖动 |
| `--action_clip` | `None` | 对 raw action 做对称裁剪 | 调试不确定 policy 时可设 `1.0` 或更小 | 限制网络异常输出, 但过小会削弱策略 |
| `--max_target_step` | `0.08` | 每帧目标关节角最大变化, rad | 初次实机可调小如 `0.02~0.05` | 小: 更柔和但滞后; 大: 跟踪快但冲击更强 |
| `--no_rate_limit` | `False` | 关闭 `max_target_step` 限速 | 只在确认策略和硬件安全后使用 | 关闭后目标可瞬间跳变, 实机风险更高 |
| `--cutoff_frequency` | `None` | 对 target position 做低通滤波 | 有高频抖动时设如 `10~30` | 降噪但增加延迟, 过低会跟不上 policy |

注意：脚本内当前 `ACTION_SCALE_BY_JOINT` 全部为 `0.25`。替换 policy 时必须确认训练侧 action scale 是否一致；如果 manifest 中记录的 scale 不一致，应先同步脚本和 manifest，再上实机。

### 安全限位参数

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--joint_limit_margin` | `0.02` | 在关节硬限位内收缩的安全边界, rad | 初次调试可调大, 如 `0.05` | margin 大更安全但动作空间更小 |
| `--no_safety_clip` | `False` | 关闭关节目标限位裁剪 | 实机上通常不要使用 | 关闭后 policy 可命令到限位外, 风险很高 |

### 观测和调试参数

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--joint_vel_scale` | `1.0` | 写入 observation 前对硬件关节速度统一缩放 | 如果实机速度单位与训练不一致才调整 | 错误缩放会让 policy 误判当前运动状态 |
| `--no_feet_contacts` | `False` | 足底触点 observation 固定为 `[0, 0]` | 足底 GPIO 未接好时使用 | policy 会失去触地信息, 步态表现可能变化 |
| `--dry_run` | `False` | 不创建 HWI, 不写电机 | 在电脑或无机器人环境检查 observation/policy 形状 | 不会真实运动, joint state 用 0 补齐 |
| `--debug` | `False` | 打印详细调试信息并默认写 log | 调参和首次上实机时建议打开 | 输出 obs/action/target/current/ref/timing 等信息 |
| `--debug_interval_s` | `1.0` | debug 打印间隔 | 想看更密集信息可调小 | 太小会刷屏并影响实时性 |
| `--debug_top_k` | `5` | debug 中每类最大项打印数量 | 需要看更多关节时调大 | 只影响日志详细程度 |
| `--log_path` | `None` | 指定 debug/log 输出文件 | 需要固定日志路径时设置 | 便于复盘, 不影响控制逻辑 |

### 推荐调参顺序

1. 先用 `--dry_run --debug` 检查 ONNX 和 motion 能正常加载。
2. 用较低 `-p 2` 或 `-p 5` 做上电和初始姿态检查。
3. 保持 `--initial_pose motion_start`，确认 frame 0 姿态和机器人当前机械零点一致。
4. 初次让 policy 动起来时降低动作强度，例如 `--action_gain 0.3 --max_target_step 0.03 --debug`。
5. 如果动作太弱，逐步提高 `--action_gain`；如果动作冲击大，降低 `--max_target_step` 或增加低通滤波。
6. 调试过程中不要关闭 `safety_clip`；只有在离线分析或非常明确的实验中才考虑 `--no_safety_clip`。

## 3. 调试脚本: 记录关节限位

只保留一个关节限位记录入口：

```bash
cd duck_runtime
python checker/hardware_joint_limit_recorder.py --list
```

`--list` 会打印当前脚本支持的 14 个真实关节顺序。这个顺序必须与
`mini_bdx_runtime/mini_bdx_runtime/rustypot_position_hwi.py` 里的 `HWI.joints`
完全一致，否则脚本会直接报错停止。

### 开始记录

建议从低增益、小步长开始：

```bash
cd duck_runtime
python checker/hardware_joint_limit_recorder.py \
  --duck_config_path duck_config.json \
  --serial_port /dev/ttyACM0 \
  --joint left_hip_roll \
  --step 0.02 \
  --kp 2 \
  --kd 0 \
  --rate 20 \
  --output checker/logs/hardware_joint_limits.json
```

脚本启动后会：

1. 读取 `duck_config.json` 中的关节 offset。
2. 创建 `HWI` 并连接舵机控制板。
3. 给所有 checker 关节设置低 KP/KD。
4. 先把所有目标移动到逻辑 0 位。
5. 进入交互式按键控制。

### 交互按键

```text
w / +       当前关节目标角增加 step
s / -       当前关节目标角减小 step
r           记录当前关节的 observed logical position
0           当前关节目标回到 0
space       所有关节目标回到 0
. / n       切到下一个关节
, / p       切到上一个关节
数字键      选择对应序号关节
q           退出并保存
```

### 参数说明

| 参数 | 默认值 | 意义 | 如何调整 | 影响 |
|---|---:|---|---|---|
| `--joint` | `left_hip_roll` | 启动时选中的关节 | 换成要记录的关节名 | 只影响初始选中项 |
| `--duck_config_path` | 仓库内 `duck_config.json` | 读取 offset | 使用当前机器人真实配置 | offset 错会导致记录的逻辑限位不可信 |
| `--serial_port` | `/dev/ttyACM0` | 舵机串口 | 按实际设备修改 | 错误则无法通信 |
| `--step` | `0.02` | 每次按键目标变化量, rad | 接近机械限位时调小 | 小更安全, 记录更细; 大移动更快但风险高 |
| `--limit` | `None` | 交互命令的对称软限制 | 默认使用脚本内 XML limit; 必要时手动设小 | 限制越小越安全, 但可能到不了真实边界 |
| `--margin` | `0.02` | 距默认关节限位的安全 margin | 初次调试可加大 | 更不容易撞限位, 但记录范围会更保守 |
| `--kp` | `2.0` | 记录时电机 KP | 初次保持低值; 需要更强保持再小幅增加 | 越大越硬, 风险和冲击更高 |
| `--kd` | `0.0` | 记录时电机 KD | 有抖动再小幅增加 | 增加阻尼, 过大变迟钝 |
| `--rate` | `20.0` | 控制循环频率, Hz | 通常不用改 | 太低响应慢, 太高可能刷屏或占用更多资源 |
| `--neutral_hold_seconds` | `1.0` | 启动后保持逻辑 0 的时间 | 想给机器人更多稳定时间可调大 | 启动更慢但更稳 |
| `--turn_off_on_exit` | `False` | 退出时是否关闭电机 torque | 需要安全下电时加上 | 加上后退出会释放电机 |
| `--output` | `checker/logs/hardware_joint_limits.json` | 限位记录 JSON 输出路径 | 每台机器人建议单独文件 | 脚本会去重并更新 min/max |
| `--list` | `False` | 只打印关节表 | 检查 joint order 时使用 | 不连接电机、不记录 |

### 记录建议

1. 每次只选一个关节，缓慢靠近正向边界，按 `r` 记录。
2. 回到 0，再缓慢靠近负向边界，按 `r` 记录。
3. 用 `.` 或 `n` 切到下一个关节，重复记录。
4. 如果接近机械极限、连杆干涉、舵机声音异常或温度升高，立即停止继续推进。
5. 退出后检查 `checker/logs/hardware_joint_limits.json`，再决定是否把限位同步到 runtime 的 `JOINT_LIMITS_BY_JOINT`。

限位记录的核心原则是：宁可保守一点，也不要把 policy 的可命令范围开到机械结构真的撞边的位置。
