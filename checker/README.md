# OpenDuck Joint Checker

这套脚本用于排查 MuJoCo 和实机之间的关节零位、offset、方向、顺序是否一致。

核心思路：

- MuJoCo 端直接写 `qpos` 并 `mj_forward`，不 `mj_step`，所以先不考虑物理接触和动力学。
- 实机端沿用 `duck_runtime` 的 `HWI`、`duck_config.json` offset 和电机 ID。
- 两边使用同一套 14 个可命令关节顺序和 real XML 硬限位。
- 每次遥控都会输出 CSV，之后可以按 joint 对比 target/current 轨迹。

## 关节列表

```bash
python checker/mujoco_joint_checker.py --list
python checker/hardware_joint_checker.py --list
```

## MuJoCo 端查看方向

在主机上运行：

```bash
cd /home/luolinfeng/duck_ws/duck_amp
python checker/mujoco_joint_checker.py \
  --joint left_hip_roll \
  --step 0.02 \
  --log checker/logs/mujoco_left_hip_roll.csv
```

按键：

- `w` 或 `+`：增加当前关节 target
- `s` 或 `-`：减小当前关节 target
- `0`：当前关节归零
- `space`：所有关节归零
- `.` / `n`：下一个关节
- `,` / `p`：上一个关节
- `q`：退出

MuJoCo checker 不调用 `mj_step`，只更新几何姿态，所以看到的是纯 MJCF 坐标方向。

## 实机端查看方向

在机器人上运行同一个仓库，或把 `checker/` 和 `duck_runtime/` 同步过去：

```bash
cd /home/lxkj/Open_Duck_Mini_Runtime
python checker/hardware_joint_checker.py \
  --joint left_hip_roll \
  --duck_config_path ~/duck_config.json \
  --serial_port /dev/ttyACM0 \
  --kp 2 \
  --kd 0 \
  --step 0.02 \
  --log checker/logs/hardware_left_hip_roll.csv
```

脚本支持两种目录结构：

- `duck_amp/checker` 和 `duck_amp/duck_runtime` 同级；
- `Open_Duck_Mini_Runtime/checker` 和 `Open_Duck_Mini_Runtime/mini_bdx_runtime` 同级。

实机端会打印：

- `target`：发送给 runtime 的逻辑目标角
- `logical`：经过 `duck_config.json` offset 修正后的读数
- `raw`：电机原始读数

如果 MuJoCo 中 `+0.02` 是向内收腿，但实机 `+0.02` 是向外劈腿，说明这个关节方向 sign 反了。

如果方向一致但 `target=0` 时姿态不一致，说明 offset/机械零位没对齐。

## 画轨迹

```bash
python checker/plot_joint_logs.py \
  checker/logs/mujoco_left_hip_roll.csv \
  checker/logs/hardware_left_hip_roll.csv \
  --joint left_hip_roll
```

也可以保存图片：

```bash
python checker/plot_joint_logs.py \
  checker/logs/mujoco_left_hip_roll.csv \
  checker/logs/hardware_left_hip_roll.csv \
  --joint left_hip_roll \
  --output checker/logs/left_hip_roll_compare.png
```

## 建议排查顺序

1. `left_hip_roll`
2. `right_hip_roll`
3. `left_hip_yaw`
4. `right_hip_yaw`
5. `left_hip_pitch`
6. `right_hip_pitch`
7. knee / ankle

先确认 `0` 位，再确认 `+step` 和 `-step` 的方向，最后再跑 policy。
