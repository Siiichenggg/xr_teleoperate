# G1_29 Payload Compensation 评估与可视化说明

这份说明对应新增的离线评估脚本：

- [teleop/utils/payload_eval.py](/home/sicheng/xr_teleoperate/teleop/utils/payload_eval.py)

它的目标不是做学术化 benchmark，而是帮助你在当前仓库里快速回答下面这些工程问题：

- 打开 payload compensation 之后，手臂在持物时的跟踪误差有没有下降
- 下降了多少，能不能给出具体数字
- 能不能做出直观的 before / after 图和图片对比
- 如果环境支持，能不能导出一个 side-by-side 对比视频

---

## 1. 现在这套评估能输出什么

### 1.1 数字指标

脚本会自动计算以下核心指标：

- active side 关节跟踪误差 RMS
- active side 关节跟踪误差 P95
- active side 关节误差范数 RMS / Max
- 末端位置误差 RMS / Max
- 末端姿态误差 RMS / Max
- Hold 段关节误差 RMS
- Hold 段末端位置误差 RMS
- Hold 段末端姿态误差 RMS

其中：

- 关节误差来自 `q_cmd - q_meas`
- 末端误差来自对命令关节角和实测关节角分别做 FK，再比较 `L_ee / R_ee`
- Hold 段由“命令速度低 + 实测速度低”联合筛选，更适合衡量静态 / 准静态持物能力

### 1.2 可视化结果

脚本会输出：

- `summary.json`
- `report.md`
- `report.html`
- `joint_error_norm.svg`
- `ee_position_error.svg`
- `ee_orientation_error.svg`
- `joint_rms_bar.svg`

如果录制里带了相机图像，`report.html` 还会自动插入：

- before / after 的代表性关键帧对比图

如果运行环境里有 `ffmpeg`，并且你加上 `--make-video`，还会尝试生成：

- `color_0_side_by_side.mp4`

---

## 2. 为了公平比较，推荐怎么录制

如果你想让“补偿前 vs 补偿后”的数字有说服力，建议不要直接拿两段随意操作的遥操作录制做比较，而是尽量用一个固定流程。

推荐协议：

1. 选择固定 payload
2. 固定抓取方式和抓取姿态
3. 固定 `payload_side`
4. 做同一套动作：
   - 抬起
   - 保持静止 3 到 5 秒
   - 缓慢移动
   - 再保持静止 3 到 5 秒
5. 录两次：
   - 一次关闭 payload compensation
   - 一次打开 payload compensation

如果可以，最好控制以下条件一致：

- 同一个操作者
- 同一个 payload
- 同一段动作节奏
- 同一 camera 视角
- 同一频率

这样 `report` 里的数字才更能反映 compensation 本身，而不是操作差异。

---

## 3. 录制前后对比数据

### 3.1 补偿前

关闭 payload compensation，录一个 episode。

示意命令：

```bash
cd /home/sicheng/xr_teleoperate/teleop
python teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex1 \
  --record \
  --task-dir ./utils/data/payload_eval \
  --task-name before_comp
```

### 3.2 补偿后

打开 payload compensation，用同样流程再录一个 episode。

示意命令：

```bash
cd /home/sicheng/xr_teleoperate/teleop
python teleop_hand_and_arm.py \
  --arm G1_29 \
  --ee dex1 \
  --record \
  --payload-enable \
  --payload-side right \
  --payload-mass 0.8 \
  --payload-com 0.02 0.0 0.08 \
  --task-dir ./utils/data/payload_eval \
  --task-name after_comp
```

### 3.3 录制中现在会额外保存什么

为了支持评估，我已经把录制内容补齐了。新录制的 episode 里，arm 部分会额外带上：

- `states.left/right_arm.qvel`
- `states.left/right_arm.torque`
- `actions.left/right_arm.torque`
- `metrics.payload.tau_nominal`
- `metrics.payload.tau_payload`
- `metrics.payload.sol_tauff`
- `metrics.payload.mass_hat`（如果估计器启用）

这意味着后续不只是能看 `q_cmd / q_meas`，也能更细地回看 payload 相关的 torque 变化。

---

## 4. 跑离线评估

假设你已经录好了：

- `before_comp/episode_0001`
- `after_comp/episode_0001`

就可以直接运行：

```bash
cd /home/sicheng/xr_teleoperate
python teleop/utils/payload_eval.py \
  --before ./teleop/utils/data/payload_eval/before_comp/episode_0001 \
  --after ./teleop/utils/data/payload_eval/after_comp/episode_0001 \
  --output-dir ./docs/g1_29_payload_compensation/eval_example \
  --active-side right
```

如果你的环境里有 `ffmpeg`，还可以附加：

```bash
  --make-video
```

---

## 5. 输出怎么看

### 5.1 `summary.json`

这是最适合程序读取或后处理的结构化结果，里面会包含：

- before 指标
- after 指标
- improvement 百分比

### 5.2 `report.md`

这是最适合直接发给同事或贴到 PR / issue 的精简文本总结。

### 5.3 `report.html`

这是最适合直观看效果的版本。里面会有：

- 核心数字表格
- before / after 曲线图
- 关键帧图像对比
- 如果有视频，也可以一起附带

---

## 6. 哪些数字最值得看

如果你要快速回答“补偿有没有明显提升”，优先看这几项：

1. `hold_ee_position_error_rms_mm`
2. `hold_ee_orientation_error_rms_deg`
3. `hold_joint_error_rms_deg`
4. `joint_error_p95_deg`

原因是：

- payload 补偿最主要改善的是静态 / 准静态持物能力
- 所以 Hold 段指标比全程平均更敏感
- `P95` 比单纯 `max` 更稳，更不容易被单点尖峰污染

如果你要做汇报，推荐用下面这个叙述顺序：

1. 先给 Hold 段末端位置误差下降百分比
2. 再给 Hold 段末端姿态误差下降百分比
3. 再补充关节 RMS 和关键帧图像

---

## 7. 当前评估的局限

这套评估已经足够做工程比较，但仍然有几个边界需要注意：

- 目前默认只围绕 `G1_29` 设计
- 更适合单手 active side 的 payload 场景
- 最适合静态 / 准静态动作
- 如果 before / after 两次录制动作差异太大，数字的可比性会下降
- 若当前环境缺少 `pinocchio`，脚本会退化为只输出 joint-space 指标，ee 指标会显示 `N/A`
- 若没有图像或没有 `ffmpeg`，视频相关输出会被跳过

---

## 8. 一句话建议

如果你接下来要正式做“效果提升”展示，最推荐的流程是：

1. 录一组固定 payload 的 before / after episode
2. 用 `payload_eval.py` 出 `summary.json + report.html`
3. 在汇报里主打 Hold 段 ee 误差下降数字
4. 再配一张关键帧 before / after 图，必要时再加 side-by-side 视频
