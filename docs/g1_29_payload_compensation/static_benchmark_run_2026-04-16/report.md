# G1_29 Payload Compensation Static Model Benchmark

## Configuration

- side: `right`
- payload_mass: `0.8` kg
- payload_com_ee: `[0.02, 0.0, 0.08]` m
- payload_scale: `1.0`
- arm_tau_limit: `None`
- samples: `500`

## Key Numbers

| Metric | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Joint RMS (deg) | 4.8353 | 0.0000 | +100.00% |
| Joint P95 (deg) | 9.9468 | 0.0000 | +100.00% |
| Joint Max (deg) | 11.2299 | 0.0000 | +100.00% |
| EE Position RMS (mm) | 44.2778 | 0.0000 | +100.00% |
| EE Position P95 (mm) | 65.4942 | 0.0000 | +100.00% |
| EE Position Max (mm) | 76.2945 | 0.0000 | +100.00% |
| EE Orientation RMS (deg) | 18.5644 | 0.0000 | +100.00% |
| EE Orientation P95 (deg) | 23.3320 | 0.0000 | +100.00% |
| EE Orientation Max (deg) | 25.3291 | 0.0000 | +100.00% |
| Residual Tau RMS (Nm) | 4.3963 | 0.0000 | +100.00% |

## Interpretation

- 这是一个**模型级**静态 benchmark，不是机器人实机数据，也不是 Isaac 闭环仿真数据。
- Before 假设前馈中没有 payload torque；After 假设前馈中加入当前实现的 payload compensation。
- 关节静态偏差用 `dq ~= residual_tau / kp` 近似，再通过 FK 估算 ee 位置 / 姿态偏差。
- 因此这些数字最适合回答：在现有动力学模型和 controller 刚度下，payload compensation 理论上能消掉多大一部分静态持物偏差。