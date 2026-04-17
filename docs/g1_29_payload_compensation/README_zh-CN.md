# G1_29 Payload Gravity Compensation 与在线质量估计说明

## 1. 文档目的

这份文档用于总结当前在 `G1_29` 双臂遥操作链路中，对 payload gravity compensation 部分做出的改动，以及这些改动背后的动力学原理和工程取舍。

这里要强调的核心定位是：

- 这次改动的目标不是“给手指单独做重力补偿”
- 而是把**手里抓取物体的重量**建模为**末端执行器上的 payload 产生的额外外力/外力矩**
- 再把这部分末端外载，通过 Jacobian 映射成**整条手臂的关节前馈力矩**
- 并在此基础上，进一步补齐观测链和估计链，使系统具备**从已知质量静态补偿，扩展到在线质量估计**的结构能力

从系统演进角度看，这不是一次孤立的 estimator 增补，而是把原有链路从“只做名义 arm feedforward”，逐步扩展为：

1. 已知质量、固定质心的静态 / 准静态 payload 重力补偿
2. 支撑未知质量估计的数据观测链
3. 可选接入的在线 `mass_hat` 估计模块
4. 带门控、低通和平滑限速的 `mass_hat -> payload_mass_cmd` 闭环写回

当前实现的总体目标，是在 `G1_29` 上先把这四条链路做成一个可分阶段启用、可记录、可回退的工程结构：

- 关闭 estimator 时，系统行为退化回手工设定 payload 参数
- 打开 estimator 但关闭闭环时，只做观测、日志和 CSV 验证
- 打开闭环时，才把 `mass_hat` 经过门控和平滑后写回 payload compensation

---

## 2. 改动总览

### 2.1 现有控制链路中的挂点

当前 `G1_29` 机械臂控制主链路可以概括为：

```text
solve_ik(...) -> pin.rnea(...) -> sol_tauff -> arm controller -> motor_cmd[id].tau
```

这次改动没有改变主链路结构，而是在这个现有链路上做了**最小侵入式扩展**：

- IK 层继续保留原始 `tau_nominal = pin.rnea(...)`
- 新增 `tau_payload`
- 最终输出改为：

```text
sol_tauff = tau_nominal + tau_payload
```

因此，这次改动的本质不是替换原控制框架，而是把 payload 对手臂动力学的影响，作为一个额外前馈项并入现有 `sol_tauff`。

### 2.2 改动文件

- `teleop/teleop_hand_and_arm.py`
  - 增加 `G1_29` payload 配置入口
  - 可选启用 `PayloadEstimatorG1_29`
  - 在主循环中组装 `arm_obs = {"q", "dq", "tau_est"}`
  - 增加 estimator debug snapshot / 周期日志 / CSV 记录
  - 增加 `payload_est_closed_loop` 的主循环门控与质量写回
- `teleop/robot_control/robot_arm_ik.py`
  - 为 `G1_29_ArmIK` 增加 `payload_cfg`
  - 新增 `_compute_payload_tau(...)`
  - 在 `solve_ik(...)` 中构造 `sol_tauff = tau_nominal + tau_payload`
- `teleop/robot_control/robot_arm.py`
  - 在 `G1_29` lowstate 缓存中加入 `tau_est`
  - 新增 `get_current_dual_arm_tau_est()`
  - 在真正写入 `motor_cmd[id].tau` 前增加 `tau` 限幅
- `teleop/robot_control/payload_estimator.py`
  - 新增 `PayloadEstimatorG1_29`
  - 基于 `q / dq / tau_est` 在线估计 `mass_hat`
  - 暴露 `get_debug_snapshot()`
  - 在 `hold / update` 两种模式下缓存最近一次可读调试状态

---

## 3. 第一阶段：已知质量的静态 / 准静态 payload gravity compensation

第一阶段的目标是：在已知 `payload_mass` 和固定 `payload_com` 的前提下，为 `G1_29` 机械臂增加 payload 重力补偿。

### 3.1 作用范围

这一阶段的实现边界非常明确：

- 只支持 `G1_29`
- 没有扩展到 `G1_23 / H1_2 / H1`
- 没有改 whole-body controller
- 没有改 hand controller

这是一个典型的“先在最小有效边界内验证建模合理性”的策略。原因很直接：

- 当前仓库已有成熟的 `G1_29` arm IK + feedforward 链路
- payload 对系统的第一影响对象是 arm torque feedforward
- 若直接放大到 whole-body，会把 torso / waist / contact / balance / base coupling 一并引入，显著扩大回归面
- 若直接改 hand controller，则会把“夹持控制”和“承载重力”混成一个问题，不利于验证

所以第一阶段的切入点非常清晰：**先把 payload 当作 ee 外载，对 arm feedforward 做补偿。**

### 3.2 参数入口

在主脚本中增加了 `G1_29` 专用 payload 参数：

- `payload_enable`
- `payload_side`
- `payload_mass`
- `payload_com`
- `payload_scale`
- `payload_debug`
- `payload_log_every`
- `arm_tau_limit`

这些参数的职责划分也比较明确：

- `payload_enable`：控制是否启用补偿
- `payload_side`：指定当前 payload 挂在哪一侧末端
- `payload_mass`：已知 payload 质量
- `payload_com`：payload 质心在 `L_ee / R_ee` 局部坐标系中的偏移
- `payload_scale`：对理论补偿力矩做缩放，便于实机调试时保守打开
- `payload_debug / payload_log_every`：控制调试输出
- `arm_tau_limit`：给 controller 层增加最终安全护栏

工程上，这样做的好处是：**补偿策略、调试开关和安全边界都被明确参数化，而不是散落在控制逻辑里。**

### 3.3 在 IK 层的挂载方式

`G1_29_ArmIK` 中保留了原始 `rnea` 输出作为名义项：

```python
tau_nominal = pin.rnea(model, data, q, v, 0)
```

随后额外计算 payload torque：

```python
tau_payload = self._compute_payload_tau(sol_q)
sol_tauff = tau_nominal + tau_payload
```

这个挂法非常关键，因为它说明：

- 原有 IK 和名义动力学逻辑没有被推翻
- payload 补偿被明确实现为一个附加前馈项
- 原系统的行为与 payload 行为在结构上是可分离的

这比把 payload 效果“揉进”别的控制参数里更清晰，也更便于后续调试与回归分析。

### 3.4 payload torque 的动力学构造

#### 3.4.1 基本思想

抓在手里的物体，本质上是一个附着在末端执行器上的刚体负载。只要物体有质量，在重力场中它就会对末端施加一个等效 wrench。

因此，这里没有把问题表述为“手指关节需要补多少”，而是表述为：

- ee 上额外承受了一个由 payload 重量产生的外力
- 如果 payload 质心不在 ee 原点上，还会产生一个额外力矩
- 这个 ee wrench 会通过雅可比矩阵映射成上游关节力矩需求

#### 3.4.2 参考坐标系

payload 的质心 `payload_com` 定义在 active side 的 `L_ee / R_ee` frame 下。

这意味着：

- `payload_com` 是一个局部几何描述
- 只和末端几何关系有关
- 不直接依赖世界系姿态

在实际计算时，需要把它旋转到世界系：

\[
r = R_{world \leftarrow ee} \cdot payload\_com
\]

其中 `r` 表示 payload 质心相对 ee 原点的世界系力臂向量。

#### 3.4.3 重力和力矩

在世界系中，payload 的重力为：

\[
f = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix}
\]

因为质心可能偏离 ee 原点，所以相对 ee 原点的附加力矩为：

\[
m = r \times f
\]

于是 ee 上的 6D wrench 为：

\[
w_{payload} =
\begin{bmatrix}
m_x & m_y & m_z & f_x & f_y & f_z
\end{bmatrix}^{\top}
\]

实现时遵循 Pinocchio 的 spatial wrench 排列约定：

```text
[m_x, m_y, m_z, f_x, f_y, f_z]
```

#### 3.4.4 Jacobian 映射到关节空间

当前实现使用 `LOCAL_WORLD_ALIGNED` 形式的 frame Jacobian：

\[
\tau_{payload} = J^\top w_{payload}
\]

之所以这样写，核心原因是虚功关系：

\[
\delta W = w^\top \delta x = \tau^\top \delta q
\]

而 `\delta x = J \delta q`，因此有：

\[
\tau = J^\top w
\]

这一步的物理意义是：

- payload 的重力首先在末端体现为 wrench
- 机械臂并不是直接在笛卡尔空间“抵消它”
- 而是要在各个关节上生成对应的附加支撑力矩

所以 `J^\top w` 的本质，就是把末端负载映射成关节空间中的力矩需求。

### 3.5 为什么只作用在 active side

当前 `G1_29` reduced model 是 14 维，顺序为：

- 左臂：`0:7`
- 右臂：`7:14`

在第一版中，你明确把 payload 视为“某一侧末端抓着的物体”，因此：

- 如果 `payload_side == left`，则只保留 `tau_payload[:7]`
- 如果 `payload_side == right`，则只保留 `tau_payload[7:14]`
- 非 active side 直接置零

这样做的意义有两层：

1. 与当前“单手抓取 payload”的使用假设保持一致
2. 避免 Jacobian 映射后数值上在另一侧产生非零耦合项，影响调试判断

这不是在说双臂系统物理上完全无耦合，而是在当前 reduced dual-arm 局部建模中，主动把补偿限定在 active side 内，以便让补偿含义、调参和验证都更明确。

### 3.6 controller 层的 `tau` 限幅

你把安全护栏放在 controller 层，而不是 IK 层，这是一个非常合理的职责拆分。

也就是说：

- IK 层只负责生成算法意义上的 `sol_tauff`
- controller 层在真正写 `motor_cmd[id].tau` 前做 `clip`

这样拆的好处是：

- 算法输出保留“理论目标值”
- 安全护栏保留“实际下发值”
- 调试时可以区分“模型计算出了多大 torque”和“最终为了安全被限制到多少 torque”

这对于分析 payload 补偿是否过强、过于激进或数值异常，非常有价值。

---

## 4. 为什么这不是“给手指单独做重力补偿”

### 4.1 问题的物理本质是 payload，而不是 finger gravity

当机械手抓住一个物体时，真正新增的负载不是“手指自身更重了”，而是：

- 末端抓持了一个有质量的物体
- 这个物体在重力作用下对 ee 产生持续外载
- 这个外载再传递到 wrist / elbow / shoulder 等所有上游关节

因此，这个问题的主导项不是手指局部问题，而是**整条机械臂的承载问题**。

### 4.2 为什么补偿应该落在 arm torque feedforward

hand controller 的主要职责是：

- 开合
- 夹持
- 接触保持
- 抓取动作执行

而 payload 重量导致的额外负担，真正需要承担的是 arm joints。

如果只改 hand controller：

- 可能改善夹持稳定性
- 但无法给肩、肘、腕增加足够的前馈支撑
- 也无法解决“抬手后整条手臂往下坠”这类核心动力学问题

所以补偿的正确落点应该是：

- 把 payload 建成 ee 外载
- 把 ee 外载映射到 arm torque feedforward
- 再由 arm controller 下发到各个电机

### 4.3 为什么第一版不直接做 whole-body

从工程实现角度，当前仓库第一版不适合直接上 whole-body，有几个现实原因：

- 当前最稳定、最直接的控制入口就是 dual-arm `sol_tauff`
- `G1_29` 现有 reduced model 已经天然适合先验证 arm-local 的 payload 补偿
- whole-body 一旦引入，就需要重新定义 torso、waist、base、contact 和 payload 之间的耦合关系
- 这样会显著增加建模复杂度、调试复杂度和实机风险

因此，当前版本的重点不是追求一次性做全，而是优先验证：

- payload 的动力学解释是否正确
- Jacobian wrench 映射是否合理
- 实际前馈响应是否改善

---

## 5. 第二阶段：补齐观测链，为未知质量估计做准备

第一阶段解决的是“已知质量怎么补”。第二阶段解决的是：**如果 payload 质量未知，系统还缺什么观测量？**

答案是：只靠 `q / dq` 不够，必须把 `tau_est` 拉进来。

### 5.1 为什么只有 `q / dq` 不够

`q` 和 `dq` 能告诉我们：

- 当前姿态
- 当前速度
- 是否满足静态 / 准静态近似
- 在模型里应该计算出什么样的 nominal torque

但它们不能告诉我们：

- 电机实际上承受了多大的关节力矩
- 模型预测和真实系统之间到底差了多少

也就是说，只靠 `q / dq`，你能做前向计算，但做不了基于真实观测的 payload 反推。

### 5.2 为什么需要 `tau_est`

未知质量估计的核心思想，是比较：

- 模型预测的名义力矩 `tau_nominal`
- 系统实际观测到的力矩 `tau_est`

两者差值中，才包含“额外 payload 负载”的信息。

因此第二阶段新增内容包括：

- 在 arm controller 的 lowstate subscriber 中缓存 `tau_est`
- 提供 `get_current_dual_arm_tau_est()`
- 在主循环中组装：

```python
arm_obs = {
    "q": current_lr_arm_q,
    "dq": current_lr_arm_dq,
    "tau_est": current_lr_arm_tau_est,
}
```

这样之后，估计器不再只是看运动学状态，而是能看到“模型预测 torque”和“实测 torque”的差异。

### 5.3 为什么未知质量估计依赖 torque residual

如果当前姿态下，无 payload 时理论上需要的关节力矩是：

\[
\tau_{nominal}
\]

而驱动实际估计到的是：

\[
\tau_{est}
\]

那么两者的残差：

\[
\tau_{residual} = \tau_{est} - \tau_{nominal}
\]

就可以视作“模型未解释掉的额外力矩”。

在静态 / 准静态抓取场景下，这部分残差最主要的来源，就是 payload 的重量。

所以第二阶段的真实意义，不是简单加了一个 getter，而是把系统从“纯前馈补偿结构”推进到了“具备基于残差做在线估计的观测结构”。

---

## 6. 第三阶段：新增 `PayloadEstimatorG1_29`

### 6.1 模块定位

`PayloadEstimatorG1_29` 当前是一个单独模块，可选接入主循环，用来在线输出和记录 `mass_hat`。

当前它的定位非常克制：

- 只支持 `G1_29`
- 只估计一个标量 `mass_hat`
- 固定 `payload_com`
- 只在 active side 上做估计
- 还没有把估计结果闭环写回补偿链

也就是说，它现在更像一个“验证型估计器”，目的是先确认：

- 当前残差是否真的和 payload 质量相关
- 标量质量估计在现有观测噪声下是否稳定
- 估计值是否具备写回补偿链的基本可信度

### 6.2 为什么第一版只估 `mass_hat`

这是一个非常重要的建模降维。

如果同时估计：

- `mass_hat`
- `com_hat`

那么问题会从“一个标量估计”立刻变成“质量和几何参数耦合估计”。

这会带来几个问题：

- 观测可辨识性明显变差
- 估计容易发散或高度抖动
- 需要更多姿态激励和更严格的实验设计
- 更难判断误差到底来自质量估错，还是质心设错

因此第一版固定 `payload_com`，只估 `mass_hat`，是一个非常工程化的取舍：

- 先把几何假设固定住
- 先判断质量这一维是否能可靠估出来
- 等验证通过后，再决定是否扩展到 `com_hat`

### 6.3 为什么估计器内部使用 `rnea(q, 0, 0)`

估计器内部不是直接拿控制链输出的 `sol_tauff`，而是自己重新算：

\[
\tau_{nominal} = rnea(q, 0, 0)
\]

这样做有两个好处：

1. 让估计器与补偿器解耦  
   如果直接拿已经叠加 payload 的 `sol_tauff` 做比较，估计会被控制链反向污染。

2. 明确采用静态 / 准静态近似  
   当前估计器假设主要场景是慢速持物、静止悬停或准静态动作，因此先忽略显著惯性项和加速度项，把问题压缩为姿态相关重力负载识别。

这与第一阶段的补偿思路是自洽的：先围绕 quasi-static payload gravity 这一主导项构建闭环。

### 6.4 单位质量 torque 模板

在固定 `payload_com` 的前提下，给定当前姿态 `q`，可以构造单位质量 `1 kg` 时的 payload torque 模板：

\[
\tau_{unit}(q)
\]

其构造方式与第一阶段完全一致，只是把质量固定为 `1`：

- 世界系重力
  \[
  f_{unit} = [0, 0, -g]^\top
  \]
- 力矩
  \[
  m_{unit} = r \times f_{unit}
  \]
- 末端 wrench
  \[
  w_{unit} = [m_{unit}, f_{unit}]
  \]
- 再通过
  \[
  \tau_{unit} = J^\top w_{unit}
  \]
  映射到关节空间

这样一来，在固定姿态、固定质心的前提下，payload torque 与质量近似满足线性关系：

\[
\tau_{payload}(q, m) \approx m \cdot \tau_{unit}(q)
\]

### 6.5 基于残差的标量质量估计

有了：

- `tau_nominal`
- `tau_est`
- `tau_unit`

之后，可以构造残差：

\[
\tau_{residual} = \tau_{est} - \tau_{nominal}
\]

若认为这部分残差主要来自 payload，则有：

\[
\tau_{residual} \approx m \cdot \tau_{unit}
\]

于是可以写成一个一维最小二乘问题：

\[
\min_m \left\| \tau_{residual} - m \tau_{unit} \right\|_2^2
\]

其闭式解就是：

\[
m_{raw} =
\frac{\tau_{unit}^{\top}\tau_{residual}}
{\tau_{unit}^{\top}\tau_{unit}}
\]

这一步非常关键。它说明当前估计器不是“凭经验猜质量”，而是在做一个非常明确的投影：

- `tau_unit` 定义了“payload 重量在关节空间中应该长成什么方向”
- `tau_residual` 是实际观测到的未解释力矩
- 将 `tau_residual` 投影到 `tau_unit` 上，就能得到最匹配的质量标量

### 6.6 为什么需要门控、EMA、限幅和 hold / fallback

真实系统中的 `tau_est` 并不干净，`tau_residual` 也不只包含 payload 信息，还会混入：

- 电机估计噪声
- 摩擦
- 轻微碰撞
- 动态项残留
- 控制误差

因此第一版估计器必须有工程稳健性处理。

#### 6.6.1 可观测性门控

如果当前姿态下 `tau_unit` 很小，那么：

\[
\tau_{unit}^{\top}\tau_{unit}
\]

就会很小，导致分母接近零，质量估计对噪声极其敏感。

所以需要 observability gate：在模板幅值过小时，不更新估计。

#### 6.6.2 `dq` 低速门控

当前估计器假设静态 / 准静态有效。如果某一时刻 `dq` 明显偏大，那么：

- 惯性项和速度相关项会污染 `tau_residual`
- 当前 residual 就不再主要反映 payload 重力

所以需要用 active side 的 `dq` 做低速门控，超阈值时直接 hold。

#### 6.6.3 合法性检查

对 `q / dq / tau_est` 做有限值检查，是为了避免：

- DDS 字段缺失
- 观测出现 NaN / inf
- 输入维度错误

否则估计器很容易把单次异常放大成连续错误。

#### 6.6.4 EMA 平滑

即使在有效区间内，`mass_raw` 也会有抖动。因此加入 EMA：

\[
mass\_hat \leftarrow \alpha \cdot mass\_clipped + (1-\alpha)\cdot mass\_hat
\]

这样能在不改变整体趋势的前提下，显著减少单帧噪声。

#### 6.6.5 限幅与 hold / fallback

限幅到 `[mass_min, mass_max]` 是为了把估计约束在合理物理范围内。

当输入非法、观测不可用或姿态不可观时：

- 若已有上一次有效值，则 hold
- 若没有，则回退到 `0.0`

这可以防止估计器在暂时失效时把输出直接打成强烈跳变，从而影响后续可能的闭环接入。

---

## 7. 这三阶段串起来以后，系统能力发生了什么变化

从系统演化角度看，当前工作最重要的价值，不在于“加了一个 estimator 文件”，而在于你把 payload 补偿能力从静态已知参数模型，扩展成了一个分阶段可验证的结构：

### 阶段一：已知质量补偿

系统已经能够在 `G1_29` 上，根据：

- 当前姿态 `q`
- 配置好的 `payload_mass`
- 配置好的 `payload_com`

实时计算：

\[
\tau_{payload} = J^\top w_{payload}
\]

并叠加到：

\[
sol\_tauff = \tau_{nominal} + \tau_{payload}
\]

这意味着已具备**明确的 payload gravity compensation 能力**。

### 阶段二：观测链补齐

系统已经能够稳定拿到：

- `q`
- `dq`
- `tau_est`

这意味着它不再只是一个纯开环前馈补偿器，而是具备了**基于实测力矩残差进行识别**的观测条件。

### 阶段三：在线质量估计能力具备

系统已经能在固定 `payload_com` 的前提下，计算 `mass_hat`，并通过主循环周期日志、CSV 与 `payload_metrics` 记录其 `update / hold` 行为。

所以更准确的表述应该是：

> 当前系统已经从“已知质量的静态 payload gravity compensation”，扩展到了“具备在线 payload 质量估计与验证链”的结构。

### 阶段四：保守闭环写回已接入

当前仓库已经支持一个仅限 `G1_29` 的最小闭环版本。它不是无保护地直接把 `mass_hat` 写回补偿链，而是经过以下几层门控后再更新 `payload_cfg["mass"]`：

- 只在 `--payload-est-enable --payload-est-closed-loop` 同时打开时启用
- 只接受 `mode=update` 且 `reason=ok` 的 estimator 输出
- 需要连续若干次有效更新后才进入 tracking
- 写回质量先做低通，再做每周期变化率限制
- estimator 一旦进入 `hold`，闭环立即冻结最近稳定质量，不把异常值传播到补偿链

当前闭环的工程定位是：

- 提供一个可用的最小在线自适应版本
- 继续保留原有 `tau` clip 和 payload debug 护栏
- 让“估计值”和“真正写回到补偿链的质量命令”在日志里可区分

---

## 8. 当前方案的边界和局限

这部分需要明确写清楚，避免对系统能力产生过度解读。

### 8.1 支持范围

- 只支持 `G1_29`
- 只支持单手 active side payload
- 只在当前 dual-arm reduced model 范围内工作

### 8.2 动力学假设边界

- 只适合静态 / 准静态 payload 场景
- 不适合高速挥动、大加速度动作、强碰撞接触、复杂操作任务中的强动态阶段

在这些情况下，`tau_residual` 不再主要由 payload 重力主导，估计精度和补偿效果都会下降。

### 8.3 参数化边界

- 当前只估 `mass_hat`
- 还没有估 `com_hat`
- `payload_com` 依然需要人工给定

这意味着一旦 `payload_com` 设置偏差较大，估计到的 `mass_hat` 可能会承担一部分几何误差。

### 8.4 控制架构边界

- 还没有引入 whole-body coupling
- 还没有把 torso / waist / base / contact 一起纳入补偿
- 也没有把 hand controller 纳入 payload 动力学统一建模

### 8.5 闭环状态边界

- 当前已经支持 `mass_hat` 的保守闭环写回
- 但闭环仍然只更新 `payload_cfg["mass"]`
- 还没有估计 `com_hat`
- 也没有把 whole-body / torso / base / hand controller 纳入统一自适应框架

换句话说，当前阶段已经从纯验证链推进到了“最小可用闭环链”，但还不是最终形态的全闭环 payload adaptive compensation。

---

## 9. 这套方案背后的工程逻辑

如果从代码审查或技术设计角度概括，这次工作的工程逻辑可以总结为三点：

### 9.1 先把 payload 作为 ee 外载建模，而不是拆成手指局部问题

这样做抓住了问题的主物理本质，也让补偿落点自然对齐到 arm torque feedforward。

### 9.2 先做已知质量补偿，再补观测链，再做在线估计

这是一种渐进式扩展方法：

- 先验证建模对不对
- 再验证观测够不够
- 最后验证估计稳不稳

这比一步到位做自适应闭环更稳健，也更便于定位问题来源。

### 9.3 把算法目标与安全护栏拆开

通过保留：

- IK 层的理论 `sol_tauff`
- controller 层的 `tau` clip

你把“模型应该给多少”与“系统允许发多少”清晰分层了。这在实机开发里非常重要，因为它让算法调试与安全约束不互相污染。

---

## 10. 下一步最自然的演进方向

在当前结构已经稳定的前提下，下一步最自然的演进是：

1. 在实机上继续整定闭环参数，例如 `min_valid_updates / alpha / max_step`
2. 评估 `payload_scale` 是否仍需保留为调试保护项
3. 设计更系统的静态 / 准静态实验，验证不同姿态下的 `mass_hat` 与 `mass_cmd` 一致性
4. 视实验结果决定是否扩展到 `com_hat`
5. 在需要时再考虑 whole-body 层面的 payload coupling

这条路线的优点是：每一步都建立在上一阶段已经验证过的结构之上，而不是同时引入多个不确定因素。

---

## 11. 效果评估与可视化

为了把“补偿前 vs 补偿后”的效果做成具体数字和直观图像，我另外补了一套离线评估链路，见：

- [EVALUATION_zh-CN.md](/home/sicheng/xr_teleoperate/docs/g1_29_payload_compensation/EVALUATION_zh-CN.md)
- [payload_eval.py](/home/sicheng/xr_teleoperate/teleop/utils/payload_eval.py)

这套评估链路的核心能力是：

- 基于录制 episode 自动输出 before / after 的关键数字
- 自动生成 joint / ee 误差对比曲线
- 自动生成 HTML 报告
- 如果有图像，则在报告中插入关键帧对比
- 如果环境支持 `ffmpeg`，还可以额外生成 side-by-side 视频

它的目标就是把“感觉补偿后更稳了”这种主观印象，转化成：

- Hold 段末端位置误差下降多少 mm
- Hold 段末端姿态误差下降多少 deg
- 关节跟踪误差 RMS / P95 下降多少

这种可以直接写进汇报或审查结论的数字。

---

## 12. 参考仓库、代码与论文

这一节把本说明背后的参考来源明确列出来，并区分：

- 哪些是当前仓库中的直接实现依据
- 哪些是支撑当前实现的外部理论 / 工具文档

需要特别说明的是：

- `tau_payload = J^\top w_payload`
- `tau_nominal = rnea(q, 0, 0)` 的静态 / 准静态理解
- `LOCAL_WORLD_ALIGNED` 的坐标系含义

这些属于标准刚体动力学和机器人静力学范式，有明确文献和官方文档支撑。

而下面这些内容，则是你在本仓库里的**工程设计选择**，不是外部文献直接“规定”的：

- 只先支持 `G1_29`
- 只先做单手 active side payload
- 固定 `payload_com`，第一版只估 `mass_hat`
- 先打通补偿链、观测链、估计链，再考虑闭环回写
- 把 `tau` clip 放在 controller 层而不是 IK 层

也就是说，文献提供的是动力学和静力学基础，而当前这套三阶段方案是你在本仓库约束下做出的工程化集成设计。

### 11.1 本仓库内的直接实现参考

以下代码文件是这份说明直接对应的实现来源：

- 主脚本参数入口与主循环接线：
  [teleop_hand_and_arm.py](/home/sicheng/xr_teleoperate/teleop/teleop_hand_and_arm.py:85)
- `G1_29` IK 中的 payload torque 构造与 `sol_tauff` 叠加：
  [robot_arm_ik.py](/home/sicheng/xr_teleoperate/teleop/robot_control/robot_arm_ik.py:231)
- arm controller 中的 `tau_est` 观测与 `tau` 限幅：
  [robot_arm.py](/home/sicheng/xr_teleoperate/teleop/robot_control/robot_arm.py:149)
- 在线质量估计模块：
  [payload_estimator.py](/home/sicheng/xr_teleoperate/teleop/robot_control/payload_estimator.py:17)

如果后续别人要复核这份文档是否和当前实现一致，优先就应该看这四个位置。

### 11.2 外部参考仓库 / 官方文档 / 论文

#### 1. Pinocchio 官方仓库

- 仓库地址：
  https://github.com/stack-of-tasks/pinocchio

用途：

- 本项目里 `pin.rnea(...)`、`computeFrameJacobian(...)`、frame placement、URDF/reduced model 等核心接口都直接依赖 Pinocchio
- 这也是当前实现最直接的外部软件基础

#### 2. Pinocchio 官方文档首页

- 文档地址：
  https://stack-of-tasks.github.io/pinocchio/

用途：

- 官方文档明确列出了 Pinocchio 支持的核心刚体动力学算法
- 其中包括 Recursive Newton-Euler Algorithm (RNEA) 和 placement Jacobians
- 这正是当前 `tau_nominal` 和 ee Jacobian 计算的基础

#### 3. Pinocchio Frame / Jacobian 文档

- 文档地址：
  https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/jnrh2023/template/frame.html

用途：

- 用来确认 `getFrameJacobian / computeFrameJacobian` 的语义
- 用来确认 Jacobian 可以在 `LOCAL / LOCAL_WORLD_ALIGNED / WORLD` 三种参考系下表达
- 这直接支撑了当前实现中为什么选 `LOCAL_WORLD_ALIGNED`

和当前实现的对应关系是：

- 你构造的 payload 重力方向 `f = [0, 0, -mg]` 是在世界系里表达的
- 力矩 `m = r \times f` 也是在世界系方向下计算的
- 因此使用 `LOCAL_WORLD_ALIGNED`，可以让“ee 原点 + 世界系方向”这组表达保持一致

#### 4. Pinocchio `ReferenceFrame` 说明

- 文档地址：
  https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/group__pinocchio__multibody.html

用途：

- 官方文档明确说明了 `WORLD / LOCAL / LOCAL_WORLD_ALIGNED` 的含义
- 尤其 `LOCAL_WORLD_ALIGNED` 表示：原点在 moving frame，但量是投影到 world-aligned basis 中

这对解释为什么当前代码中既使用 ee frame 挂点，又使用世界系重力方向，非常关键。

#### 5. Pinocchio 软件论文

- 论文信息：
  Justin Carpentier, Guilhem Saurel, Gabriele Buondonno, Joseph Mirabel, Florent Lamiraux, Olivier Stasse, Nicolas Mansard,
  *The Pinocchio C++ library -- A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives*,
  IEEE International Symposium on System Integrations (SII), 2019.
- 仓库中的引用入口：
  https://github.com/stack-of-tasks/pinocchio

用途：

- 这篇论文是 Pinocchio 官方推荐引用的核心软件论文
- 如果你后续把这部分工作写成正式报告、论文或技术文档，这篇文献应该是引用 Pinocchio 最合适的来源之一

#### 6. Roy Featherstone, *Rigid Body Dynamics Algorithms*

- 图书页面：
  https://link.springer.com/book/10.1007/978-1-4899-7560-7
- DOI：
  https://doi.org/10.1007/978-1-4899-7560-7

用途：

- Pinocchio 官方就明确说明其多体动力学算法建立在 Featherstone 体系之上
- `RNEA`、空间向量、刚体动力学递推等内容，Featherstone 是最经典的基础来源之一
- 你这里把 `tau_nominal` 理解为基于刚体动力学模型得到的名义关节力矩，本质上就是在用这套范式

#### 7. Modern Robotics: Chapter 5, Velocity Kinematics and Statics

- 章节页面：
  https://modernrobotics.northwestern.edu/chapters/chapter5/
- 开源教材 PDF：
  https://hades.mech.northwestern.edu/images/2/2e/MR-largefont-v2.pdf

用途：

- 这个来源最适合支撑“为什么末端 wrench 可以通过 Jacobian transpose 映射到 joint torque”
- Chapter 5 明确覆盖了 Jacobian 与静力学关系
- 其中 statics of open chains 直接对应：
  \[
  \tau = J^\top F
  \]

对于这次文档里“payload 是 ee 外载，然后通过 `J^\top w` 进入关节空间”的解释，这是非常贴切的参考来源。

### 11.3 如何理解这些参考与当前实现的关系

如果要用一句话概括：

- Pinocchio + Featherstone 负责回答“怎么做刚体动力学和 Jacobian 计算”
- Modern Robotics 负责回答“为什么 ee wrench 能通过 `J^\top` 映射到 joint torque”
- 当前仓库代码负责回答“在 `G1_29` 这个具体遥操作项目里，应该把这套理论挂到哪里、分几阶段做、怎么留安全护栏和估计接口”

因此，这次工作是：

- 理论上依托标准机器人动力学 / 静力学框架
- 工程上在本仓库里做了一个分阶段、可验证、可继续闭环扩展的 payload compensation 结构

---

## 13. 一句话总结

这次改动的真正价值，不是“单独做了一个 payload estimator”，而是把 `G1_29` 的 payload 重力补偿能力，从**手工给定质量的静态补偿**，系统性地推进到了**具备在线质量估计接口和观测闭环基础的工程化结构**。
