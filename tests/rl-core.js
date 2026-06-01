/* =====================================================================
 * rl-core.js  ——  「烂尾项目答辩决策」强化学习核心引擎（纯函数 / 无副作用）
 * ---------------------------------------------------------------------
 * UMD 模块：同时可被 Node (require) 与浏览器 (<script> -> window.RLCore) 加载。
 *
 * !!! 单一事实源 (single source of truth) !!!
 * index.html 内联了本文件的逐字镜像（搜索标记: ===== RL CORE MIRROR =====）。
 * 修改本文件后，必须同步更新 index.html 中的镜像块，并重新 `npm test`。
 *
 * 决策建模：
 *   状态 (固定)   ：代码质量 C=0.1，核心在校人员可用度 F=0
 *   动作          ：A=1 去答辩 / A=0 不去
 *   去的奖励曲面  ：Z = 100·C·F − 150·(1−C) − 100·(1−F) − 50
 *                   → (0.1, 0) 处 = −285，即「死局」深渊
 *   不去的奖励    ：恒为 +80（复习期末 / 搞 GitHub 开源 / 保住头发的机会价值）
 * ===================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.RLCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------- 数学基元 -------------------------------------------------------
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const logit = (p) => Math.log(p / (1 - p));
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  /** mulberry32：32 位确定性 PRNG，给定种子产出可复现序列，便于测试与回放。 */
  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ------- 奖励模型 -------------------------------------------------------
  const STAY_REWARD = 80; // 不去答辩的机会价值（正向）

  /** 去答辩的奖励曲面，X=代码质量 Y=人员在校度 ∈ [0,1] */
  function goReward(C, F) {
    return 100 * C * F - 150 * (1 - C) - 100 * (1 - F) - 50;
  }

  function rewardFor(action, C, F) {
    return action === 1 ? goReward(C, F) : STAY_REWARD;
  }

  /** 生成 3D 曲面网格 + 关键点（全局最低 / 最高 / 团队死局坐标） */
  function surfaceData(opts) {
    const steps = (opts && opts.steps) || 40;
    const grid = [];
    let min = { x: 0, y: 0, z: Infinity };
    let max = { x: 0, y: 0, z: -Infinity };
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      for (let j = 0; j <= steps; j++) {
        const y = j / steps;
        const z = goReward(x, y);
        grid.push([x, y, z]);
        if (z < min.z) min = { x, y, z };
        if (z > max.z) max = { x, y, z };
      }
    }
    return { grid, min, max, abyss: { x: 0.1, y: 0, z: goReward(0.1, 0) } };
  }

  // ------- 训练器通用参数解析 --------------------------------------------
  function cfg(opts, defaults) {
    return Object.assign({ episodes: 1000, C: 0.1, F: 0, initPGo: 0.85 }, defaults, opts);
  }

  /** PPO 风格的裁剪策略梯度（单状态 bandit 简化版）。
   *  以价值基线计算优势，按 clipEps 做信任域裁剪，避免单步策略突变。 */
  function trainPPO(opts) {
    const o = cfg(opts, { lr: 0.08, clipEps: 0.2, seed: 42 });
    const rng = mulberry32(o.seed);
    const Rgo = goReward(o.C, o.F);
    let theta = logit(o.initPGo);
    const pGo = [], reward = [];
    for (let t = 0; t < o.episodes; t++) {
      const p = sigmoid(theta);
      const a = rng() < p ? 1 : 0;                 // on-policy 采样
      const r = a === 1 ? Rgo : STAY_REWARD;
      const baseline = p * Rgo + (1 - p) * STAY_REWARD;
      const adv = r - baseline;                    // 优势
      const gradLog = a === 1 ? 1 - p : -p;        // ∂logπ(a)/∂θ
      theta += clamp(o.lr * adv * gradLog, -o.clipEps, o.clipEps);
      const pN = sigmoid(theta);
      pGo.push(pN);
      reward.push(pN * Rgo + (1 - pN) * STAY_REWARD);
    }
    return pack(pGo, reward, { theta });
  }

  /** DPO：直接偏好优化。偏好对固定为「不去 ≻ 去」，参考策略取均匀分布。
   *  推导可得 ∂L/∂θ = β·(1 − σ(−βθ))，沿负梯度即把 P(去) 压向 0。 */
  function trainDPO(opts) {
    const o = cfg(opts, { beta: 1.0, lr: 0.25, seed: 13 });
    const Rgo = goReward(o.C, o.F);
    let theta = logit(o.initPGo);
    const pGo = [], reward = [];
    for (let t = 0; t < o.episodes; t++) {
      const grad = o.beta * (1 - sigmoid(-o.beta * theta));
      theta -= o.lr * grad;
      const p = sigmoid(theta);
      pGo.push(p);
      reward.push(p * Rgo + (1 - p) * STAY_REWARD);
    }
    return pack(pGo, reward, { theta });
  }

  /** 表格型 Q-Learning（单状态 → 退化为 2 臂老虎机）。
   *  ε-贪心行为策略，软最大化导出展示用概率。Q[去]→−285，Q[不去]→80。 */
  function trainQLearning(opts) {
    const o = cfg(opts, { alpha: 0.1, epsilon: 0.1, tau: 60, seed: 7 });
    const rng = mulberry32(o.seed);
    const Rgo = goReward(o.C, o.F);
    let Qstay = 0;
    let Qgo = o.tau * logit(o.initPGo); // 让初始 softmax 概率 ≈ initPGo
    const pGo = [], reward = [];
    for (let t = 0; t < o.episodes; t++) {
      let a;
      if (rng() < o.epsilon) a = rng() < 0.5 ? 1 : 0; // 探索
      else a = Qgo > Qstay ? 1 : 0;                   // 利用
      const r = a === 1 ? Rgo : STAY_REWARD;
      if (a === 1) Qgo += o.alpha * (r - Qgo);
      else Qstay += o.alpha * (r - Qstay);
      const p = sigmoid((Qgo - Qstay) / o.tau);
      pGo.push(p);
      reward.push(p * Rgo + (1 - p) * STAY_REWARD);
    }
    return pack(pGo, reward, { Q: [Qstay, Qgo] });
  }

  /** RLHF：奖励模型 + PPO。真实奖励经过带高斯噪声的「奖励模型」估计，
   *  路径更抖，但强负信号仍把策略拉向「不去」。报告的是真实期望收益。 */
  function trainRLHF(opts) {
    const o = cfg(opts, { lr: 0.08, clipEps: 0.2, rmNoise: 8, seed: 99 });
    const rng = mulberry32(o.seed);
    const Rgo = goReward(o.C, o.F);
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());
    let theta = logit(o.initPGo);
    const pGo = [], reward = [];
    for (let t = 0; t < o.episodes; t++) {
      const p = sigmoid(theta);
      const a = rng() < p ? 1 : 0;
      const trueR = a === 1 ? Rgo : STAY_REWARD;
      const rmR = trueR + o.rmNoise * gauss();      // 奖励模型的含噪估计
      const baseline = p * Rgo + (1 - p) * STAY_REWARD;
      const adv = rmR - baseline;
      const gradLog = a === 1 ? 1 - p : -p;
      theta += clamp(o.lr * adv * gradLog, -o.clipEps, o.clipEps);
      const pN = sigmoid(theta);
      pGo.push(pN);
      reward.push(pN * Rgo + (1 - pN) * STAY_REWARD); // 报告真实期望收益
    }
    return pack(pGo, reward, { theta });
  }

  function pack(pGo, reward, extra) {
    return Object.assign(
      {
        pGo,
        reward,
        finalPGo: pGo[pGo.length - 1],
        finalReward: reward[reward.length - 1],
      },
      extra || {}
    );
  }

  /** 一键横向对比四种对齐算法。 */
  function runAll(opts) {
    const o = opts || {};
    return {
      ppo: trainPPO(Object.assign({}, o, { seed: o.seed != null ? o.seed : 42 })),
      dpo: trainDPO(Object.assign({}, o, { seed: o.seed != null ? o.seed + 1 : 13 })),
      ql: trainQLearning(Object.assign({}, o, { seed: o.seed != null ? o.seed + 2 : 7 })),
      rlhf: trainRLHF(Object.assign({}, o, { seed: o.seed != null ? o.seed + 3 : 99 })),
    };
  }

  return {
    sigmoid, logit, clamp, mulberry32,
    STAY_REWARD, goReward, rewardFor, surfaceData,
    trainPPO, trainDPO, trainQLearning, trainRLHF, runAll,
  };
});
