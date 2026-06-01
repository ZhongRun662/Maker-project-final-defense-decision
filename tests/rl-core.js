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
  
  async function fetchBackend(algo, opts) {
    const o = cfg(opts, {});
    const payload = {
      episodes: o.episodes,
      seed: o.seed || 42,
      C: o.C,
      F: o.F,
      initPGo: o.initPGo
    };
    try {
      const res = await fetch(`http://127.0.0.1:8000/train/${algo}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      return await res.json();
    } catch(e) {
      console.error('Error fetching backend:', e);
      throw e;
    }
  }

  async function trainPPO(opts) { return fetchBackend('ppo', opts); }
  async function trainDPO(opts) { return fetchBackend('dpo', Object.assign({}, opts, { seed: opts && opts.seed != null ? opts.seed : 13 })); }
  async function trainQLearning(opts) { return fetchBackend('ql', Object.assign({}, opts, { seed: opts && opts.seed != null ? opts.seed : 7 })); }
  async function trainRLHF(opts) { return fetchBackend('rlhf', Object.assign({}, opts, { seed: opts && opts.seed != null ? opts.seed : 99 })); }

  async function runAll(opts) {
    const o = opts || {};
    const ppo = await trainPPO(Object.assign({}, o, { seed: o.seed != null ? o.seed : 42 }));
    const dpo = await trainDPO(Object.assign({}, o, { seed: o.seed != null ? o.seed + 1 : 13 }));
    const ql = await trainQLearning(Object.assign({}, o, { seed: o.seed != null ? o.seed + 2 : 7 }));
    const rlhf = await trainRLHF(Object.assign({}, o, { seed: o.seed != null ? o.seed + 3 : 99 }));
    return { ppo, dpo, ql, rlhf };
  }

  return {
    sigmoid, logit, clamp, mulberry32,
    STAY_REWARD, goReward, rewardFor, surfaceData,
    trainPPO, trainDPO, trainQLearning, trainRLHF, runAll,
  };
});
