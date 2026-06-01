'use strict';
/**
 * TDD 规格：决策 RL 核心引擎 (rl-core.js)
 * 运行： node --test   （或 npm test）
 *
 * 这些断言定义了「要不要去给烂尾项目擦屁股答辩」这道决策题的数学事实：
 *   - 去答辩 (A=1) 的奖励曲面在团队真实坐标 (C=0.1, F=0) 处深达 -285（死局）。
 *   - 不去 (A=0) 的机会价值恒为 +80（复习期末 / 搞开源 / 保命）。
 *   - 四种对齐算法 (PPO / DPO / Q-Learning / RLHF) 都必须把「去」的概率从初始 85%
 *     的愧疚冲动，理性收敛到接近 0%，期望收益从负数深渊回升到 +80 平稳期。
 */

const test = require('node:test');
const assert = require('node:assert');
const rl = require('./rl-core.js');

const approx = (a, b, eps = 1e-6) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b} (±${eps})`);

// ---------------------------------------------------------------------------
// 数学基元
// ---------------------------------------------------------------------------
test('sigmoid 基本性质', () => {
  approx(rl.sigmoid(0), 0.5);
  assert.ok(rl.sigmoid(50) > 0.999999);
  assert.ok(rl.sigmoid(-50) < 1e-6);
  assert.ok(rl.sigmoid(1) > rl.sigmoid(0)); // 单调递增
});

test('logit 是 sigmoid 的反函数', () => {
  for (const p of [0.1, 0.5, 0.85, 0.99]) approx(rl.sigmoid(rl.logit(p)), p);
});

test('mulberry32 是确定性可复现 PRNG，输出落在 [0,1)', () => {
  const a = rl.mulberry32(12345);
  const b = rl.mulberry32(12345);
  for (let i = 0; i < 1000; i++) {
    const x = a();
    assert.strictEqual(x, b(), '相同种子必须产出相同序列');
    assert.ok(x >= 0 && x < 1, '取值必须在 [0,1)');
  }
  // 不同种子应给出不同序列
  assert.notStrictEqual(rl.mulberry32(1)(), rl.mulberry32(2)());
});

// ---------------------------------------------------------------------------
// 奖励模型
// ---------------------------------------------------------------------------
test('STAY_REWARD（不去的机会价值）= 80', () => {
  assert.strictEqual(rl.STAY_REWARD, 80);
});

test('goReward 奖励曲面在关键坐标的精确取值', () => {
  approx(rl.goReward(0.1, 0), -285); // 团队真实位置：死局
  approx(rl.goReward(0, 0), -300);   // 全局最深渊（代码与人都归零）
  approx(rl.goReward(1, 1), 50);     // 理想国（满分代码 + 满员）
  approx(rl.goReward(1, 0), -150);
  approx(rl.goReward(0, 1), -200);
});

test('rewardFor 按动作分发奖励', () => {
  approx(rl.rewardFor(1, 0.1, 0), -285); // 去
  approx(rl.rewardFor(0, 0.1, 0), 80);   // 不去
});

// ---------------------------------------------------------------------------
// 3D 曲面数据
// ---------------------------------------------------------------------------
test('surfaceData 生成完整网格并定位深渊 / 巅峰', () => {
  const steps = 40;
  const s = rl.surfaceData({ steps });
  assert.strictEqual(s.grid.length, (steps + 1) * (steps + 1));
  assert.ok(s.grid.every((p) => p.length === 3), '每个点必须是 [x,y,z]');
  approx(s.min.z, -300);                 // 全局最低
  approx(s.max.z, 50);                    // 全局最高
  approx(s.abyss.z, -285);                // 团队死局坐标
  assert.strictEqual(s.abyss.x, 0.1);
  assert.strictEqual(s.abyss.y, 0);
});

// ---------------------------------------------------------------------------
// 强化学习 / 对齐 训练器：通用收敛契约
// ---------------------------------------------------------------------------
const TRAINERS = ['trainPPO', 'trainDPO', 'trainQLearning', 'trainRLHF'];

for (const name of TRAINERS) {
  test(`${name}: 输出结构与确定性`, () => {
    const run = () => rl[name]({ episodes: 1000, seed: 2025 });
    const r1 = run();
    const r2 = run();
    assert.strictEqual(r1.pGo.length, 1000);
    assert.strictEqual(r1.reward.length, 1000);
    assert.deepStrictEqual(r1.pGo, r2.pGo, '相同种子必须完全可复现');
    assert.deepStrictEqual(r1.reward, r2.reward);
  });

  test(`${name}: 把「去答辩」概率从 85% 收敛到接近 0`, () => {
    const r = rl[name]({ episodes: 1000, seed: 2025, initPGo: 0.85 });
    assert.ok(r.pGo[0] > 0.5, '初始应保留愧疚冲动 (>50%)');
    assert.ok(r.finalPGo < 0.05, `${name} 末态 P(去)=${r.finalPGo} 应 < 5%`);
    assert.ok(r.finalPGo < r.pGo[0], '概率必须整体下降');
  });

  test(`${name}: 期望收益从负数深渊回升并稳定在 +80 附近`, () => {
    const r = rl[name]({ episodes: 1000, seed: 2025 });
    assert.ok(r.reward[0] < 0, '起点必须在负收益深渊');
    assert.ok(r.finalReward > 78, `末态收益 ${r.finalReward} 应逼近 STAY_REWARD(80)`);
    assert.ok(r.finalReward <= rl.STAY_REWARD + 1e-6);
    assert.ok(r.finalReward > r.reward[0], '收益必须整体上升');
  });
}

test('PPO 与 Q-Learning 收敛更彻底（P(去) < 1%）', () => {
  approx(rl.trainPPO({ seed: 2025 }).finalPGo, 0, 0.01);
  approx(rl.trainQLearning({ seed: 2025 }).finalPGo, 0, 0.01);
});

// ---------------------------------------------------------------------------
// 多算法对比入口
// ---------------------------------------------------------------------------
test('runAll 同时跑四种算法用于横向对比', () => {
  const all = rl.runAll({ episodes: 1000, seed: 2025 });
  for (const k of ['ppo', 'dpo', 'ql', 'rlhf']) {
    assert.ok(all[k], `缺少算法结果: ${k}`);
    assert.strictEqual(all[k].reward.length, 1000);
    assert.ok(all[k].finalPGo < 0.05);
  }
});
