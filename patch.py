import re

def update_rl_core():
    with open('tests/rl-core.js', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find where function trainPPO starts and replace from there to the pack function
    start_str = "function trainPPO(opts) {"
    end_str = "return {\n    sigmoid"
    
    if start_str not in content or end_str not in content:
        print("Could not find boundaries in rl-core.js")
        return
        
    start_idx = content.find(start_str)
    # Actually, let's keep the comments if we want, or just replace the logic
    # Find the comment "/** PPO "
    # It's safer to use regex or string replace.
    
    # Let's replace the whole section
    before_logic = content[:start_idx]
    
    after_logic = content[content.find(end_str):]
    
    new_logic = """
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

  """
    
    # Write back to rl-core.js
    new_content = before_logic + new_logic + after_logic
    with open('tests/rl-core.js', 'w', encoding='utf-8') as f:
        f.write(new_content)

    # Now update index.html mirror
    with open('index.html', 'r', encoding='utf-8') as f:
        html = f.read()

    # In index.html, we need to replace the mirror and also the handler calling it
    start_html = html.find("function trainPPO(opts) {")
    end_html = html.find("return { sigmoid, logit, clamp, mulberry32, STAY_REWARD")
    
    if start_html != -1 and end_html != -1:
        before_html = html[:start_html]
        after_html = html[end_html:]
        new_html = before_html + new_logic + after_html
    else:
        new_html = html
        
    # Also update index.html calling logic from:
    # const all = RLCore.runAll({ episodes: 1000, initPGo: 0.85 });
    # to:
    # const all = await RLCore.runAll({ episodes: 1000, initPGo: 0.85 });
    new_html = new_html.replace("const all = RLCore.runAll", "const all = await RLCore.runAll")
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(new_html)
        
    # Now update tests/rl-core.test.js
    with open('tests/rl-core.test.js', 'r', encoding='utf-8') as f:
        tests = f.read()
        
    tests = tests.replace("test(`${name}: 输出结构与确定性`, () => {", "test(`${name}: 输出结构与确定性`, async () => {")
    tests = tests.replace("const r1 = run();", "const r1 = await run();")
    tests = tests.replace("const r2 = run();", "const r2 = await run();")
    
    tests = tests.replace("test(`${name}: 把「去答辩」概率从 85% 收敛到接近 0`, () => {", "test(`${name}: 把「去答辩」概率从 85% 收敛到接近 0`, async () => {")
    tests = tests.replace("const r = rl[name]({", "const r = await rl[name]({")
    
    tests = tests.replace("test(`${name}: 期望收益从负数深渊回升并稳定在 +80 附近`, () => {", "test(`${name}: 期望收益从负数深渊回升并稳定在 +80 附近`, async () => {")

    tests = tests.replace("test('PPO 与 Q-Learning 收敛更彻底（P(去) < 1%）', () => {", "test('PPO 与 Q-Learning 收敛更彻底（P(去) < 1%）', async () => {")
    tests = tests.replace("approx(rl.trainPPO({ seed: 2025 }).finalPGo, 0, 0.01);", "approx((await rl.trainPPO({ seed: 2025 })).finalPGo, 0, 0.01);")
    tests = tests.replace("approx(rl.trainQLearning({ seed: 2025 }).finalPGo, 0, 0.01);", "approx((await rl.trainQLearning({ seed: 2025 })).finalPGo, 0, 0.01);")

    tests = tests.replace("test('runAll 同时跑四种算法用于横向对比', () => {", "test('runAll 同时跑四种算法用于横向对比', async () => {")
    tests = tests.replace("const all = rl.runAll({ episodes: 1000, seed: 2025 });", "const all = await rl.runAll({ episodes: 1000, seed: 2025 });")
    
    with open('tests/rl-core.test.js', 'w', encoding='utf-8') as f:
        f.write(tests)

update_rl_core()
