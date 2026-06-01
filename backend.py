import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    episodes: int = 1000
    seed: int = 42
    C: float = 0.1
    F: float = 0.0
    initPGo: float = 0.85

def goReward(C, F):
    return 100 * C * F - 150 * (1 - C) - 100 * (1 - F) - 50

STAY_REWARD = 80

class SimplePolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.fc(x)

@app.post("/train/ppo")
def train_ppo(req: TrainRequest):
    torch.manual_seed(req.seed)
    
    policy = SimplePolicyNet()
    # Initialize to mimic initPGo
    with torch.no_grad():
        policy.fc.weight.fill_(0)
        logit_init = np.log(req.initPGo / (1 - req.initPGo))
        policy.fc.bias.fill_(logit_init)
        
    optimizer = optim.Adam(policy.parameters(), lr=0.08)
    state = torch.tensor([req.C, req.F], dtype=torch.float32)
    
    pGo_history = []
    reward_history = []
    Rgo = goReward(req.C, req.F)
    clip_eps = 0.2
    
    for _ in range(req.episodes):
        optimizer.zero_grad()
        p = torch.sigmoid(policy(state))
        
        a = 1 if torch.rand(1).item() < p.item() else 0
        r = Rgo if a == 1 else STAY_REWARD
        
        baseline = p.item() * Rgo + (1 - p.item()) * STAY_REWARD
        adv = r - baseline
        
        log_prob = torch.log(p) if a == 1 else torch.log(1 - p)
        loss = - (adv * log_prob)
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(policy.parameters(), clip_eps)
        optimizer.step()
        
        with torch.no_grad():
            new_p = torch.sigmoid(policy(state)).item()
            pGo_history.append(new_p)
            reward_history.append(new_p * Rgo + (1 - new_p) * STAY_REWARD)
            
    return {
        "pGo": pGo_history,
        "reward": reward_history,
        "finalPGo": pGo_history[-1],
        "finalReward": reward_history[-1]
    }

# Mock DPO, Q-Learning, RLHF endpoints
for algo in ["dpo", "ql", "rlhf"]:
    @app.post(f"/train/{algo}")
    def train_mock(req: TrainRequest):
        # We reuse PPO logic here for the 'dry-run' just to prove neural net backend works
        return train_ppo(req)

if __name__ == "__main__":
    print("🚀 [PyTorch Backend] RL Engine starting on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
