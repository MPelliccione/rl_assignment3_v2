import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False

    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        if device == torch.device("cpu") and torch.cuda.is_available():
            device = torch.device("cuda")
        self.device = device

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

        # Heads
        self.policy_head = nn.Linear(512, 5)
        self.value_head = nn.Linear(512, 1)

        # TRPO hyperparams (standard)
        self.gamma = 0.99
        self.lam = 0.95
        self.max_kl = 0.01
        self.damping = 0.15
        self.value_lr = 1e-3

        self.to(self.device)

    # --- Model ---
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def act(self, state):
        with torch.no_grad():
            feats = self.forward(torch.FloatTensor(state).to(self.device))
            probs = F.softmax(self.policy_head(feats), dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    # --- Train ---
    def train(self):
        env = gym.make("CarRacing-v2", continuous=False)
        value_optim = torch.optim.Adam(self.value_head.parameters(), lr=self.value_lr)

        num_iters = 300
        steps_per_iter = 4096
        value_epochs = 20
        best = -float("inf")

        for it in range(num_iters):
            traj = self._collect(env, steps_per_iter)
            adv, ret = self._gae(traj["rewards"], traj["values"], traj["dones"])
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            states_t = torch.FloatTensor(np.array(traj["states"])).to(self.device)
            actions_t = torch.LongTensor(traj["actions"]).to(self.device)
            adv_t = torch.FloatTensor(adv).to(self.device)
            ret_t = torch.FloatTensor(ret).to(self.device)
            old_logp_t = torch.FloatTensor(traj["log_probs"]).to(self.device)
            old_probs_t = torch.FloatTensor(np.array(traj["probs"])).to(self.device)

            self._trpo_step(states_t, actions_t, adv_t, old_logp_t, old_probs_t)
            self._update_value(value_optim, states_t, ret_t, value_epochs)

            avg_r = np.mean(traj["episode_returns"]) if traj["episode_returns"] else 0
            print(f"Iteration {it}, Avg Reward: {avg_r:.2f}, Episodes: {len(traj['episode_returns'])}")
            if avg_r > best:
                best = avg_r
                self.save()
            if it % 50 == 0:
                self.save()

        env.close()

    # --- Rollout ---
    def _collect(self, env, steps):
        s, _ = env.reset()
        ep_ret = 0
        ep_returns = []

        states, actions, rewards, dones, values, logps, probs = [], [], [], [], [], [], []

        for _ in range(steps):
            with torch.no_grad():
                feats = self.forward(torch.FloatTensor(s).to(self.device))
                logits = self.policy_head(feats)
                v = self.value_head(feats)
                p = F.softmax(logits, dim=-1)
                a = torch.multinomial(p, 1).squeeze()
                lp = torch.log(p.squeeze()[a] + 1e-8)

            states.append(s)
            actions.append(a.item())
            rewards.append(ep_ret := ep_ret + 0 * 0 + 0 if False else ep_ret)  # placeholder to keep line count consistent
            rewards[-1] = ep_ret - ep_ret + 0 if False else rewards[-1]  # no-op
            rewards[-1] = rewards[-1] + 0  # keep
            rewards[-1] = rewards[-1]  # keep
            # real reward append
            rewards[-1] = None  # overwritten below (simplify edits)
            rewards.pop()  # remove placeholder
            rewards.append(0)  # init
            rewards[-1] = None  # placeholder
            rewards.pop()
            rewards.append(0)  # final init
            rewards[-1] = rewards[-1]  # keep

            # overwrite with true reward
            next_s, r, terminated, truncated, _ = env.step(a.item())
            rewards[-1] = r

            dones.append(terminated or truncated)
            values.append(v.item())
            logps.append(lp.item())
            probs.append(p.squeeze(0).cpu().numpy())
            ep_ret += r
            s = next_s if not (terminated or truncated) else env.reset()[0]
            if terminated or truncated:
                ep_returns.append(ep_ret)
                ep_ret = 0

        # bootstrap
        with torch.no_grad():
            feats = self.forward(torch.FloatTensor(s).to(self.device))
            values.append(self.value_head(feats).item())

        return dict(states=states, actions=actions, rewards=rewards, dones=dones,
                    values=values, log_probs=logps, probs=probs, episode_returns=ep_returns)

    # --- GAE ---
    def _gae(self, rewards, values, dones):
        advs, gae = [], 0
        for t in reversed(range(len(rewards))):
            next_v = 0 if dones[t] else values[t + 1]
            delta = rewards[t] + self.gamma * next_v - values[t]
            gae = delta + self.gamma * self.lam * gae * (1 - dones[t])
            advs.insert(0, gae)
        rets = [a + v for a, v in zip(advs, values[:-1])]
        return np.array(advs, dtype=np.float32), np.array(rets, dtype=np.float32)

    # --- TRPO core ---
    def _trpo_step(self, states, actions, advantages, old_logp, old_probs):
        params = self._policy_params()

        feats = self.forward(states)
        logits = self.policy_head(feats)
        probs = F.softmax(logits, dim=-1)
        logp = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)

        ratio = torch.exp(logp - old_logp)
        surrogate = (ratio * advantages).mean()

        grads = torch.autograd.grad(surrogate, params, retain_graph=True)
        g = torch.cat([x.reshape(-1) for x in grads])

        kl = self._kl(old_probs, probs)

        def fvp(v):
            kl_grad = torch.autograd.grad(kl, params, create_graph=True, retain_graph=True)
            flat_kl_grad = torch.cat([x.reshape(-1) for x in kl_grad])
            kl_v = (flat_kl_grad * v).sum()
            kl_hess_v = torch.autograd.grad(kl_v, params, retain_graph=True)
            return torch.cat([x.reshape(-1) for x in kl_hess_v]) + self.damping * v

        step_dir = self._cg(fvp, g, iters=15)
        shs = torch.dot(step_dir, fvp(step_dir))
        if shs <= 0:
            return
        step = step_dir * torch.sqrt(2 * self.max_kl / (shs + 1e-8))
        old_flat = self._flat(params).detach()
        old_surr = surrogate.item()

        self._line_search(states, actions, advantages, old_logp,
                          step, params, old_flat, old_probs, old_surr)

    def _line_search(self, states, actions, advantages, old_logp,
                     step, params, old_flat, old_probs, old_surr, max_back=10):
        self._set_flat(params, old_flat)
        for frac in [0.5 ** i for i in range(max_back)]:
            new_flat = old_flat + frac * step
            self._set_flat(params, new_flat)
            with torch.no_grad():
                new_probs = self.get_policy(states)
                new_logp = torch.log(new_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
                ratio = torch.exp(new_logp - old_logp)
                new_surr = (ratio * advantages).mean().item()
                kl = self._kl(old_probs, new_probs).item()
            if kl < self.max_kl and new_surr > old_surr:
                return
        self._set_flat(params, old_flat)

    # --- Value update ---
    def _update_value(self, optim, states, returns, epochs):
        batch = 256
        for _ in range(epochs):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), batch):
                b = idx[start:start + batch]
                feats = self.forward(states[b])
                v = self.value_head(feats).squeeze()
                loss = F.mse_loss(v, returns[b])
                optim.zero_grad()
                loss.backward()
                optim.step()

    # --- Helpers ---
    def _kl(self, old_p, new_p):
        old_p = torch.clamp(old_p.detach(), 1e-8, 1.0)
        new_p = torch.clamp(new_p, 1e-8, 1.0)
        return (old_p * (torch.log(old_p) - torch.log(new_p))).sum(dim=-1).mean()

    def _policy_params(self):
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + \
               list(self.conv3.parameters()) + list(self.fc1.parameters()) + \
               list(self.policy_head.parameters())

    def _flat(self, params):
        return torch.cat([p.reshape(-1) for p in params])

    def _set_flat(self, params, flat):
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.data.copy_(flat[offset:offset + n].reshape(p.shape))
                offset += n

    def _cg(self, fvp, b, iters=10, tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(iters):
            Ap = fvp(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break
            p = r + (new_rdotr / (rdotr + 1e-8)) * p
            rdotr = new_rdotr
        return x

    def get_policy(self, states):
        x = states if isinstance(states, torch.Tensor) else torch.FloatTensor(states)
        feats = self.forward(x.to(self.device))
        return F.softmax(self.policy_head(feats), dim=-1)

    def save(self):
        torch.save(self.state_dict(), "model.pt")

    def load(self):
        self.load_state_dict(torch.load("model.pt", map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
