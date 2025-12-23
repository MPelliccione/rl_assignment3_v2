import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Policy(nn.Module):
    continuous = False

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        # Auto-select GPU if available
        if device == torch.device('cpu') and torch.cuda.is_available():
            device = torch.device('cuda')
        self.device = device

        # Actor (policy) network
        self.actor_conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.actor_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.actor_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.actor_fc1 = nn.Linear(64 * 8 * 8, 512)
        self.policy_head = nn.Linear(512, 5)

        # Critic (value) network
        self.critic_conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.critic_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.critic_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.critic_fc1 = nn.Linear(64 * 8 * 8, 512)
        self.value_head = nn.Linear(512, 1)
        
        # Hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.delta = 0.01  
        self.damping = 0.15
        self.value_lr = 1e-3

        # Move to device
        self.to(self.device)

    def forward(self, x):
        # This can be removed or used for shared logic if needed
        pass

    def actor_forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        x = F.relu(self.actor_conv1(x))
        x = F.relu(self.actor_conv2(x))
        x = F.relu(self.actor_conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.actor_fc1(x))
        return x

    def critic_forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        x = F.relu(self.critic_conv1(x))
        x = F.relu(self.critic_conv2(x))
        x = F.relu(self.critic_conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.critic_fc1(x))
        return x
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            features = self.actor_forward(state)
            logits = self.policy_head(features)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def _get_policy_params(self):
        """Update actor network + policy head"""
        return list(self.actor_conv1.parameters()) + list(self.actor_conv2.parameters()) + \
               list(self.actor_conv3.parameters()) + list(self.actor_fc1.parameters()) + \
               list(self.policy_head.parameters())

    def train(self):
        # Value optimizer updates critic network + value head
        value_optimizer = torch.optim.Adam(
            list(self.critic_conv1.parameters()) + list(self.critic_conv2.parameters()) +
            list(self.critic_conv3.parameters()) + list(self.critic_fc1.parameters()) +
            list(self.value_head.parameters()),
            lr=self.value_lr
        )
        
        env = gym.make('CarRacing-v2', continuous=False)
        
        num_iterations = 300
        steps_per_iter = 4096
        value_epochs = 20
        
        best_reward = -float('inf')
        
        for iteration in range(num_iterations):
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            state, _ = env.reset()
            episode_reward = 0
            episode_rewards = []
            old_probs = []
            
            for _ in range(steps_per_iter):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    actor_features = self.actor_forward(state_tensor)
                    critic_features = self.critic_forward(state_tensor)
                    logits = self.policy_head(actor_features)
                    value = self.value_head(critic_features)
                    probs = F.softmax(logits, dim=-1)
                    
                    action = torch.multinomial(probs, 1).squeeze()
                    log_prob = torch.log(probs.squeeze()[action] + 1e-8)
                
                states.append(state)
                actions.append(action.item())
                values.append(value.item())
                log_probs.append(log_prob.item())
                old_probs.append(probs.squeeze(0).cpu().numpy())
                
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_reward = 0
                    state, _ = env.reset()
                else:
                    state = next_state
            
            # Bootstrap
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                critic_features = self.critic_forward(state_tensor)
                last_value = self.value_head(critic_features).item()
            values.append(last_value)

            # GAE
            advantages, returns = self._compute_gae(rewards, values, dones)
            
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            advantages_t = torch.FloatTensor(advantages).to(self.device)
            returns_t = torch.FloatTensor(returns).to(self.device)
            old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
            old_probs_t = torch.FloatTensor(np.array(old_probs)).to(self.device)
            
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # TRPO policy update
            self._trpo_update(states_t, actions_t, advantages_t, old_log_probs_t, old_probs_t)
            
            # Value update
            for _ in range(value_epochs):
                batch_size = 256
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    
                    batch_states = states_t[batch_idx]
                    batch_returns = returns_t[batch_idx]
                    
                    critic_features = self.critic_forward(batch_states)
                    values_pred = self.value_head(critic_features)
                    value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()
            
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, Episodes: {len(episode_rewards)}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save()
            
            if iteration % 50 == 0:
                self.save()
        
        env.close()
        return
    
    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0  # Terminal state has no future value
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lam * gae * (1 - dones[t])
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def _compute_kl(self, old_probs, new_probs):
        # KL(old || new): old_probs from rollout (no grad), new_probs current (with grad)
        old_probs = torch.clamp(old_probs.detach(), 1e-8, 1.0)
        new_probs = torch.clamp(new_probs, 1e-8, 1.0)
        return (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=-1).mean()

    def _trpo_update(self, states, actions, advantages, old_log_probs, old_probs):
        policy_params = self._get_policy_params()
        
        actor_features = self.actor_forward(states)  # Use actor_forward
        logits = self.policy_head(actor_features)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate = (ratio * advantages).mean()
        
        grads = torch.autograd.grad(surrogate, policy_params, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        
        kl = self._compute_kl(old_probs, probs)
        
        def fvp(v):
            kl_grad = torch.autograd.grad(kl, policy_params, create_graph=True, retain_graph=True)
            flat_kl_grad = torch.cat([g.reshape(-1) for g in kl_grad])
            kl_v = (flat_kl_grad * v).sum()
            kl_grad_grad = torch.autograd.grad(kl_v, policy_params, retain_graph=True)
            return torch.cat([g.reshape(-1) for g in kl_grad_grad]) + self.damping * v
        
        step_dir = self._conjugate_gradient(fvp, flat_grad, iters=15)
        
        shs = torch.dot(step_dir, fvp(step_dir))
        if shs <= 0:
            return
        step_scale = torch.sqrt(2 * self.delta / (shs + 1e-8))
        full_step = step_dir * step_scale
        
        old_params = self._flat_params(policy_params).detach().clone()
        old_surrogate = surrogate.item()
        self._line_search(actor_features, actions, advantages, old_log_probs,  # Pass actor_features
                          full_step, policy_params, old_params, old_probs, old_surrogate)

    def _line_search(self, actor_features, actions, advantages, old_log_probs,  # Accept actor_features
                     full_step, params, old_params, old_probs, old_surrogate, max_backtracks=10):
        self._set_flat_params(params, old_params)
        for step_frac in [0.5 ** i for i in range(max_backtracks)]:
            new_params = old_params + step_frac * full_step
            self._set_flat_params(params, new_params)
            with torch.no_grad():
                logits = self.policy_head(actor_features)  # Use actor_features
                new_probs = F.softmax(logits, dim=-1)
                new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
                ratio = torch.exp(new_log_probs - old_log_probs)
                new_surrogate = (ratio * advantages).mean().item()
                kl = self._compute_kl(old_probs, new_probs).item()
            if kl < self.delta and new_surrogate > old_surrogate:
                return
        self._set_flat_params(params, old_params)

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
    
    def _flat_params(self, params):
        return torch.cat([p.reshape(-1) for p in params])

    def _set_flat_params(self, params, flat):
        offset = 0
        with torch.no_grad():
            for p in params:
                numel = p.numel()
                p.data.copy_(flat[offset:offset+numel].reshape(p.shape))
                offset += numel

    def get_policy(self, states):
        if isinstance(states, torch.Tensor):
            state_tensor = states.to(self.device)
        else:
            state_tensor = torch.FloatTensor(states).to(self.device)
        # use actor_forward (not forward which is a no-op)
        features = self.actor_forward(state_tensor)
        logits = self.policy_head(features)
        return F.softmax(logits, dim=-1)
    
    def _conjugate_gradient(self, fvp, b, iters=10, tol=1e-10):
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
