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

        # CNN for processing 96x96x3 images
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # Policy head (5 discrete actions for CarRacing)
        self.policy_head = nn.Linear(512, 5)

        # Value head
        self.value_head = nn.Linear(512, 1)
        
        # TRPO hyperparameters
        self.gamma = 0.99
        self.lam = 0.95
        self.max_kl = 0.01
        self.damping = 0.1
        self.value_lr = 1e-3
        
        # Store last action
        self.last_action = None
        
        # Move to device
        self.to(self.device)

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
            state = torch.FloatTensor(state).to(self.device)
            features = self.forward(state)
            logits = self.policy_head(features)
            probs = F.softmax(logits, dim=-1)
            self.last_action = torch.multinomial(probs, 1).item()
        return 
    
    def get_action(self):
        return self.last_action

    def _get_policy_params(self):
        """Get all parameters that affect the policy (CNN + policy head)"""
        return list(self.conv1.parameters()) + list(self.conv2.parameters()) + \
               list(self.conv3.parameters()) + list(self.fc1.parameters()) + \
               list(self.policy_head.parameters())

    def train(self):
        # Separate optimizer for value function only
        value_optimizer = torch.optim.Adam(self.value_head.parameters(), lr=self.value_lr)
        
        env = gym.make('CarRacing-v2', continuous=False)
        
        num_iterations = 500
        steps_per_iter = 4096
        value_epochs = 10
        
        best_reward = -float('inf')
        
        for iteration in range(num_iterations):
            states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
            state, _ = env.reset()
            episode_reward = 0
            episode_rewards = []
            
            for _ in range(steps_per_iter):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    features = self.forward(state_tensor)
                    logits = self.policy_head(features)
                    value = self.value_head(features)
                    probs = F.softmax(logits, dim=-1)
                    
                    action = torch.multinomial(probs, 1).squeeze()
                    log_prob = torch.log(probs.squeeze()[action] + 1e-8)
                
                states.append(state)
                actions.append(action.item())
                values.append(value.item())
                log_probs.append(log_prob.item())
                
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
            
            # Calculate value of the last state for bootstrapping
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                features = self.forward(state_tensor)
                last_value = self.value_head(features).item()
            
            # Append bootstrap value to values list
            values.append(last_value)

            # Compute GAE
            advantages, returns = self._compute_gae(rewards, values, dones)
            
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            advantages_t = torch.FloatTensor(advantages).to(self.device)
            returns_t = torch.FloatTensor(returns).to(self.device)
            old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
            
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # TRPO policy update
            self._trpo_update(states_t, actions_t, advantages_t, old_log_probs_t)
            
            # Update value function
            for _ in range(value_epochs):
                batch_size = 256
                indices = np.random.permutation(len(states))
                for start in range(0, len(states), batch_size):
                    end = start + batch_size
                    batch_idx = indices[start:end]
                    
                    batch_states = states_t[batch_idx]
                    batch_returns = returns_t[batch_idx]
                    
                    features = self.forward(batch_states)
                    values_pred = self.value_head(features)
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
        
        # Only
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns
    
    def _trpo_update(self, states, actions, advantages, old_log_probs):
        policy_params = self._get_policy_params()
        
        features = self.forward(states)
        logits = self.policy_head(features)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()
        
        grads = torch.autograd.grad(policy_loss, policy_params, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        
        old_probs = probs.detach()
        kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1).mean()
        
        def fvp(v):
            kl_grad = torch.autograd.grad(kl, policy_params, create_graph=True, retain_graph=True)
            flat_kl_grad = torch.cat([g.reshape(-1) for g in kl_grad])
            kl_v = (flat_kl_grad * v).sum()
            kl_grad_grad = torch.autograd.grad(kl_v, policy_params, retain_graph=True)
            return torch.cat([g.reshape(-1) for g in kl_grad_grad]) + self.damping * v
        
        step_dir = self._conjugate_gradient(fvp, flat_grad)
        
        shs = 0.5 * (step_dir * fvp(step_dir)).sum()
        lm = torch.sqrt(shs / self.max_kl + 1e-8)
        full_step = step_dir / (lm + 1e-8)
        
        self._line_search(states, actions, advantages, old_log_probs, full_step, policy_params)

    def _conjugate_gradient(self, fvp, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            Ap = fvp(p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr
        return x

    def _line_search(self, states, actions, advantages, old_log_probs, full_step, params, max_backtracks=10):
        with torch.no_grad():
            old_params = torch.cat([p.reshape(-1) for p in params])
            
            features = self.forward(states)
            logits = self.policy_head(features)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            ratio = torch.exp(log_probs - old_log_probs)
            old_loss = -(ratio * advantages).mean()
            
            for step_frac in [0.5 ** i for i in range(max_backtracks)]:
                new_params = old_params - step_frac * full_step
                
                offset = 0
                for p in params:
                    numel = p.numel()
                    p.copy_(new_params[offset:offset + numel].reshape(p.shape))
                    offset += numel
                
                features = self.forward(states)
                logits = self.policy_head(features)
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                new_loss = -(ratio * advantages).mean()
                
                old_probs_batch = F.softmax(logits.detach(), dim=-1)
                kl = (old_probs_batch * (torch.log(old_probs_batch + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1).mean()
                
                if kl < self.max_kl and new_loss < old_loss:
                    return
            
            offset = 0
            for p in params:
                numel = p.numel()
                p.copy_(old_params[offset:offset + numel].reshape(p.shape))
                offset += numel

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
