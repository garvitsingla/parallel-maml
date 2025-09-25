import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import gymnasium as gym 
from maml_rl.episode import BatchEpisodes
from maml_rl.utils.reinforcement_learning import reinforce_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = None
mission_encoder = None


def rollout_one_task(args):
    (make_env_fn, task, policy_cls, policy_kwargs,
     policy_state_dict, adapted_params_cpu, batch_size, gamma) = args

    env = make_env_fn()
    env.reset_task(task)

    policy = policy_cls(**policy_kwargs)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    obs_list, act_list, rew_list, ep_id_list = [], [], [], []
    total_steps = 0

    for ep in range(batch_size):
        obs, info = env.reset()
        done, steps = False, 0
        while not done:
            obs_vec = preprocess_obs(obs)
            with torch.no_grad():
                pi = policy(torch.from_numpy(obs_vec[None, :]).float(),
                            params=adapted_params_cpu)
                action = pi.sample().item()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
            obs_list.append(obs_vec)
            act_list.append(action)
            rew_list.append(r)
            ep_id_list.append(ep)
        total_steps += steps

    return (task, total_steps, obs_list, act_list, rew_list, ep_id_list)




# Mission Wrapper
class BabyAIMissionTaskWrapper(gym.Wrapper):
    def __init__(self, env, missions=None):
        assert missions is not None, "tasks not there"
        super().__init__(env)
        self.missions = missions
        self.current_mission = None

    def sample_tasks(self, n_tasks):
        return list(np.random.choice(self.missions, n_tasks, replace=False))

    def reset_task(self, mission):
        self.current_mission = mission
        if hasattr(self.env, 'set_forced_mission'):
            self.env.set_forced_mission(mission)

    def reset(self, **kwargs):        
        result = super().reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        if self.current_mission is not None:
            obs['mission'] = self.current_mission
        if isinstance(result, tuple):
            return obs, info
        else:
            return obs



# Mission Encoder
class MissionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def preprocess_obs(obs):

    image = obs["image"].flatten() / 255.0
    direction = np.eye(4)[obs["direction"]]

    return np.concatenate([image,direction])  


# Sampler
class MultiTaskSampler(object):
    def __init__(self,    
                 env=None, 
                 env_fn=None,               
                 batch_size=None,       
                 policy=None,
                 baseline=None,     
                 seed=None,
                 num_workers=0):  
                 
        assert env is not None, "Must pass BabyAI env"
        self.env = env
        self.env_fn = env_fn
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.seed = seed
        self.num_workers = num_workers

    def sample_tasks(self, num_tasks):
        return self.env.sample_tasks(num_tasks)

    def sample(self, meta_batch_size, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        
        tasks = self.sample_tasks(meta_batch_size)  

        train_episodes_all = []
        valid_episodes_all = []
        all_step_counts = []  

        if (self.num_workers or 0) == 0:
            for task_index, task in enumerate(tasks):
                self.env.reset_task(task)
                train_batches = []
                params = None
                for _ in range(num_steps):
                    batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                    for ep in range(self.batch_size):
                        obs, info = self.env.reset()
                        done = False

                        episode_obs = []
                        episode_actions = []
                        episode_rewards = []
                        step_count = 0

                        while not done:
                            obs_vec = preprocess_obs(obs)
                            if np.isnan(obs_vec).any():
                                print("NaN in obs_vec, skipping episode")
                                break
                            obs_tensor = np.expand_dims(obs_vec, axis=0)
                            obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                            with torch.no_grad():
                                pi = self.policy(obs_tensor, params=params)
                                action = pi.sample().item()

                            if np.isnan(action):
                                print("NaN in action, skipping episode")
                                break
                            obs, reward, terminated, truncated, info = self.env.step(action)
                            step_count += 1

                            if np.isnan(reward):
                                print("NaN in reward, skipping episode")
                                break

                            done = terminated or truncated
                            episode_obs.append(obs_vec)
                            episode_actions.append(action)
                            episode_rewards.append(reward)

                        all_step_counts.append(step_count)  
                        
                        if len(episode_obs) > 0 and not np.isnan(episode_obs).any():
                            batch.append(
                                episode_obs,
                                [np.array(a) for a in episode_actions],
                                [np.array(r) for r in episode_rewards],
                                [ep]*len(episode_obs)
                            )
                                                
                    self.baseline.fit(batch)
                    batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                    
                    if torch.isnan(batch.advantages).any():
                        print("NaN in batch advantages!")
                    if torch.isnan(batch.observations).any():
                        print("NaN in batch observations!")

                    loss = reinforce_loss(self.policy, batch, params=params)
                    params = self.policy.update_params(loss, params=params, step_size=fast_lr, first_order=True)
                    train_batches.append(batch)
                train_episodes_all.append(train_batches)

                # Outer Loop
                valid_batch = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                for ep in range(self.batch_size):
                    obs, info = self.env.reset()
                    done = False
                    while not done:
                        obs_vec = preprocess_obs(obs)
                        obs_tensor = np.expand_dims(obs_vec, axis=0)
                        obs_tensor = torch.from_numpy(obs_tensor).float().to(device)
                        with torch.no_grad():
                            pi = self.policy(obs_tensor, params=params)
                            action = pi.sample().item()
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        done = terminated or truncated
                        valid_batch.append([obs_vec], [np.array(action)], [np.array(reward)], [ep])
                
                self.baseline.fit(valid_batch)
                valid_batch.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                valid_episodes_all.append(valid_batch)

        else:
            assert self.env_fn is not None, "env_fn not there"

            policy_state_dict_cpu = {k: v.cpu() for k, v in self.policy.state_dict().items()}
            policy_cls = self.policy.__class__
            policy_kwargs = dict(
                input_size=self.policy.input_size,
                output_size=self.policy.output_size,
                hidden_sizes=self.policy.hidden_sizes,
                nonlinearity=self.policy.nonlinearity
            )

            task_params = [None for _ in tasks]
            per_task_train_batches = [[] for _ in tasks]

            for _ in range(num_steps):
                worker_args = []
                for t, p in zip(tasks, task_params):
                    p_cpu = None if p is None else {k: v.detach().cpu() for k, v in p.items()}
                    worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                        policy_state_dict_cpu, p_cpu, self.batch_size, gamma))
                with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                        results = list(ex.map(rollout_one_task, worker_args))

                # Build batches, fit baseline, compute advantages, then update params for next step
                new_task_params = []
                for task_idx, (mission, step_count, obs_list, act_list, rew_list, ep_list) in enumerate(results):
                    be = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                    for o, a, r, e in zip(obs_list, act_list, rew_list, ep_list):
                        be.append([o], [np.array(a)], [np.array(r)], [np.array(e)])
                    self.baseline.fit(be)
                    be.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)
                    per_task_train_batches[task_idx].append(be)
                    all_step_counts.append(step_count)

                    # compute inner loss & update θ→θ'
                    loss = reinforce_loss(self.policy, be, params=task_params[task_idx])
                    theta_prime = self.policy.update_params(loss,
                                                            params=task_params[task_idx],
                                                            step_size=fast_lr,
                                                            first_order=True)
                    new_task_params.append(theta_prime)
                task_params = new_task_params 

            
            # Outer Loop
            worker_args = []
            for t, p in zip(tasks, task_params):
                p_cpu = {k: v.detach().cpu() for k, v in p.items()}
                worker_args.append((self.env_fn, t, policy_cls, policy_kwargs,
                                    policy_state_dict_cpu, p_cpu, self.batch_size, gamma))
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(ex.map(rollout_one_task, worker_args))

            for task_idx, (mission, step_count, obs_list, act_list, rew_list, ep_list) in enumerate(results):
                vb = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
                for o, a, r, e in zip(obs_list, act_list, rew_list, ep_list):
                    vb.append([o], [np.array(a)], [np.array(r)], [np.array(e)])
                self.baseline.fit(vb)
                vb.compute_advantages(self.baseline, gae_lambda=gae_lambda, normalize=True)

                train_episodes_all.append(per_task_train_batches[task_idx])
                valid_episodes_all.append(vb)
                all_step_counts.append(step_count)
            
        return (train_episodes_all, valid_episodes_all, all_step_counts)
