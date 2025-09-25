from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import copy
import torch
import numpy as np
import random
from environment import (GoToLocalMissionEnv, 
                         GoToOpenMissionEnv, 
                         GoToObjDoorMissionEnv,  
                         PickupDistMissionEnv,
                         OpenDoorMissionEnv, 
                         OpenDoorLocMissionEnv,
                         OpenTwoDoorsMissionEnv,
                         OpenDoorsOrderMissionEnv,
                         ActionObjDoorMissionEnv)
from sampler_lang import BabyAIMissionTaskWrapper, MissionEncoder, MissionParamAdapter
import sampler_lang
import sampler_maml
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
import time
from maml_rl.utils.reinforcement_learning import reinforce_loss
from maml_rl.episode import BatchEpisodes
from maml_rl.baseline import LinearFeatureBaseline


start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJECTS = ['box']
COLORS = ['red', 'green', 'blue', 'purple','yellow', 'grey']
PREP_LOCS = ['on', 'at', 'to']

# Location names
LOC_NAMES = ['right', 'front']

DOOR_COLORS = ['yellow', 'grey']

# For Pickup
PICKUP_MISSIONS = [f"pick up the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For GoToLocal
LOCAL_MISSIONS = [f"go to the {color} {obj}" for color in COLORS for obj in OBJECTS]

# For environments that include doors (GoToObjDoor, GoToOpen, Open)
DOOR_MISSIONS = [f"go to the {color} door" for color in DOOR_COLORS]
OPEN_DOOR_MISSIONS = [f"open the {color} door" for color in DOOR_COLORS]
DOOR_LOC_MISSIONS = [f"open the door {prep} the {loc}" for prep in PREP_LOCS for loc in LOC_NAMES]
OPEN_TWO_DOORS_MISSIONS = [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
OPEN_DOORS_ORDER_MISSIONS = (
    [f"open the {c1} door" for c1 in DOOR_COLORS] +
    [f"open the {c1} door, then open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS] +
    [f"open the {c1} door after you open the {c2} door" for c1 in DOOR_COLORS for c2 in DOOR_COLORS]
)

ACTION_OBJ_DOOR_MISSIONS = (
    [f"pick up the {c} box" for c in COLORS] +
    [f"go to the {c} box"   for c in COLORS] +
    [f"go to the {c} door"   for c in DOOR_COLORS] +
    [f"open a {c} door"      for c in DOOR_COLORS]
)



def build_env(env_name, room_size, num_dists, max_steps, missions):
    if env_name == "PickupDist":
        base = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "GoToLocal":
        base = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "OpenDoor":
        base = OpenDoorMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "OpenDoorLoc":
        base = OpenDoorLocMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "OpenTwoDoors":
        base = OpenTwoDoorsMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "OpenDoorsOrder":
        base = OpenDoorsOrderMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    elif env_name == "ActionObjDoor":
        base = ActionObjDoorMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
    else:
        raise ValueError(f"Unknown env_name {env_name}")
    return BabyAIMissionTaskWrapper(base, missions=missions)



def eval_one_mission_worker(mission, env_name, room_size, num_dists, max_steps,
                            missions, vectorizer_bytes,
                            policy_lang_state, mission_encoder_state, mission_adapter_state,
                            policy_lang2_state, mission_adapter2_state,
                            maml_vectorizer_bytes, maml_encoder_state, policy_maml_state,
                            hidden_sizes, nonlinearity, fast_lr, max_episode_steps, n_episodes):
    """
    Returns: dict with means/stdevs for each policy kind for this mission.
    Everything runs on CPU inside the worker to avoid CUDA forking issues.
    """
    import torch, numpy as np, pickle
    from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy

    device_cpu = torch.device("cpu")

    # Recreate env
    env = build_env(env_name, room_size, num_dists, max_steps, missions)

    # Recreate vectorizers/encoders/adapters/policies from state_dicts
    vectorizer = pickle.loads(vectorizer_bytes)
    sampler_lang.vectorizer = vectorizer
    mission_encoder = MissionEncoder(len(vectorizer.get_feature_names_out()), 32, 64, 32).to(device_cpu)
    mission_encoder.load_state_dict(policy_lang_state["mission_encoder"] if "mission_encoder" in policy_lang_state
                                    else mission_encoder_state)
    mission_encoder.eval()

    # Policy (language) and its adapter
    # Note: input size depends on preprocess_obs; reconstruct like in your main.py
    dummy_obs, _ = env.reset()
    input_size_lang = sampler_lang.preprocess_obs(dummy_obs).shape[0]
    output_size = env.base_env.action_space.n  # base_env is inside the wrapper
    policy_lang = CategoricalMLPPolicy(
        input_size=input_size_lang,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity
    ).to(device_cpu)
    policy_lang.load_state_dict(policy_lang_state["policy"] if "policy" in policy_lang_state
                                else policy_lang_state)
    policy_lang.eval()

    mission_adapter = MissionParamAdapter(32, [p.shape for p in policy_lang.parameters()]).to(device_cpu)
    mission_adapter.load_state_dict(mission_adapter_state)
    mission_adapter.eval()

    # Second adapter (“without language” theta-only case if you keep it)
    mission_adapter2 = MissionParamAdapter(32, [p.shape for p in policy_lang.parameters()]).to(device_cpu)
    mission_adapter2.load_state_dict(mission_adapter2_state)
    mission_adapter2.eval()

    # Recreate MAML side
    maml_vectorizer = pickle.loads(maml_vectorizer_bytes)
    sampler_maml.vectorizer = maml_vectorizer
    sampler_maml.mission_encoder = sampler_maml.MissionEncoder(
        len(maml_vectorizer.get_feature_names_out()), 32, 64, 32
    ).to(device_cpu)
    sampler_maml.mission_encoder.load_state_dict(maml_encoder_state)
    sampler_maml.mission_encoder.eval()

    input_size_maml = sampler_maml.preprocess_obs(dummy_obs).shape[0]
    policy_maml = CategoricalMLPPolicy(
        input_size=input_size_maml,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        nonlinearity=nonlinearity
    ).to(device_cpu)
    policy_maml.load_state_dict(policy_maml_state)
    policy_maml.eval()

    # Scratch (random) policy on same input size as lang
    scratch_policy = CategoricalMLPPolicy(
        input_size=input_size_lang, output_size=output_size,
        hidden_sizes=hidden_sizes, nonlinearity=nonlinearity
    ).to(device_cpu).eval()

    # Helper to run one episode
    def run_one_episode(policy, preprocess_obs, params=None):
        obs, _ = env.reset_task(mission)
        steps, done = 0, False
        while not done and steps < max_episode_steps:
            obs_vec = preprocess_obs(obs)
            obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device_cpu)
            with torch.no_grad():
                dist = policy(obs_tensor, params=params) if params is not None else policy(obs_tensor)
                action = dist.sample().item()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        return steps

    # Compute theta' for lang-adapted
    theta_prime = sampler_lang.get_language_adapted_params(
        policy_lang, mission, mission_encoder, mission_adapter, vectorizer, device_cpu
    )
    theta_prime2 = sampler_lang.get_language_adapted_params2(
        policy_lang, mission, mission_encoder, mission_adapter2, vectorizer, device_cpu
    )

    # Roll multiple episodes
    def roll_many(policy, preprocess_obs, params=None):
        vals = []
        for _ in range(n_episodes):
            vals.append(run_one_episode(policy, preprocess_obs, params))
        return float(np.mean(vals)), float(np.std(vals))

    mean_lang, std_lang = roll_many(policy_lang, sampler_lang.preprocess_obs, theta_prime)
    mean_unadapt, std_unadapt = roll_many(policy_lang, sampler_lang.preprocess_obs, None)
    mean_maml, std_maml = roll_many(policy_maml, sampler_maml.preprocess_obs, None)
    mean_rand, std_rand = roll_many(scratch_policy, sampler_lang.preprocess_obs, None)
    mean_wo_lang, std_wo_lang = roll_many(policy_lang, sampler_lang.preprocess_obs, theta_prime2)

    return {
        "mission": mission,
        "lang": (mean_lang, std_lang),
        "unadapted": (mean_unadapt, std_unadapt),
        "maml": (mean_maml, std_maml),
        "random": (mean_rand, std_rand),
        "without_lang": (mean_wo_lang, std_wo_lang),
    }
