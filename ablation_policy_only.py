import torch
import numpy as np
import random
from collections import OrderedDict
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
import sampler_lang as SL
from maml_rl.policies.categorical_mlp import CategoricalMLPPolicy
import pickle
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


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


room_size=9
num_dists=3
max_steps=350
model = "ActionObjDoor_7_3_300"  
delta_theta = 0.4
num_batches = 50

# # GoToLocal
# base_env = GoToLocalMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions=LOCAL_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n available missions: {LOCAL_MISSIONS}\n ")


# # Pickup
# base_env = PickupDistMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions=PICKUP_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n available missions: {PICKUP_MISSIONS}\n delta_theta: {delta_theta}\n")



# # GoToObjDoor
# base_env = GoToObjDoorMissionEnv(max_steps=max_steps, num_distractors=num_dists)
# missions=LOCAL_MISSIONS + DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"num_dists: {num_dists}\n max_steps: {max_steps}\n")



# # GoToOpen
# base_env = GoToOpenMissionEnv(room_size=room_size, num_dists=num_dists, max_steps=max_steps)
# missions=LOCAL_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size} \nnum_dists: {num_dists} \nmax_steps: {max_steps} \n")



# # OpenDoorMissionEnv
# base_env = OpenDoorMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")



# # OpenDoorLocMissionEnv
# base_env = OpenDoorLocMissionEnv(room_size=room_size, max_steps=max_steps)
# missions = OPEN_DOOR_MISSIONS + DOOR_LOC_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n")




# # OpenTwoDoors
# base_env = OpenTwoDoorsMissionEnv(room_size=room_size, max_steps=None)
# missions = OPEN_TWO_DOORS_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}")
#         # \nmax_steps: {max_steps} \n")





# # OpenDoorsOrder
# base_env = OpenDoorsOrderMissionEnv(room_size=room_size)
# missions = OPEN_DOORS_ORDER_MISSIONS
# env = BabyAIMissionTaskWrapper(base_env, missions=missions)
# print(f"room_size: {room_size}\n")



# ActionObjDoor
base_env = ActionObjDoorMissionEnv()
missions = ACTION_OBJ_DOOR_MISSIONS
env = BabyAIMissionTaskWrapper(base_env, missions=missions)
print("General setup for ActionObjDoor")
# # print(f"room_size: {room_size}  \nmax_steps: {max_steps} \n num_distractors: {num_dists} \n")




# restore saved lang-adapted policy 

lang_model = torch.load(f"lang_model/lang_policy_{model}_{delta_theta}_{num_batches}.pth", map_location=device)
with open(f"lang_model/vectorizer_lang_{model}_{delta_theta}_{num_batches}.pkl", "rb") as f:
    vectorizer_lang = pickle.load(f)


SL.vectorizer = vectorizer_lang  
mission_encoder_output_dim = 32
SL.mission_encoder = MissionEncoder(len(SL.vectorizer.get_feature_names_out()), 32, 64, mission_encoder_output_dim).to(device)
SL.mission_encoder.load_state_dict(lang_model["mission_encoder"])
SL.mission_encoder.eval()
mission_encoder = SL.mission_encoder
preprocess_obs = SL.preprocess_obs

dummy_obs, _ = env.reset()
input_size = preprocess_obs(dummy_obs).shape[0]
output_size = base_env.action_space.n
hidden_sizes = (64, 64)
nonlinearity = torch.nn.functional.tanh

# Policy language
policy_lang = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_lang.load_state_dict(lang_model["policy"])
policy_lang.eval()
policy_param_shapes = [p.shape for p in policy_lang.parameters()]

# Adapter
mission_adapter = MissionParamAdapter(mission_encoder_output_dim, policy_param_shapes).to(device)
mission_adapter.load_state_dict(lang_model["mission_adapter"])    
mission_adapter.eval()

# Policy Only Ablation

policy_only_model = torch.load(f"ablation_policy_only/lang_policy_{model}_{num_batches}.pth", map_location=device)

policy_only = CategoricalMLPPolicy(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=hidden_sizes,
    nonlinearity=nonlinearity,
).to(device)  
policy_only.load_state_dict(policy_only_model["policy"])
policy_only.eval()
policy_param_shapes = [p.shape for p in policy_only.parameters()]




def get_language_adapted_params(policy, mission_str, mission_encoder, mission_adapter, vectorizer, device):
    mission_vec = vectorizer.transform([mission_str]).toarray()[0]
    mission_tensor = torch.from_numpy(mission_vec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        mission_emb = mission_encoder(mission_tensor)
        mission_emb = mission_emb.to(device)
        delta_thetas = mission_adapter(mission_emb)
        delta_thetas = [delta * delta_theta for delta in delta_thetas]
    policy_params = list(policy.parameters())
    param_names = list(dict(policy.named_parameters()).keys())
    theta_prime = OrderedDict(
        (name, param + delta.squeeze(0))
        for name, param, delta in zip(param_names, policy_params, delta_thetas)
    )
    return theta_prime


def evaluate_policy(env, policy, params=None, max_steps=max_steps, render=False):
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done and steps < max_steps:
        if render:
            env.render("human")
        obs_vec = preprocess_obs(obs)
        obs_tensor = torch.from_numpy(obs_vec).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if params is not None:
                dist = policy(obs_tensor, params=params)
            else:
                dist = policy(obs_tensor)
            action = dist.sample().item()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    return steps


# Evaluation
N_MISSIONS = 20
N_EPISODES = 40

results_lang = []
results_policy_only = []

print("Comparing language-adapted policy and random policy on random missions:")
for i in range(N_MISSIONS):
    mission = random.choice(missions)
    print(f"\nMission {i+1}/{N_MISSIONS}: '{mission}'")

    # 1. Lang-adapted policy
    SL.vectorizer = vectorizer_lang
    SL.mission_encoder = mission_encoder
    theta_prime = get_language_adapted_params(policy_lang, mission, mission_encoder, mission_adapter, vectorizer_lang, device)
    lang_steps = []
    print("  [Lang-adapted policy episodes]")
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_lang, params=theta_prime)
        lang_steps.append(steps)
    mean_lang = np.mean(lang_steps)
    std_lang = np.std(lang_steps)
    results_lang.append(mean_lang)
    print(f"    --> Avg steps: {mean_lang:.2f} ± {std_lang:.2f}")


    # 2. Unadapted lang policy
    
    policy_params = list(policy_only.parameters())
    param_names = list(dict(policy_only.named_parameters()).keys())
    theta_prime = OrderedDict(
        (name, param)
        for name, param in zip(param_names, policy_params)
    )

    policy_only_steps = []
    print("  [Policy only episodes]")
    for ep in range(N_EPISODES):
        env.reset_task(mission)
        steps = evaluate_policy(env, policy_only, params=theta_prime)
        policy_only_steps.append(steps)
    mean_lang = np.mean(policy_only_steps)
    std_lang = np.std(policy_only_steps)
    results_policy_only.append(mean_lang)
    print(f"    --> Avg steps: {mean_lang:.2f} ± {std_lang:.2f}")



end_time = time.time()

print(f"Execution time: {(end_time - start_time)/60} minutes\n")

print(f"room_size: {room_size}\n num_dists: {num_dists}\n max_steps: {max_steps}\n available missions: {missions}\n delta_theta: {delta_theta}\n num_batches: {num_batches}\n")

# Results
print("\n===== FINAL AGGREGATE RESULTS =====")
print(f"Lang-adapted policy: {np.mean(results_lang):.2f} ± {np.std(results_lang):.2f}")
print(f"unadapted Lang policy: {np.mean(results_policy_only):.2f} ± {np.std(results_policy_only):.2f}")
