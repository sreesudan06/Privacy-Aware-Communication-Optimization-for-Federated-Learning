import socket
import threading
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import struct
from collections import defaultdict
import logging
import sys
import psutil

np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 32
EPOCHS_PER_ROUND = 1
THRESHOLD = 0.6      # Threshold for weight difference between rounds
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
TIMEOUT = 20         # Timeout in seconds for waiting for models
R_PRIME = 100        # Maximum number of rounds
MINIMUM_ROUNDS = 20  # Minimum rounds before checking termination criteria
COUNT_THRESHOLD = 5  # Number of consecutive rounds for weight difference and no crashes

# ================= PRIVGAT CONFIG =================
PRIVGAT_ENABLED = True
PRIVGAT_DENSITY = 0.02
PRIVGAT_MIN_K = 256
PRIVGAT_CLIP_VALUE = 0.05
PRIVGAT_NOISE_STD = 0.002
PRIVGAT_PRIVACY_LAMBDA = 0.4
TOKEN_EXPIRY = 3
TOKEN_BYTES = 0
FULL_WEIGHT_BYTES = 0
ORIGINAL_BYTES = 0
# ================= ENERGY GATEWAY =================
ENERGY_GATEWAY = {}
ENERGY_HISTORY = {}

LAST_ENERGY = {}
ENERGY_UPDATE_INTERVAL = 10
FORCE_ENERGY_UPDATE = {}
LAST_DENSITY = {}

# ================= CUS =================
CLIENT_CUS = {}
CLIENT_BYTES = {}
CLIENT_PREV_LOSS = {}
CLIENT_RELIABILITY = {}
CLIENT_HISTORY = {}

# ==========================================================
# IMOCC GLOBALS
# ==========================================================

CLIENT_POLICY = {}
CLIENT_STATE = {}
POLICY_HISTORY = {}
IMOCC_HISTORY = {}
CLIENT_CRASH_STATUS = {}
CLIENT_PRIVACY_RISK = {}
COMMUNICATION_ROUND = {}

# ==========================================================
# Adaptive Optimization Weights (Phase 3.3)
# ==========================================================

OBJECTIVE_WEIGHTS = {

    "energy": 0.35,

    "communication": 0.30,

    "reliability": 0.20,

    "privacy": 0.15

}

WEIGHT_HISTORY = {}

# ==========================
# CUS Default Weights
# ==========================

CUS_WEIGHTS = {
    "energy": 0.35,
    "communication": 0.30,
    "learning": 0.25,
    "reliability": 0.10
}

AGGREGATION_HISTORY = {}

DECISION_TIMELINE = {}
# ==========================================================
# Phase 5.2
# Global Communication Statistics
# ==========================================================

GLOBAL_STATS = {

    "total_rounds": 0,

    "total_tokens": 0,

    "total_full_models": 0,

    "total_crashes": 0,

    "average_density": [],

    "average_aggregation": [],

    "average_energy": [],

    "average_cus": [],

    "average_trust": [],

    "average_aas": []

}
print("[CONFIG] PrivGAT initialized")

def send_message(conn, message):
    global TOKEN_BYTES
    global FULL_WEIGHT_BYTES
    global ORIGINAL_BYTES

    data = pickle.dumps(message)
    payload = len(data)
    # Calculate what a full model would cost
    if message["type"] == "weights":
        ORIGINAL_BYTES += payload

    elif message["type"] == "privGAT_token":
        full_equivalent = len(
             pickle.dumps({
            "type": "weights",
            "weights": [np.zeros(shape) for shape in message["shapes"]]
        })
       )

        ORIGINAL_BYTES += full_equivalent

    if message["type"] == "privGAT_token":
        TOKEN_BYTES += payload
    else:
        FULL_WEIGHT_BYTES += payload

    print(f"[BYTES SENT] {payload}")
    message["payload_bytes"] = payload

    message_length = struct.pack('!I', payload)
    conn.sendall(message_length + data)

def receive_message(conn):
    message_length_data = conn.recv(4)
    if not message_length_data:
        return None
    message_length = struct.unpack('!I', message_length_data)[0]
    data = b''
    while len(data) < message_length:
        part = conn.recv(min(4096, message_length - len(data)))
        data += part
    return pickle.loads(data)

def parse_input_file():
    try:
        with open("inputf.txt", "r") as file:
            lines = file.read().splitlines()
            if len(lines) < 4:
                raise ValueError("Input file does not contain enough lines.")
            
            num_clients, num_machines = map(int, lines[0].split())
            current_machine_ip = lines[1].strip()
            all_ips = [ip.strip() for ip in lines[2].split(",")]
            num_faults = int(lines[3])
            faults = []
            
            if len(lines) < 4 + num_faults:
                raise ValueError(f"Input file does not contain enough lines for the specified number of faults ({num_faults}).")
            
            for i in range(num_faults):
                id, fr, y = map(int, lines[4+i].split(','))
                faults.append((id, fr, y))
        return num_clients, num_machines, current_machine_ip, all_ips, faults
    except FileNotFoundError:
        print("The input file was not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None, None, None

NUM_CLIENTS, NUM_MACHINES, CURRENT_MACHINE_IP, ips, faults = parse_input_file()

if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    exit(1)

# Configure the logger with a dynamic filename based on input parameters
logger = logging.getLogger('federated_learning')
logger.setLevel(logging.INFO)

# Create a formatter for log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Dynamically name the log file as test_log_<num_clients>_<num_machines>_<num_crashes>.txt
log_filename = f"min40_crash_test_{TIMEOUT}_log_{NUM_CLIENTS}_{NUM_MACHINES}_{len(faults)}.txt"
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create a stream handler to print logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Custom filter to log only crash-related messages to the file
class CrashFilter(logging.Filter):
    def filter(self, record):
        return "crash" in record.msg.lower() or "crashing" in record.msg.lower()

# Add filter to file handler only (not console)
file_handler.addFilter(CrashFilter())

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Redirect print statements to the logger
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

# Redirect stdout to the logger
sys.stdout = LoggerWriter(logger, logging.INFO)
 
retries_list = [1] * NUM_CLIENTS
adj = [[j for j in range(NUM_CLIENTS) if j != i] for i in range(NUM_CLIENTS)]
terminate_messages = [0] * NUM_CLIENTS
model_messages = [0] * NUM_CLIENTS

# CIFAR-10 dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indices = np.random.permutation(len(train_dataset))

def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.5, fixed_data_per_client=5000):
    num_classes = 10  # CIFAR-10 has 10 classes
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}
    
    for c, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(indices)).astype(int)
        start_idx = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(indices[start_idx:start_idx + count])
            start_idx += count
    
    # Truncate or oversample to ensure fixed_data_per_client for each client
    final_client_indices = {}
    for client_id, indices in client_indices.items():
        np.random.shuffle(indices)
        if len(indices) > fixed_data_per_client:
            final_client_indices[client_id] = indices[:fixed_data_per_client]
        else:
            # Oversample if not enough data
            final_client_indices[client_id] = np.random.choice(
                indices, fixed_data_per_client, replace=True
            ).tolist()

    client_data = [
        torch.utils.data.Subset(dataset, final_client_indices[i]) for i in range(num_clients)
    ]
    return client_data

client_data = create_dirichlet_non_iid_splits_fixed(train_dataset, NUM_CLIENTS, alpha=0.5, fixed_data_per_client=FIXED_DATA_PER_CLIENT)

msg_lck = threading.Lock()

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # CIFAR-10 has 3 channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjusted for CIFAR-10 image dimensions
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def flatten_weights(weights):
    shapes = [w.shape for w in weights]
    flat = np.concatenate([w.flatten() for w in weights])
    print(f"[FLATTEN] size={len(flat)}")
    return flat, shapes

def compute_privacy_risk(delta):
    risk = np.abs(delta)
    print(f"[RISK] mean={np.mean(risk):.6f}")
    return risk

def compute_communication_utility(
        client_id,
        energy,
        loss,
        bytes_sent,
        crash):

    previous_loss = CLIENT_PREV_LOSS.get(client_id, loss)

    loss_gain = max(previous_loss - loss, 0)

    CLIENT_PREV_LOSS[client_id] = loss

    reliability = CLIENT_RELIABILITY.get(client_id, 1.0)

    if crash:
        reliability *= 0.90
    else:
        reliability = min(reliability + 0.01, 1.0)

    CLIENT_RELIABILITY[client_id] = reliability

    energy_score = energy / 100

    communication_score = max(
        0,
        1 - bytes_sent / (10 * 1024 * 1024)
    )

    learning_score = min(loss_gain, 1)
    
    cus = (
          CUS_WEIGHTS["energy"] * energy_score +
          CUS_WEIGHTS["communication"] * communication_score +
          CUS_WEIGHTS["learning"] * learning_score +
          CUS_WEIGHTS["reliability"] * reliability
          )

    CLIENT_CUS[client_id] = cus

    print(
        f"[CUS] "
        f"Client={client_id} "
        f"CUS={cus:.4f}"
    )

    return cus

class CommunicationState:

    def __init__(
        self,
        client_id,
        energy,
        cus,
        reliability,
        privacy_risk,
        crashed,
        current_round
    ):

        self.client_id = client_id

        self.energy = energy

        self.cus = cus

        self.reliability = reliability

        self.privacy_risk = privacy_risk

        self.crashed = crashed

        self.current_round = current_round

    def to_dict(self):

        return {

            "client_id": self.client_id,

            "energy": self.energy,

            "cus": self.cus,

            "reliability": self.reliability,

            "privacy_risk": self.privacy_risk,

            "crashed": self.crashed,

            "round": self.current_round

        }

class CommunicationPolicy:

    def __init__(self):
        self.policy = {

           "density": 0.02,
           "compression": 0.95,
           "aggregation_weight": 1.0,
           "priority": "NORMAL",
           "token_expiry": TOKEN_EXPIRY,
           "communication_mode": "BALANCED",
           "timestamp": 0
        }

    def update(

        self,

        density,

        compression,

        aggregation_weight,

        priority,

        timestamp

    ):

        self.policy["density"] = density

        self.policy["compression"] = compression

        self.policy["aggregation_weight"] = aggregation_weight

        self.policy["priority"] = priority

        self.policy["timestamp"] = timestamp

    def get_policy(self):

        return self.policy

class IMOCC:

    def __init__(self):

        self.states = {}

        self.policies = {}

        print()

        print("=" * 70)

        print("IMOCC INITIALIZED")

        print("=" * 70)

    # ----------------------------------------------------

    def collect_state(

        self,

        client_id,

        energy,

        cus,

        reliability,

        privacy_risk,

        crashed,

        current_round

    ):

        state = CommunicationState(

            client_id,

            energy,

            cus,

            reliability,

            privacy_risk,

            crashed,

            current_round

        )

        self.states[client_id] = state

        CLIENT_STATE[client_id] = state

        return state
    
    def build_policy(self, state):
        density = 0.02
        compression = 0.95
        aggregation = 1.0
        expiry = TOKEN_EXPIRY
        priority = "NORMAL"
        mode = "BALANCED"
        
        if state.energy > 80:
            density += 0.01
            compression -= 0.05
            mode = "HIGH_PERFORMANCE"

        elif state.energy < 50:
            density -= 0.005
            compression += 0.05
            mode = "ENERGY_SAVING"

        if state.cus < 0.45:
            priority = "HIGH"
            density += 0.005

        elif state.cus > 0.80:
            priority = "LOW"

        if state.privacy_risk > 0.03:
            compression += 0.05
            aggregation *= 0.90
            expiry += 2
            mode = "PRIVACY_SAFE"
     
        if state.crashed:
            priority = "CRITICAL"
            aggregation *= 0.80
            expiry += 3
            density += 0.01
        density = max(0.01, min(0.05, density))
        compression = max(0.80, min(0.99, compression))
        aggregation = max(0.5, min(1.2, aggregation))

        return {

            "density": density,

            "compression": compression,

            "aggregation_weight": aggregation,

            "priority": priority,

            "token_expiry": expiry,

            "communication_mode": mode
        }
        
    def update_objective_weights(self, state):
        global OBJECTIVE_WEIGHTS

        energy = state.energy / 100.0

        communication = state.cus

        reliability = state.reliability

        privacy = 1.0 - min(state.privacy_risk, 1.0)

        # ----------------------------------------
        # Adaptive Weight Update
        # ----------------------------------------

        OBJECTIVE_WEIGHTS["energy"] = 0.25 + (1-energy)*0.20

        OBJECTIVE_WEIGHTS["communication"] = \
            0.20 + (1-communication)*0.20

        OBJECTIVE_WEIGHTS["reliability"] = \
            0.15 + (1-reliability)*0.15

        OBJECTIVE_WEIGHTS["privacy"] = \
            0.10 + (1-privacy)*0.15

        total = sum(OBJECTIVE_WEIGHTS.values())

        for key in OBJECTIVE_WEIGHTS:
            OBJECTIVE_WEIGHTS[key] /= total

        WEIGHT_HISTORY.setdefault(
            state.client_id,
            []
        ).append(
        OBJECTIVE_WEIGHTS.copy()
        )

        print()

        print("="*70)

        print("[ADAPTIVE OBJECTIVE WEIGHTS]")

        for k,v in OBJECTIVE_WEIGHTS.items():
            print(f"{k:<18}: {v:.4f}")

        print("="*70)
        
        # =====================================================
    # Phase 3.2
    # Multi-Objective Decision Optimization Algorithm
    # =====================================================

    def optimize_policy(self, state):

        # -----------------------------
        # Objective Scores
        # -----------------------------

        energy_score = state.energy / 100.0

        communication_score = state.cus

        reliability_score = state.reliability

        privacy_score = 1.0 - min(state.privacy_risk, 1.0)
        
        self.update_objective_weights(state)

        # Weighted Optimization

        optimization_score = (

              OBJECTIVE_WEIGHTS["energy"] * energy_score +

              OBJECTIVE_WEIGHTS["communication"] * communication_score +

              OBJECTIVE_WEIGHTS["reliability"] * reliability_score +

              OBJECTIVE_WEIGHTS["privacy"] * privacy_score

             )
        # -----------------------------
        # Density Optimization
        # -----------------------------

        density = 0.01 + optimization_score * 0.04

        density = max(0.01, min(0.05, density))

        # -----------------------------
        # Compression Optimization
        # -----------------------------

        compression = 0.99 - (optimization_score * 0.15)

        compression = max(0.80, min(0.99, compression))

        # -----------------------------
        # Aggregation Optimization
        # -----------------------------

        aggregation = 0.6 + optimization_score * 0.6

        aggregation = max(0.50, min(1.20, aggregation))

        # -----------------------------
        # Token Expiry Optimization
        # -----------------------------

        expiry = TOKEN_EXPIRY

        if state.privacy_risk > 0.03:
            expiry += 2

        if state.crashed:
            expiry += 3

        # -----------------------------
        # Priority Optimization
        # -----------------------------

        if state.crashed:

            priority = "CRITICAL"

        elif optimization_score > 0.80:

            priority = "LOW"

        elif optimization_score < 0.45:

            priority = "HIGH"

        else:

            priority = "NORMAL"

        # -----------------------------
        # Communication Mode
        # -----------------------------

        if optimization_score > 0.80:

            mode = "HIGH_PERFORMANCE"

        elif optimization_score < 0.45:

            mode = "ENERGY_SAVING"

        else:

            mode = "BALANCED"

        print()

        print("="*70)

        print("[MODOA OPTIMIZATION]")

        print(f"Optimization Score : {optimization_score:.4f}")

        print(f"Density            : {density:.4f}")

        print(f"Compression        : {compression:.4f}")

        print(f"Aggregation Weight : {aggregation:.4f}")

        print("="*70)

        return {

            "density": density,

            "compression": compression,

            "aggregation_weight": aggregation,

            "priority": priority,

            "token_expiry": expiry,

            "communication_mode": mode,

            "optimization_score": optimization_score

        }
    
    def generate_policy(self, client_id):
        state = self.states[client_id]

        decision = self.optimize_policy(state)

        policy = CommunicationPolicy()

        policy.update(

             density=decision["density"],

             compression=decision["compression"],

             aggregation_weight=decision["aggregation_weight"],

             priority=decision["priority"],

             timestamp=state.current_round

        )

        policy.policy["token_expiry"] = decision["token_expiry"]

        policy.policy["communication_mode"] = decision["communication_mode"]

        CLIENT_POLICY[client_id] = policy

        self.policies[client_id] = policy

        print()
        print("=" * 70)
        print("[IMOCC POLICY GENERATED]")
        print(f"Client : {client_id}")
        print("=" * 70)

        return policy
    # ----------------------------------------------------

    def print_state(

        self,

        client_id

    ):

        state = self.states[client_id]

        print()

        print("=" * 70)

        print("[IMOCC STATE]")

        print(f"Client           : {state.client_id}")

        print(f"Energy           : {state.energy:.2f}")

        print(f"CUS              : {state.cus:.4f}")

        print(f"Reliability      : {state.reliability:.4f}")

        print(f"Privacy Risk     : {state.privacy_risk:.4f}")

        print(f"Crash            : {state.crashed}")

        print(f"Round            : {state.current_round}")

        print("=" * 70)
        
    def print_policy_summary(self, client_id):
        p = CLIENT_POLICY[client_id].get_policy()

        print()

        print("=" * 70)

        print("COMMUNICATION POLICY")

        print("=" * 70)

        print(f"Density             : {p['density']:.4f}")

        print(f"Compression         : {p['compression']:.2f}")

        print(f"Aggregation Weight  : {p['aggregation_weight']:.2f}")

        print(f"Priority            : {p['priority']}")

        print(f"Token Expiry        : {p['token_expiry']}")

        print(f"Mode                : {p['communication_mode']}")
        
        print()

        print("Adaptive Objective Weights")

        for key, value in OBJECTIVE_WEIGHTS.items():
            print(f"  {key:<15}: {value:.4f}")

        print("=" * 70)

   

    def print_policy(

        self,

        client_id

    ):

        policy = CLIENT_POLICY[client_id]

        print()

        print("=" * 70)

        print("[IMOCC POLICY]")

        print(policy.get_policy())

        print("=" * 70)

# ==========================================================
# MODO
# Multi-Objective Decision Optimizer
# ==========================================================

class MODO:

    def __init__(self):

        self.history = {}

        print()
        print("="*70)
        print("MODO INITIALIZED")
        print("="*70)

    def objective_scores(
            self,
            state,
            policy
    ):

        energy_score = state.energy / 100

        communication_score = 1.0 - (
            policy["density"] / 0.05
        )

        learning_score = state.cus

        reliability_score = state.reliability

        privacy_score = 1.0 - state.privacy_risk

        return {

            "energy": energy_score,

            "communication": communication_score,

            "learning": learning_score,

            "reliability": reliability_score,

            "privacy": privacy_score

        }
    
        # =====================================================
    # Phase 3.4
    # Pareto Policy Selector
    # =====================================================

    def pareto_select(self, state):

        candidates = []

        densities = [0.01, 0.02, 0.03, 0.04, 0.05]

        compressions = [0.80, 0.85, 0.90, 0.95, 0.99]

        aggregations = [0.6, 0.8, 1.0, 1.2]

        for density in densities:

            for compression in compressions:

                for aggregation in aggregations:

                    communication_score = 1.0 - density / 0.05

                    energy_score = state.energy / 100

                    reliability_score = state.reliability

                    privacy_score = (
                        compression * (1.0 - state.privacy_risk)
                    )

                    overall = (

                        OBJECTIVE_WEIGHTS["energy"]
                        * energy_score +

                        OBJECTIVE_WEIGHTS["communication"]
                        * communication_score +

                        OBJECTIVE_WEIGHTS["reliability"]
                        * reliability_score +

                        OBJECTIVE_WEIGHTS["privacy"]
                        * privacy_score

                    )

                    candidates.append({

                        "density": density,

                        "compression": compression,

                        "aggregation": aggregation,

                        "score": overall

                    })

        best = max(

            candidates,

            key=lambda x: x["score"]

        )

        print()

        print("="*70)

        print("[PARETO DECISION]")

        print(f"Density      : {best['density']:.4f}")

        print(f"Compression  : {best['compression']:.2f}")

        print(f"Aggregation  : {best['aggregation']:.2f}")

        print(f"Score        : {best['score']:.4f}")

        print("="*70)

        return best

    def print_scores(self, scores):

        print()

        print("="*70)
        print("MODO OBJECTIVES")
        print("="*70)

        print(f"Energy          : {scores['energy']:.3f}")
        print(f"Communication   : {scores['communication']:.3f}")
        print(f"Learning        : {scores['learning']:.3f}")
        print(f"Reliability     : {scores['reliability']:.3f}")
        print(f"Privacy         : {scores['privacy']:.3f}")

        print("="*70)
        
    def save_history(self,client_id,round_number,scores,policy):
        if client_id not in self.history:
            self.history[client_id] = []

        self.history[client_id].append({

             "round": round_number,

             "energy": scores["energy"],

             "communication": scores["communication"],

             "learning": scores["learning"],

             "reliability": scores["reliability"],

             "privacy": scores["privacy"],

             "density": policy["density"],

             "compression": policy["compression"],

             "aggregation": policy["aggregation_weight"]

            })
        
    def print_history(self, client_id):
        if client_id not in self.history:
            return

        print()
        print("="*70)
        print("MODO HISTORY")
        print("="*70)

        for h in self.history[client_id]:
            print(
               f"R{h['round']:3d} | "
               f"E={h['energy']:.3f} "
               f"C={h['communication']:.3f} "
               f"L={h['learning']:.3f} "
               f"R={h['reliability']:.3f} "
               f"P={h['privacy']:.3f} "
               f"D={h['density']:.3f} "
               f"A={h['aggregation']:.3f}"
             )

        print("="*70)
        
# ==========================================================
# GLOBAL IMOCC INSTANCE
# ==========================================================

imocc = IMOCC()
modo = MODO()


def build_privGAT_token(sender_id,
                        sender_weights,
                        peer_weights,
                        current_round):
    sender_flat, shapes = flatten_weights(sender_weights)
    peer_flat, _ = flatten_weights(peer_weights)
    
    delta = sender_flat - peer_flat
    delta = np.clip(delta, -PRIVGAT_CLIP_VALUE, PRIVGAT_CLIP_VALUE)
    
    risk = compute_privacy_risk(delta)
    CLIENT_PRIVACY_RISK[sender_id] = float(np.mean(risk))

    score = np.abs(delta) / (
         1 + PRIVGAT_PRIVACY_LAMBDA * risk
     )
    if sender_id in CLIENT_POLICY:
        density = CLIENT_POLICY[sender_id].get_policy()["density"]
    else:
        density = LAST_DENSITY[sender_id]
    k = max(
        PRIVGAT_MIN_K,
        int(len(delta)*density)
     )
    """
    print(
      f"[TOKEN DENSITY] "
      f"Round={current_round} "
      f"Density={density:.4f}"
    )"""
    indices = np.argsort(score)[-k:]
    values = delta[indices]
    values += np.random.normal(0, PRIVGAT_NOISE_STD, len(values))
    expiry = TOKEN_EXPIRY

    if sender_id in CLIENT_POLICY:
        expiry = CLIENT_POLICY[sender_id].get_policy()["token_expiry"]

    token = {

         "type": "privGAT_token",

         "sender_id": sender_id,

         "indices": indices,

         "values": values,

         "shapes": shapes,

         "expiry": current_round + expiry

         }
   
    print(f"[TOKEN BUILD] k={len(indices)}")
    print(
         f"[IMOCC] "
         f"Density={density:.4f} "
         f"Expiry={expiry}"
         )
    return token

def apply_privGAT_token(receiver_weights, token, current_round):
    if current_round > token["expiry"]:
        print("[TOKEN EXPIRED]")
        return receiver_weights
        
    flat, shapes = flatten_weights(receiver_weights)
    flat[token["indices"]] += token["values"]
    reconstructed = unflatten_weights(flat, shapes)
    print("[TOKEN APPLIED]")
    return reconstructed

def unflatten_weights(flat, shapes):
    reconstructed = []
    start = 0
    for shape in shapes:
        size = np.prod(shape)
        reconstructed.append(flat[start:start+size].reshape(shape))
        start += size
    return reconstructed

def update_gateway(client_id,
                   current_round,
                   cpu_usage,
                   memory_usage,
                   crash_count):

    reliability = max(
        50,
        100 - crash_count * 5
    )

    energy_score = (
        0.5 * (100 - cpu_usage)
        +
        0.3 * (100 - memory_usage)
        +
        0.2 * reliability
    )

    ENERGY_GATEWAY[client_id] = energy_score

    if client_id not in ENERGY_HISTORY:
        ENERGY_HISTORY[client_id] = []

    ENERGY_HISTORY[client_id].append(
        (
            current_round,
            energy_score
        )
    )

    print("\n==============================")
    print(f"[GATEWAY]")
    print(f"Client      : {client_id}")
    print(f"Energy      : {energy_score:.2f}")
    print(f"CUS         : {CLIENT_CUS.get(client_id,0):.4f}")
    print("==============================")

    return energy_score

def query_energy(client_id):

    if client_id in ENERGY_GATEWAY:
        return ENERGY_GATEWAY[client_id]

    return 100


def tcp_client(id, target_id, target_ip, message):
    global retries_list
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    retries = 5
    while retries > 0:
        try:
            client.connect((target_ip, 8650 + target_id))
            send_message(client, message)
            client.close()
            break
        except Exception as e:
            '''
            print(
                   f"[CONNECT FAIL] "
                   f"{id}->{target_id}: {e}"
                 )'''
            retries -= 1
            retries_list[target_id] -= 1
            time.sleep(2)
            if retries == 0:
                break

def broadcast_weights(id, weights, current_round, terminate, ips, latest_models, crash_away_list, prev_list):
    global model_messages
    for pid in adj[id]:
        
        if PRIVGAT_ENABLED and pid in latest_models:
            message = build_privGAT_token(
                     id,
                     weights,
                     latest_models[pid],
                     current_round
             )
            print(f"[SEND TOKEN] {id}->{pid}")
            GLOBAL_STATS["total_tokens"] += 1
        else:
            message = {
                'type': 'weights',
                'weights': weights,
                'round': current_round,
                'terminate': terminate,
                'id': id
            }
            print(f"[SEND FULL] {id}->{pid}")
            GLOBAL_STATS["total_full_models"] += 1
            
        with msg_lck:
            model_messages[id] += 1
            
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)
        CLIENT_BYTES[id] = TOKEN_BYTES + FULL_WEIGHT_BYTES
        
    latest_models[id] = weights
    

def broadcast_terminate(id, ips):
    global terminate_messages
    message = {'type': 'terminate'}
    for pid in adj[id]:
        terminate_messages[id] += 1
        target_ip = ips[pid]
        tcp_client(id, pid, target_ip, message)

def tcp_server(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list):
    server = socket.socket(
               socket.AF_INET,
               socket.SOCK_STREAM
    )
    server.setsockopt(
                 socket.SOL_SOCKET,
                 socket.SO_REUSEADDR,
                 1
    )
    server.bind(
       (local_ip, 8650 + id)
     )
    server.listen(NUM_CLIENTS - 1)

    while True:
        conn, addr = server.accept()
        msg = receive_message(conn)
        if msg['type'] == 'terminate':
            terminate_flags.append(1)
            break
        
        if msg["type"] == "privGAT_token":
            received_weights.append(msg)

            # Token received

            conn.close()
            continue
        
        if msg['type'] == 'weights':
            received_weights.append(msg)
            if msg['terminate'] == 1:
                terminate_flags.append(1)
            latest_models[msg['id']] = msg['weights']  # Update latest models
            if msg['id'] not in prev_list[id]:
                prev_list[id].append(msg['id'])
        conn.close()
    server.close()
    
# ==========================================================
# Phase 4.3
# Trust-Aware Peer Weighting
# ==========================================================

def compute_trust_score(client_id):

    reliability = CLIENT_RELIABILITY.get(client_id, 1.0)

    cus = CLIENT_CUS.get(client_id, 0.5)

    energy = LAST_ENERGY.get(client_id, 100) / 100.0

    crashed = CLIENT_CRASH_STATUS.get(client_id, False)

    trust = (
        0.40 * reliability +
        0.35 * cus +
        0.25 * energy
    )

    if crashed:
        trust *= 0.75

    trust = max(0.50, min(1.20, trust))

    print(
        f"[TRUST] "
        f"Client={client_id} "
        f"Trust={trust:.3f}"
    )

    return trust

# ==========================================================
# Phase 4.4
# Adaptive Aggregation Score (AAS)
# ==========================================================

def compute_adaptive_aggregation_score(client_id):

    energy = LAST_ENERGY.get(client_id, 100) / 100.0

    cus = CLIENT_CUS.get(client_id, 0.5)

    reliability = CLIENT_RELIABILITY.get(client_id, 1.0)

    privacy = 1.0 - min(
        CLIENT_PRIVACY_RISK.get(client_id, 0.0),
        1.0
    )

    trust = compute_trust_score(client_id)

    aas = (

        OBJECTIVE_WEIGHTS["energy"] * energy +

        OBJECTIVE_WEIGHTS["communication"] * cus +

        OBJECTIVE_WEIGHTS["reliability"] * reliability +

        OBJECTIVE_WEIGHTS["privacy"] * privacy

    )

    aas *= trust

    aas = max(0.50, min(1.20, aas))

    print()

    print("="*70)

    print("[ADAPTIVE AGGREGATION SCORE]")

    print(f"Client        : {client_id}")

    print(f"Score         : {aas:.4f}")

    print("="*70)

    return aas

# ==========================================================
# Phase 4.5
# Aggregation History Manager
# ==========================================================

def save_aggregation_history(
        client_id,
        round_number,
        aggregation_weight,
        aas,
        trust,
        energy,
        cus,
        reliability,
        privacy):

    if client_id not in AGGREGATION_HISTORY:
        AGGREGATION_HISTORY[client_id] = []

    AGGREGATION_HISTORY[client_id].append({

        "round": round_number,

        "aggregation_weight": aggregation_weight,

        "aas": aas,

        "trust": trust,

        "energy": energy,

        "cus": cus,

        "reliability": reliability,

        "privacy": privacy

    })


def print_aggregation_history(client_id):

    if client_id not in AGGREGATION_HISTORY:
        return

    print()
    print("="*80)
    print("ADAPTIVE AGGREGATION HISTORY")
    print("="*80)

    for h in AGGREGATION_HISTORY[client_id]:

        print(
            f"R{h['round']:3d} | "
            f"W={h['aggregation_weight']:.3f} | "
            f"AAS={h['aas']:.3f} | "
            f"Trust={h['trust']:.3f} | "
            f"Energy={h['energy']:.3f} | "
            f"CUS={h['cus']:.3f} | "
            f"Rel={h['reliability']:.3f} | "
            f"Privacy={h['privacy']:.3f}"
        )

    print("="*80)
    
# ==========================================================
# Phase 5.2
# Global Statistics Updater
# ==========================================================

def update_global_statistics(
        client_id,
        round_number,
        density,
        aggregation_weight,
        energy,
        cus,
        trust,
        aas,
        crashed=False):

    GLOBAL_STATS["total_rounds"] = max(
        GLOBAL_STATS["total_rounds"],
        round_number + 1
    )

    GLOBAL_STATS["average_density"].append(density)

    GLOBAL_STATS["average_aggregation"].append(
        aggregation_weight
    )

    GLOBAL_STATS["average_energy"].append(energy)

    GLOBAL_STATS["average_cus"].append(cus)

    GLOBAL_STATS["average_trust"].append(trust)

    GLOBAL_STATS["average_aas"].append(aas)

    if crashed:
        GLOBAL_STATS["total_crashes"] += 1
        
def print_global_statistics():

    print()
    print("="*80)
    print("GLOBAL COMMUNICATION STATISTICS")
    print("="*80)

    print(f"Rounds                 : {GLOBAL_STATS['total_rounds']}")

    print(f"Token Messages         : {GLOBAL_STATS['total_tokens']}")

    print(f"Full Model Messages    : {GLOBAL_STATS['total_full_models']}")

    print(f"Crash Events           : {GLOBAL_STATS['total_crashes']}")

    print(f"Average Density        : {np.mean(GLOBAL_STATS['average_density']):.4f}")

    print(f"Average Aggregation    : {np.mean(GLOBAL_STATS['average_aggregation']):.4f}")

    print(f"Average Energy         : {np.mean(GLOBAL_STATS['average_energy']):.4f}")

    print(f"Average CUS            : {np.mean(GLOBAL_STATS['average_cus']):.4f}")

    print(f"Average Trust          : {np.mean(GLOBAL_STATS['average_trust']):.4f}")

    print(f"Average AAS            : {np.mean(GLOBAL_STATS['average_aas']):.4f}")

    print("="*80)
    

# ==========================================================
# Phase 5.3
# Decision Timeline Recorder
# ==========================================================

def record_decision_timeline(
        client_id,
        round_number,
        policy,
        optimization_score,
        trust,
        aas):

    if client_id not in DECISION_TIMELINE:
        DECISION_TIMELINE[client_id] = []

    DECISION_TIMELINE[client_id].append({

        "round": round_number,

        "density": policy["density"],

        "compression": policy["compression"],

        "aggregation": policy["aggregation_weight"],

        "priority": policy["priority"],

        "mode": policy["communication_mode"],

        "expiry": policy["token_expiry"],

        "optimization": optimization_score,

        "trust": trust,

        "aas": aas

    })


def print_decision_timeline(client_id):

    if client_id not in DECISION_TIMELINE:
        return

    print()
    print("=" * 100)
    print("IMOCC DECISION TIMELINE")
    print("=" * 100)

    for d in DECISION_TIMELINE[client_id]:

        print(

            f"R{d['round']:3d} | "

            f"D={d['density']:.3f} | "

            f"C={d['compression']:.2f} | "

            f"A={d['aggregation']:.2f} | "

            f"P={d['priority']:<8} | "

            f"M={d['mode']:<17} | "

            f"E={d['expiry']:2d} | "

            f"OPT={d['optimization']:.3f} | "

            f"T={d['trust']:.3f} | "

            f"AAS={d['aas']:.3f}"

        )

    print("=" * 100)
    

# ==========================================================
# Phase 5.4
# Final Performance Report
# ==========================================================

def print_final_performance_report():

    print()
    print("=" * 100)
    print("FINAL IMOCC PERFORMANCE REPORT")
    print("=" * 100)

    print(f"Total Communication Rounds : {GLOBAL_STATS['total_rounds']}")
    print(f"Token Messages             : {GLOBAL_STATS['total_tokens']}")
    print(f"Full Model Messages        : {GLOBAL_STATS['total_full_models']}")
    print(f"Crash Events               : {GLOBAL_STATS['total_crashes']}")

    print()

    print(f"Average Energy             : {np.mean(GLOBAL_STATS['average_energy']):.4f}")
    print(f"Average CUS                : {np.mean(GLOBAL_STATS['average_cus']):.4f}")
    print(f"Average Trust Score        : {np.mean(GLOBAL_STATS['average_trust']):.4f}")
    print(f"Average AAS               : {np.mean(GLOBAL_STATS['average_aas']):.4f}")
    print(f"Average Density            : {np.mean(GLOBAL_STATS['average_density']):.4f}")
    print(f"Average Aggregation Weight : {np.mean(GLOBAL_STATS['average_aggregation']):.4f}")

    print()

    print("Adaptive Components")
    print("------------------------------")
    print("✓ Energy-Aware Gateway")
    print("✓ Communication Utility Score")
    print("✓ IMOCC Policy Engine")
    print("✓ MODO Optimizer")
    print("✓ Pareto Policy Selection")
    print("✓ Adaptive Aggregation")
    print("✓ Trust-Aware Weighting")
    print("✓ Adaptive Aggregation Score")
    print("✓ Aggregation History")
    print("✓ Decision Timeline")

    print("=" * 100)

# ==========================================================
# Phase 5.5
# Automatic Experiment Summary
# ==========================================================

def print_experiment_summary():

    print()
    print("=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    actual_bytes = TOKEN_BYTES + FULL_WEIGHT_BYTES

    saved_bytes = ORIGINAL_BYTES - actual_bytes

    reduction = (
        saved_bytes /
        max(ORIGINAL_BYTES, 1)
    ) * 100

    print(f"Original Communication : {ORIGINAL_BYTES} Bytes")
    print(f"Actual Communication   : {actual_bytes} Bytes")
    print(f"Bandwidth Reduction    : {reduction:.2f}%")

    print()

    print("Framework Status")

    print("----------------------------")

    print("Energy Gateway              : COMPLETED")

    print("CUS Module                  : COMPLETED")

    print("IMOCC Controller            : COMPLETED")

    print("MODO Optimizer              : COMPLETED")

    print("Pareto Optimization         : COMPLETED")

    print("Adaptive Aggregation        : COMPLETED")

    print("Trust-Aware Weighting       : COMPLETED")

    print("Aggregation History         : COMPLETED")

    print("Decision Timeline           : COMPLETED")

    print("Performance Report          : COMPLETED")

    print()

    print("Overall Result")

    print("----------------------------")

    print("IMOCC Framework Executed Successfully")

    print("=" * 100)

  
# ==========================================================
# Phase 4.2
# Policy-Based Aggregation Weight Assignment
# ==========================================================



def build_aggregation_weights(peer_ids, local_id):

    aggregation_weights = []

    for peer_id in peer_ids:
        if peer_id in CLIENT_POLICY:
            base_weight = CLIENT_POLICY[peer_id].get_policy()["aggregation_weight"]
        else:
            base_weight = 1.0

        aas = compute_adaptive_aggregation_score(peer_id)

        weight = base_weight * aas
        trust = compute_trust_score(peer_id)

        save_aggregation_history(

             client_id=peer_id,

             round_number=COMMUNICATION_ROUND.get(peer_id, 0),

             aggregation_weight=weight,

             aas=aas,

             trust=trust,

             energy=LAST_ENERGY.get(peer_id,100)/100,

             cus=CLIENT_CUS.get(peer_id,0),

            reliability=CLIENT_RELIABILITY.get(peer_id,1),
 
            privacy=CLIENT_PRIVACY_RISK.get(peer_id,0)

         )
        aggregation_weights.append(weight)

    if local_id in CLIENT_POLICY:
        base_weight = CLIENT_POLICY[local_id].get_policy()["aggregation_weight"]

    else:
        base_weight = 1.0

    local_aas = compute_adaptive_aggregation_score(local_id)

    local_weight = base_weight * local_aas

    aggregation_weights.append(local_weight)

    local_trust = compute_trust_score(local_id)

    save_aggregation_history(

        client_id=local_id,

        round_number=COMMUNICATION_ROUND.get(local_id,0),

        aggregation_weight=local_weight,

        aas=local_aas,

        trust=local_trust,

        energy=LAST_ENERGY.get(local_id,100)/100,

        cus=CLIENT_CUS.get(local_id,0),

        reliability=CLIENT_RELIABILITY.get(local_id,1),

        privacy=CLIENT_PRIVACY_RISK.get(local_id,0)

      )

    print()

    print("=" * 70)
    print("[POLICY-BASED AGGREGATION WEIGHTS]")
    print("=" * 70)

    for cid, weight in zip(peer_ids + [local_id], aggregation_weights):

        print(f"Client {cid} : {weight:.3f}")

    print("=" * 70)

    return aggregation_weights

def average_weights(weights_list, aggregation_weights=None):

    avg_weights = []

    if aggregation_weights is None:

        aggregation_weights = [1.0] * len(weights_list)

    total_weight = sum(aggregation_weights)

    normalized = [

        w / total_weight

        for w in aggregation_weights

    ]

    for layer in zip(*weights_list):

        weighted_layer = np.zeros_like(layer[0])

        for model, weight in zip(layer, normalized):

            weighted_layer += model * weight

        avg_weights.append(weighted_layer)
    
    print()
    print("="*60)
    print("[IMOCC AGGREGATION]")
    print(f"Aggregation Weights : {aggregation_weights}")
    print("="*60)

    return avg_weights

def compute_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def models_are_similar(weights1, weights2, threshold):
    for w1, w2 in zip(weights1, weights2):
        norm = np.linalg.norm(w1 - w2)
        if norm > threshold:
            return False
    return True

# ==========================================================
# Phase 5.1
# Final Communication Statistics
# ==========================================================

def print_final_statistics(client_id):

    print()
    print("=" * 80)
    print("FINAL IMOCC COMMUNICATION SUMMARY")
    print("=" * 80)

    if client_id in CLIENT_HISTORY:

        avg_cus = np.mean(CLIENT_HISTORY[client_id])

    else:

        avg_cus = 0

    if client_id in ENERGY_HISTORY:

        energies = [e for (_, e) in ENERGY_HISTORY[client_id]]
        avg_energy = np.mean(energies)

    else:

        avg_energy = 0

    if client_id in AGGREGATION_HISTORY:

        avg_weight = np.mean([
            x["aggregation_weight"]
            for x in AGGREGATION_HISTORY[client_id]
        ])

        avg_trust = np.mean([
            x["trust"]
            for x in AGGREGATION_HISTORY[client_id]
        ])

        avg_aas = np.mean([
            x["aas"]
            for x in AGGREGATION_HISTORY[client_id]
        ])

    else:

        avg_weight = 0
        avg_trust = 0
        avg_aas = 0

    print(f"Average Energy              : {avg_energy:.2f}")

    print(f"Average Communication Score : {avg_cus:.4f}")

    print(f"Average Trust Score         : {avg_trust:.4f}")

    print(f"Average Aggregation Score   : {avg_aas:.4f}")

    print(f"Average Aggregation Weight  : {avg_weight:.4f}")

    print("=" * 80)
    

def print_round_summary(client_id, round_no, accuracy, loss, aggregation_weights):

    policy = CLIENT_POLICY[client_id].get_policy()

    print()
    print("=" * 90)
    print(f"CLIENT {client_id} | ROUND {round_no} SUMMARY")
    print("=" * 90)

    print(f"Accuracy                 : {accuracy:.2f}%")
    print(f"Loss                     : {loss:.4f}")
    print(f"Energy Score             : {LAST_ENERGY[client_id]:.2f}")
    print(f"CUS                      : {CLIENT_CUS[client_id]:.4f}")
    print(f"Reliability              : {CLIENT_RELIABILITY[client_id]:.4f}")
    print(f"Trust Score              : {compute_trust_score(client_id):.4f}")
    print(f"Adaptive Aggregation     : {compute_adaptive_aggregation_score(client_id):.4f}")

    print("-" * 90)

    print(f"Density                  : {policy['density']:.4f}")
    print(f"Compression              : {policy['compression']:.2f}")
    print(f"Aggregation Weight       : {aggregation_weights[-1]:.4f}")
    print(f"Priority                 : {policy['priority']}")
    print(f"Communication Mode       : {policy['communication_mode']}")
    print(f"Token Expiry             : {policy['token_expiry']}")

    print("=" * 90)
    print()


def client_logic(id, local_ip, ips, faults):
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    previous_weights = None
    current_round = 0
    FORCE_ENERGY_UPDATE[id] = False
    CLIENT_CRASH_STATUS[id] = False
    received_weights = []
    terminate_flags = []
    counter = 0
    crash_counter = 0
    latest_models = {}
    crash_away_list = [False] * NUM_CLIENTS
    prev_list = [[] for _ in range(NUM_CLIENTS)]
    crashed_in_rounds = []  # Track rounds with new crashes

    server_thread = threading.Thread(target=tcp_server, args=(id, received_weights, terminate_flags, local_ip, latest_models, crash_away_list, prev_list))
    server_thread.start()
    time.sleep(5)

    while current_round < R_PRIME:
        print("\n" + "="*90)
        print(f"ROUND {current_round}")
        print("="*90)
        
        model.train()
        for epoch in range(EPOCHS_PER_ROUND):
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                CLIENT_PREV_LOSS[id] = float(loss.item())

        weights = [param.cpu().detach().numpy() for param in model.parameters()]
        if (
              current_round % ENERGY_UPDATE_INTERVAL == 0
              or id not in LAST_ENERGY
              or FORCE_ENERGY_UPDATE[id]
         ):

             cpu_usage = psutil.cpu_percent()

             memory_usage = (
                    psutil.virtual_memory().percent
             )

             energy_score = update_gateway(
                  id,
                  current_round,
                  cpu_usage,
                  memory_usage,
                  crash_counter
             )
             
             density = (
                  0.01
                  +
                   0.04*(energy_score/100)
             )

             LAST_DENSITY[id] = density

             LAST_ENERGY[id] = energy_score
             cus = compute_communication_utility(
                    client_id=id,
                    energy=energy_score,
                    loss=CLIENT_PREV_LOSS.get(id, 0),
                    bytes_sent=CLIENT_BYTES.get(id, 0),
                    crash=FORCE_ENERGY_UPDATE[id]
             )
             CLIENT_HISTORY.setdefault(id, []).append(cus)
             FORCE_ENERGY_UPDATE[id] = False
             CLIENT_CRASH_STATUS[id] = False

        else:
            energy_score = LAST_ENERGY[id]
            density = LAST_DENSITY[id]
            
            
        
        
        # Check if this client should crash
        for fault in faults:
            if fault[0] == id and fault[1] == current_round:
                for _ in range(fault[2]):
                    broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
                print(f"Client {id} is crashing at round {current_round}")
                return

        # Check if termination flag is received from other clients
        if terminate_flags:
            print(f"Client {id} received termination flag at round {current_round}")
            broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
            break

        broadcast_weights(id, weights, current_round, terminate=0, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)

        t_start = time.time()
        while (time.time() - t_start) < TIMEOUT:
            time.sleep(0.01)

        # Update crash_away_list and detect new crashes for logging purposes only
        new_crashes = []
        for client_id in range(NUM_CLIENTS):
            received_ids = []
            for msg in received_weights:
                if "id" in msg:
                    received_ids.append(
                                 msg["id"]
                                 )

                elif "sender_id" in msg:
                    received_ids.append(
                    msg["sender_id"]
                 )

            if (
                    client_id not in received_ids
                    and not crash_away_list[client_id]
                    and client_id != id
                  ):
                crash_away_list[client_id] = True
                new_crashes.append(client_id)
                print(f"Client {id} detected crash of client {client_id} at round {current_round}")

        if len(new_crashes) > 0:
            
            crashed_in_rounds.append(current_round)
            
            crash_counter = 0
            
            FORCE_ENERGY_UPDATE[id] = True
            CLIENT_CRASH_STATUS[id] = True

   
    

            print(
                 f"[CRASH DETECTED] "
                 f"Round={current_round}"
                )

            print(
                 "[ENERGY WILL BE "
                  "RECALCULATED NEXT ROUND]"
                 )

        else:
            crash_counter += 1

        # Include all received weights, even from crashed clients in previous rounds
        reconstructed_models = []

        for msg in received_weights:
            if msg["type"] == "weights":
                reconstructed_models.append(msg["weights"])

            else:
                peer_model = apply_privGAT_token(weights, msg, current_round)

                # STORE reconstructed peer model
                sender_id = msg.get("sender_id", 0)

                latest_models[sender_id] = peer_model

                

                reconstructed_models.append(peer_model)
        
        print("\n--- COMMUNICATION STATUS ---")
        print(f"Energy Score        : {LAST_ENERGY[id]:.2f}")
        print(f"Token Density       : {LAST_DENSITY[id]:.4f}")
        print(f"Privacy Risk        : {CLIENT_PRIVACY_RISK.get(id, 0):.4f}")
        print(f"Communication Score : {CLIENT_CUS.get(id, 0):.4f}")
        print(f"Reliability         : {CLIENT_RELIABILITY.get(id, 1.0):.2f}")
        print(f"Received Models     : {len(reconstructed_models)}")

        total_weights = reconstructed_models + [weights]
        peer_ids = []

        for msg in received_weights:
            if msg["type"] == "weights":
                peer_ids.append(msg["id"])

            else:
                peer_ids.append(msg["sender_id"])
        
        state = imocc.collect_state(
               client_id=id,
               energy=LAST_ENERGY.get(id, 100),
               cus=CLIENT_CUS.get(id, 0),
               reliability=CLIENT_RELIABILITY.get(id, 1.0),
               privacy_risk=CLIENT_PRIVACY_RISK.get(id, 0),
               crashed=CLIENT_CRASH_STATUS.get(id, False),
               current_round=current_round
             )
        COMMUNICATION_ROUND[id] = current_round
        policy = imocc.generate_policy(id)

        current_policy = policy.get_policy()
        
        print("\n--- IMOCC DECISION ---")
        print(current_policy.keys())
        print(f"Mode                : {current_policy['communication_mode']}")
        print(f"Priority            : {current_policy['priority']}")
        print(f"Density             : {current_policy['density']:.4f}")
        print(f"Compression         : {current_policy['compression']:.2f}")
        print(f"Aggregation Weight  : {current_policy['aggregation_weight']:.2f}")
        print(f"Expiry              : {current_policy['token_expiry']}")
        scores = modo.objective_scores(state,policy.get_policy())
        print("\n--- MULTI OBJECTIVE OPTIMIZATION ---")

        print(f"Energy Objective        : {scores['energy']:.3f}")
        print(f"Communication Objective : {scores['communication']:.3f}")
        print(f"Learning Objective      : {scores['learning']:.3f}")
        print(f"Reliability             : {scores['reliability']:.3f}")
        print(f"Privacy                 : {scores['privacy']:.3f}")
        
        modo.save_history(id,current_round,scores,policy.get_policy())
        pareto = modo.pareto_select(state)
        
        
        trust = compute_trust_score(id)
        aas = compute_adaptive_aggregation_score(id)
        record_decision_timeline(client_id=id,round_number=current_round,policy=policy.get_policy(),optimization_score=pareto["score"],trust=trust,aas=aas)

        aggregation_weights = build_aggregation_weights(peer_ids,id)
        
        trust = compute_trust_score(id)

        aas = compute_adaptive_aggregation_score(id)

        update_global_statistics(

             client_id=id,

             round_number=current_round,

             density=CLIENT_POLICY[id].get_policy()["density"],

             aggregation_weight=aggregation_weights[-1],

             energy=LAST_ENERGY[id] / 100,

             cus=CLIENT_CUS[id],

             trust=trust,

             aas=aas,

             crashed=CLIENT_CRASH_STATUS[id]

        )

        print("\n--- MODEL AGGREGATION ---")

        print(f"Models Aggregated : {len(total_weights)}")

        for i,w in enumerate(aggregation_weights):
            print(f"Weight {i:<2}: {w:.4f}")

        new_weights = average_weights(

        total_weights,

        aggregation_weights

        )
        for param, new_weight in zip(model.parameters(), new_weights):
            param.data = torch.tensor(new_weight).to(DEVICE)

        accuracy = compute_accuracy(model, test_loader)

        print_round_summary(
               client_id=id,
               round_no=current_round,
               accuracy=accuracy,
               loss=CLIENT_PREV_LOSS[id],
               aggregation_weights=aggregation_weights
             )
        
        # ==========================================================
        # IMOCC Integration
        # ==========================================================

        

        if id not in POLICY_HISTORY:
            POLICY_HISTORY[id] = []
        POLICY_HISTORY[id].append(
            policy.get_policy()
        )

        # Only check for termination criteria after minimum rounds
        if current_round >= MINIMUM_ROUNDS:
            if previous_weights is not None and models_are_similar(new_weights, previous_weights, THRESHOLD):
                counter += 1
            else:
                counter = 0

            # Check if no new crashes in the last COUNT_THRESHOLD rounds
            no_recent_crashes = True
            for r in range(current_round - COUNT_THRESHOLD + 1, current_round + 1):
                if r in crashed_in_rounds:
                    no_recent_crashes = False
                    break

            if counter >= COUNT_THRESHOLD and no_recent_crashes:
                print(f"Client {id} met termination criteria at round {current_round}: stable weights for {COUNT_THRESHOLD} rounds and no crashes")
                broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)
                break
         
         
        print("\n" + "-"*90)
        print("ROUND SUMMARY")
        print("-"*90)

        print(f"Energy Score        : {LAST_ENERGY[id]:.2f}")
        print(f"CUS                 : {CLIENT_CUS[id]:.3f}")
        print(f"Trust               : {compute_trust_score(id):.3f}") 
        print(f"AAS                 : {compute_adaptive_aggregation_score(id):.3f}")
        print(f"Density             : {CLIENT_POLICY[id].get_policy()['density']:.4f}")
        print(f"Aggregation Weight  : {CLIENT_POLICY[id].get_policy()['aggregation_weight']:.3f}")
        print(f"Accuracy            : {accuracy:.2f}%")
        print("="*90)
        previous_weights = new_weights
        current_round += 1
        received_weights.clear()

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_weights(id, weights, current_round, terminate=1, ips=ips, latest_models=latest_models, crash_away_list=crash_away_list, prev_list=prev_list)

    broadcast_terminate(id, ips)
    modo.print_history(id)
    print_aggregation_history(id)
    print_final_statistics(id)
    print_decision_timeline(id)
    server_thread.join()

def main():
    global model_messages, terminate_messages
    start_time = time.time()
    print( f"Starting Federated Learning on Machine ({CURRENT_MACHINE_IP})")

    threads = []
    for i in range(NUM_CLIENTS):
        if ips[i] == str(CURRENT_MACHINE_IP):
            client_thread = threading.Thread(target=client_logic, args=(i, CURRENT_MACHINE_IP, ips, faults))
            threads.append(client_thread)
            client_thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    total_time = end_time - start_time
   
    total_model_messages = sum(model_messages)
    total_termination_messages = sum(terminate_messages)

    print("\nFederated Learning Completed")
    print("\n===== COMMUNICATION =====")
    actual_bytes = TOKEN_BYTES + FULL_WEIGHT_BYTES

    saved_bytes = ORIGINAL_BYTES - actual_bytes

    compression_percent = (
    saved_bytes / max(ORIGINAL_BYTES, 1)
    ) * 100

    print(f"Original Bytes: {ORIGINAL_BYTES}")
    print(f"Actual Bytes Sent: {actual_bytes}")

    print(f"Token Bytes: {TOKEN_BYTES}")
    print(f"Full Bytes: {FULL_WEIGHT_BYTES}")

    print(f"Saved Bytes: {saved_bytes}")

    print(f"Bandwidth Reduction: {compression_percent:.2f}%")

    print(
       f"Compression Ratio: "
       f"{actual_bytes / max(ORIGINAL_BYTES,1):.4f}"
     )
    print("Current Machine IP:", CURRENT_MACHINE_IP)
    print("Number of Clients:", NUM_CLIENTS)
    print(f"Total model messages passed: {total_model_messages-((NUM_CLIENTS//2)*(NUM_CLIENTS-1))-total_termination_messages}")
    print("Total Termination Messages Passed:", total_termination_messages)
    print(f"Total Time Taken: {total_time:.2f} seconds")
    
    print_global_statistics()
    print_final_performance_report()
    print_experiment_summary()



if __name__ == "__main__":
    main()