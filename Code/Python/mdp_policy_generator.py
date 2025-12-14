
import json
import numpy as np
import mdptoolbox
import time
import os
import psutil




# -------- Load Input Data --------
with open('scenario.json', 'r') as f:
    data = json.load(f)



modules = data['problem']['modules']
devices = data['problem']['devices']



mod_specs = data['module_specifications']
dev_specs = data['device_specifications']



mdp_param = data['mdp_parameters']
gamma = mdp_param['discount_factor']
eps = mdp_param['convergence_threshold']
reward_compliance = mdp_param['reward_compliance']
overhead_per_sec_lvl = mdp_param['overhead_per_security_level']
breach_penalty_factor = mdp_param['breach_penalty_factor']
resource_shortfall_penalty = mdp_param['resource_shortfall_penalty']



# Load residual breach probability parameters
base_residual = mdp_param.get('base_residual_breach_prob', 0.005)
gap_mult = mdp_param.get('gap_breach_multiplier', 0.2)
impact_factor = mdp_param.get('breach_impact_factor', 100)



num_mod = len(modules)
num_dev = len(devices)



print("\n" + "="*90)
print("COMPLEX MDP POLICY GENERATION WITH BALANCED DISTRIBUTION")
print("="*90)



print("\n[COMPLEX SCENARIO CONFIGURATION]")
print(f"Modules: {len(modules)}")
print(f"Devices: {len(devices)}")
print(f"Discount Factor (γ): {gamma}")
print(f"Convergence Threshold (ε): {eps}")



print(f"\n[SECURITY PARAMETERS]")
print(f"Compliance Reward: {reward_compliance}")
print(f"Breach Penalty Factor: {breach_penalty_factor}")
print(f"Resource Shortfall Penalty: {resource_shortfall_penalty}")
print(f"Base Residual Breach Probability: {base_residual}")
print(f"Gap Breach Multiplier: {gap_mult}")


# -------- PRE-CALCULATE ALL POSSIBLE ALLOCATIONS --------
def pre_allocate_all_scenarios():
    """Pre-compute device utilization for all modules to calculate fair overload penalties"""
    device_cumulative_usage = {dev: {"ram": 0, "mips": 0} for dev in devices}
    
    sorted_modules = sorted(enumerate(modules), key=lambda x: mod_specs[modules[x[0]]].get('attack_criticality', 1))
    
    allocations = {}
    for idx, mod in sorted_modules:
        best_device = None
        min_utilization = float('inf')
        
        for dev in devices:
            test_ram = device_cumulative_usage[dev]["ram"] + mod_specs[mod]["ram_requirement"]
            test_mips = device_cumulative_usage[dev]["mips"] + mod_specs[mod]["mips_requirement"]
            
            util = (test_ram / dev_specs[dev]["ram"]) + (test_mips / dev_specs[dev]["mips"])
            
            if (test_ram <= dev_specs[dev]["ram"] and 
                test_mips <= dev_specs[dev]["mips"] and 
                util < min_utilization):
                min_utilization = util
                best_device = dev
        
        if best_device is None:
            best_penalty = float('inf')
            for dev in devices:
                test_ram = device_cumulative_usage[dev]["ram"] + mod_specs[mod]["ram_requirement"]
                test_mips = device_cumulative_usage[dev]["mips"] + mod_specs[mod]["mips_requirement"]
                
                ram_penalty = max(0, test_ram - dev_specs[dev]["ram"]) * 10
                mips_penalty = max(0, test_mips - dev_specs[dev]["mips"]) * 8
                total_penalty = ram_penalty + mips_penalty
                
                if total_penalty < best_penalty:
                    best_penalty = total_penalty
                    best_device = dev
        
        if best_device:
            device_cumulative_usage[best_device]["ram"] += mod_specs[mod]["ram_requirement"]
            device_cumulative_usage[best_device]["mips"] += mod_specs[mod]["mips_requirement"]
            allocations[mod] = best_device
    
    return allocations, device_cumulative_usage

# Pre-allocate to understand realistic distribution
pre_alloc, baseline_usage = pre_allocate_all_scenarios()

print("\n[BASELINE REALISTIC DISTRIBUTION - FOR REFERENCE]")
print("="*90)
device_dist = {dev: 0 for dev in devices}
for mod, dev in pre_alloc.items():
    device_dist[dev] += 1

for dev, count in sorted(device_dist.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"{dev}: {count} modules ({count/len(modules)*100:.1f}%)")


# -------- Construct Reward & Transition Matrices --------
R = np.zeros((num_mod, num_dev))
P = [np.eye(num_mod) for _ in range(num_dev)]



print("\n" + "="*90)
print("[REWARD MATRIX CALCULATION - BALANCED OVERLOAD PENALTIES]")
print("="*90)



print(f"\nProcessing {num_mod} modules × {num_dev} devices = {num_mod * num_dev} allocations...")



for i, mod in enumerate(modules):
    m_spec = mod_specs[mod]
    attack_criticality = m_spec.get('attack_criticality', 1)
    
    if i % 5 == 0:
        print(f"\n[{i+1}/{num_mod}] Processing modules...")
    
    for j, dev in enumerate(devices):
        d_spec = dev_specs[dev]
        
        # CRITICAL: Security compliance check
        if m_spec["security_requirement"] <= d_spec["security_level"]:
            compliance_reward = reward_compliance
            device_sec_level = d_spec["security_level"]
            residual_breach_prob = base_residual / (device_sec_level + 1)
            breach_penalty = residual_breach_prob * attack_criticality * breach_penalty_factor
        else:
            compliance_reward = 0
            gap = m_spec["security_requirement"] - d_spec["security_level"]
            gap_breach_prob = min(0.95, gap_mult * gap)
            breach_penalty = gap_breach_prob * attack_criticality * breach_penalty_factor
        
        # Per-module resource penalty
        ram_shortfall = max(0, m_spec["ram_requirement"] - d_spec["ram"])
        mips_shortfall = max(0, m_spec["mips_requirement"] - d_spec["mips"])
        resource_penalty = (ram_shortfall + mips_shortfall) / (m_spec["ram_requirement"] + m_spec["mips_requirement"]) * resource_shortfall_penalty if (m_spec["ram_requirement"] + m_spec["mips_requirement"]) > 0 else 0
        
        # Overhead: only penalize weak devices
        security_gap_from_optimal = max(0, 5 - d_spec["security_level"])
        overhead = overhead_per_sec_lvl * security_gap_from_optimal
        
        # ====== BALANCED OVERLOAD PENALTY (REDUCED from ×15,×12) ======
        if mod in pre_alloc and pre_alloc[mod] == dev:
            # This module naturally fits here - minimal overload penalty
            overload_penalty = 0
        else:
            # This is a "forced" allocation that deviates from optimal
            baseline_dev_ram = baseline_usage[dev]["ram"]
            baseline_dev_mips = baseline_usage[dev]["mips"]
            
            ram_to_capacity = d_spec["ram"] - baseline_dev_ram
            mips_to_capacity = d_spec["mips"] - baseline_dev_mips
            
            ram_excess = max(0, m_spec["ram_requirement"] - ram_to_capacity) if ram_to_capacity >= 0 else m_spec["ram_requirement"]
            mips_excess = max(0, m_spec["mips_requirement"] - mips_to_capacity) if mips_to_capacity >= 0 else m_spec["mips_requirement"]
            
            # BALANCED penalty - reduced from (×15,×12) to (×5,×3)
            overload_penalty = (ram_excess * 5 + mips_excess * 3)
        
        # Total Reward = Compliance - Penalties + Overhead
        R[i, j] = compliance_reward - breach_penalty - resource_penalty - overhead - overload_penalty



print("\n" + "="*90)
print("[REWARD MATRIX SUMMARY]")
print("="*90)
print(f"Matrix Shape: {R.shape} ({num_mod} modules × {num_dev} devices)")
print(f"Reward Range: [{np.min(R):.2f}, {np.max(R):.2f}]")
print(f"Mean Reward: {np.mean(R):.2f}")
print(f"\n✓ BALANCED overload penalties applied (×5, ×3)!")
print(f"✓ MDP will distribute intelligently while respecting security!")



# -------- Solve with Value Iteration --------
print("\n" + "="*90)
print("[SOLVING MDP WITH VALUE ITERATION]")
print("="*90)



start_time = time.time()
vi = mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon=eps)
vi.run()
end_time = time.time()



print(f"\n✓ Convergence achieved in {vi.iter} iterations")
print(f"✓ Convergence time: {end_time - start_time:.6f} seconds")



# -------- Extract Policy --------
policy_indices = [int(x) for x in vi.policy]
policy = {modules[i]: devices[idx] for i, idx in enumerate(policy_indices)}



# -------- Calculate Allocation by Device --------
allocation_by_device = {dev: [] for dev in devices}
for i, mod in enumerate(modules):
    dev_idx = policy_indices[i]
    dev_name = devices[dev_idx]
    allocation_by_device[dev_name].append(mod)


# -------- Compute Final Metrics --------
iterations = vi.iter
solve_time = end_time - start_time
mean_policy_reward = float(np.mean([R[i, idx] for i, idx in enumerate(policy_indices)]))


compliance = 0
breach_count = 0
critical_breach_count = 0
non_critical_breach_count = 0
weighted_breach_cost = 0.0
device_utilization = {dev: {"ram_used": 0, "mips_used": 0, "modules": []} for dev in devices}


# Calculate utilization and breaches
for i, mod in enumerate(modules):
    dev_idx = policy_indices[i]
    dev_name = devices[dev_idx]
    m_spec = mod_specs[mod]
    d_spec = dev_specs[dev_name]
    attack_criticality = m_spec.get('attack_criticality', 1)
    
    device_utilization[dev_name]["ram_used"] += m_spec["ram_requirement"]
    device_utilization[dev_name]["mips_used"] += m_spec["mips_requirement"]
    device_utilization[dev_name]["modules"].append(mod)
    
    if m_spec["security_requirement"] <= d_spec["security_level"]:
        compliance += 1
        device_sec_level = d_spec["security_level"]
        residual_breach_prob = base_residual / (device_sec_level + 1)
        residual_cost = residual_breach_prob * attack_criticality * breach_penalty_factor
        weighted_breach_cost += residual_cost
    else:
        breach_count += 1
        gap = m_spec["security_requirement"] - d_spec["security_level"]
        gap_breach_prob = min(0.95, gap_mult * gap)
        breach_cost = gap_breach_prob * attack_criticality * breach_penalty_factor
        weighted_breach_cost += breach_cost
        
        if attack_criticality >= 4:
            critical_breach_count += 1
        else:
            non_critical_breach_count += 1


security_compliance_rate = compliance / num_mod

critical_modules = sum(1 for m in modules if mod_specs[m].get('attack_criticality', 1) >= 4)
non_critical_modules = num_mod - critical_modules

critical_breach_rate = (critical_breach_count / critical_modules * 100) if critical_modules > 0 else 0
non_critical_breach_rate = (non_critical_breach_count / non_critical_modules * 100) if non_critical_modules > 0 else 0

# Calculate policy file size
policy_file_tmp = "mdp_policy.json.tmp"
with open(policy_file_tmp, "w") as f:
    json.dump({"policy": policy}, f)
policy_size_kb = os.path.getsize(policy_file_tmp) / 1024
os.remove(policy_file_tmp)
memory_overhead_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# -------- VERIFICATION PRINTOUT --------
print("\n" + "="*90)
print("[FINAL ALLOCATION RESULTS - BALANCED]")
print("="*90)


print("\nDEVICE DISTRIBUTION:")
device_module_count = {dev: len(allocation_by_device[dev]) for dev in devices}
for dev, count in sorted(device_module_count.items(), key=lambda x: -x[1]):
    if count > 0:
        pct = (count / len(modules)) * 100
        print(f"  {dev:25s}: {count:2d} modules ({pct:5.1f}%)")


print(f"\nRESOURCE UTILIZATION:")
for dev, util in sorted(device_utilization.items(), key=lambda x: -len(x[1]["modules"])):
    if util["modules"]:
        ram_percent = (util["ram_used"] / dev_specs[dev]["ram"]) * 100
        mips_percent = (util["mips_used"] / dev_specs[dev]["mips"]) * 100
        print(f"  {dev:25s}: RAM {ram_percent:5.1f}% | MIPS {mips_percent:5.1f}%")


print(f"\nPERFORMANCE SUMMARY:")
print(f"Total Reward: {float(np.sum([R[i, idx] for i, idx in enumerate(policy_indices)])):.2f}")
print(f"Mean Reward: {mean_policy_reward:.4f}")
print(f"Security Compliance: {compliance}/{num_mod} = {security_compliance_rate*100:.1f}%")
print(f"Security Breaches: {breach_count}")
print(f"  - Critical Breaches: {critical_breach_count} ({critical_breach_rate:.1f}%)")
print(f"  - Non-Critical Breaches: {non_critical_breach_count} ({non_critical_breach_rate:.1f}%)")
print(f"Weighted Breach Cost: {weighted_breach_cost:.2f}")


# -------- Final Output --------
output = {
    "metadata": {
        "solver": "mdptoolbox.ValueIteration",
        "problem_complexity": f"{num_mod}_modules_{num_dev}_devices",
        "convergence_iterations": int(iterations),
        "time_to_converge_sec": round(solve_time, 6),
        "policy_size_kb": round(policy_size_kb, 4),
        "memory_overhead_mb": round(memory_overhead_mb, 4),
        "mdp_parameters": mdp_param,
        "approach": "Security-aware MDP with Balanced Distribution & Optimized Penalties"
    },
    "metrics": {
        "security_compliance_rate": round(security_compliance_rate, 4),
        "breach_count": breach_count,
        "mean_policy_reward": round(mean_policy_reward, 4),
        "weighted_breach_cost": round(weighted_breach_cost, 2),
        "critical_breach_count": critical_breach_count,
        "non_critical_breach_count": non_critical_breach_count,
        "total_modules": num_mod,
        "total_devices": num_dev
    },
    "policy": policy,
    "resource_utilization": device_utilization,
    "verification": {
        "individual_rewards": {modules[i]: float(R[i, policy_indices[i]]) for i in range(num_mod)},
        "total_reward": float(np.sum([R[i, idx] for i, idx in enumerate(policy_indices)]))
    }
}


with open("mdp_policy.json", "w") as f:
    json.dump(output, f, indent=2)


print("\n" + "="*90)
print("✅ FINAL MDP policy generated with BALANCED penalties")
print("✅ Realistic distribution across cloud/fog/edge")
print("✅ Saved as: mdp_policy.json")
print("="*90 + "\n")