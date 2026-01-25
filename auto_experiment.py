# run_comparison.py
import os
import subprocess
import re
import matplotlib.pyplot as plt
import yaml

# ==========================================
# 1. 실험 설정
# ==========================================
# DiffCF는 학습 시 n_steps가 고정되므로, 이 리스트만큼 재학습합니다.
diff_steps_list = [10, 20, 50, 100, 300] 

# FlowCF는 한 번 학습 후, 추론 시 이 리스트만큼 스텝을 바꿔가며 평가합니다.
flow_inference_steps = [1, 2, 5, 10, 20, 50, 100]

common_epochs = 100

# 결과 저장용
results_diff = {'steps': [], 'recall10': [], 'recall20': [], 'time': []}
results_flow = {'steps': [], 'recall10': [], 'recall20': [], 'time': []}

def update_yaml(file_path, n_steps, epochs):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['n_steps'] = n_steps
    config['epochs'] = epochs
    config['stopping_step'] = 0 # Early Stopping 끄기
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)

def run_command(cmd):
    """커맨드를 실행하고 __PARSE_RESULT__ 라인을 파싱해서 리턴"""
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 로그 파일에 저장 (디버깅용)
    with open("experiment.log", "a") as f:
        f.write(f"\n\nCMD: {cmd}\n")
        f.write(result.stdout)
        
    # 파싱
    match = re.search(r"__PARSE_RESULT__:(.*)", result.stdout)
    if match:
        data = match.group(1).split(',')
        return float(data[0]), float(data[1]), float(data[2]) # r@10, r@20, time
    else:
        print("Warning: Failed to parse result.")
        return 0.0, 0.0, 0.0

# ==========================================
# 2. DiffCF 실험 (각 Step마다 재학습 필요)
# ==========================================
print("========== Starting DiffCF Experiments ==========")
for steps in diff_steps_list:
    print(f"\n[DiffCF] Training with n_steps={steps}...")
    
    # 1. Config 수정
    update_yaml('diffcf.yaml', steps, common_epochs)
    
    # 2. 학습 (run.py)
    # run.py가 완료되면 saved/ 폴더에 .pth가 생김
    # 가장 최근에 생긴 .pth 파일을 찾아야 함
    # 기존 파일 꼬임 방지를 위해 saved 폴더 비우기 권장 (여기서는 생략하고 시간순 정렬로 찾음)
    subprocess.run("python run.py --config diffcf.yaml --act leakyrelu", shell=True)
    
    # 3. 최신 체크포인트 찾기
    list_of_files = os.listdir('saved')
    paths = [os.path.join('saved', basename) for basename in list_of_files if basename.endswith('.pth')]
    latest_file = max(paths, key=os.path.getctime)
    
    # 4. 평가 (evaluate.py)
    # DiffCF는 학습된 n_steps를 그대로 사용하므로 --steps 옵션 안 줌
    r10, r20, inf_time = run_command(f"python evaluate.py --config diffcf.yaml --checkpoint {latest_file}")
    
    results_diff['steps'].append(steps)
    results_diff['recall10'].append(r10)
    results_diff['recall20'].append(r20)
    results_diff['time'].append(inf_time)

# ==========================================
# 3. FlowCF 실험 (1번 학습 -> N번 평가)
# ==========================================
print("\n========== Starting FlowCF Experiments ==========")
# 1. 학습용 Config 설정 (학습은 n_steps=20 정도로 고정해도 됨, 어차피 Continuous라 무관)
update_yaml('flowcf.yaml', 20, common_epochs)

print("[FlowCF] Training Once...")
subprocess.run("python run.py --config flowcf.yaml --act leakyrelu", shell=True)

# 2. 체크포인트 찾기
list_of_files = os.listdir('saved')
paths = [os.path.join('saved', basename) for basename in list_of_files if basename.endswith('.pth')]
latest_file = max(paths, key=os.path.getctime)

# 3. Step 바꿔가며 평가
for steps in flow_inference_steps:
    print(f"\n[FlowCF] Evaluating with inference steps={steps}...")
    # --steps 옵션으로 추론 스텝 강제 변경
    r10, r20, inf_time = run_command(f"python evaluate.py --config flowcf.yaml --checkpoint {latest_file} --steps {steps}")
    
    results_flow['steps'].append(steps)
    results_flow['recall10'].append(r10)
    results_flow['recall20'].append(r20)
    results_flow['time'].append(inf_time)

# ==========================================
# 4. 그래프 그리기
# ==========================================
import matplotlib.pyplot as plt

# 그래프 1: 수렴 속도 비교 (Metric vs Steps)
plt.figure(figsize=(10, 6))
plt.plot(results_diff['steps'], results_diff['recall20'], 'o--', label='DiffCF (Retrained)', color='red', linewidth=2)
plt.plot(results_flow['steps'], results_flow['recall20'], 's-', label='FlowCF (Inference Varying)', color='blue', linewidth=2)

# 최고점 강조
max_diff_idx = results_diff['recall20'].index(max(results_diff['recall20']))
max_flow_idx = results_flow['recall20'].index(max(results_flow['recall20']))

plt.plot(results_diff['steps'][max_diff_idx], results_diff['recall20'][max_diff_idx], 'r*', markersize=15)
plt.plot(results_flow['steps'][max_flow_idx], results_flow['recall20'][max_flow_idx], 'b*', markersize=15)

plt.xscale('log') # 스텝 차이가 크므로 로그 스케일 추천
plt.xlabel('Number of Steps (log scale)', fontsize=12)
plt.ylabel('Recall@20', fontsize=12)
plt.title('Performance Convergence: FlowCF vs DiffCF', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(diff_steps_list + [1, 2, 5], labels=diff_steps_list + [1, 2, 5])
plt.savefig('n_step_convergence.png')
print("Saved n_step_convergence.png")

# 그래프 2: 추론 효율성 비교 (Metric vs Inference Steps)
# FlowCF의 장점(적은 스텝으로 고성능)을 강조하는 그래프
plt.figure(figsize=(10, 6))

plt.plot(results_flow['steps'], results_flow['recall20'], 'b-o', label='FlowCF', linewidth=2)
plt.plot(results_diff['steps'], results_diff['recall20'], 'r--s', label='DiffCF', linewidth=2, alpha=0.6)

plt.xlabel('Inference Steps (s_step)', fontsize=12)
plt.ylabel('Recall@20', fontsize=12)
plt.title('Inference Efficiency: Performance vs Steps', fontsize=14)
plt.legend()
plt.grid(True)
plt.xlim(0, 105) # 100 step까지만 보여줘서 앞쪽 강조
plt.savefig('inference_efficiency.png')
print("Saved inference_efficiency.png")

# 그래프 3 (보너스): 실제 시간 vs 성능 (Real Speed Trade-off)
plt.figure(figsize=(10, 6))
plt.plot(results_flow['time'], results_flow['recall20'], 'b-o', label='FlowCF')
plt.plot(results_diff['time'], results_diff['recall20'], 'r--s', label='DiffCF')
plt.xlabel('Inference Time (seconds)', fontsize=12)
plt.ylabel('Recall@20', fontsize=12)
plt.title('Real-World Speed-Accuracy Trade-off', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('real_time_tradeoff.png')
print("Saved real_time_tradeoff.png")