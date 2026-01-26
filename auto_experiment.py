import subprocess
import yaml
import os
import re
import time
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. 설정 및 유틸리티
# =========================================================
# 실험할 N 스텝 리스트 (광범위한 탐색)
COMPARISON_STEPS = [1, 2, 3, 4, 5, 10]
COMMON_EPOCHS = 100
STOPPING_STEP = 15  # Early Stopping Patience

results = {
    'diff_r20': [], 'diff_time': [],
    'flow_best_r20': [], 'flow_best_s': [], 'flow_best_time': [],
    'best_model_n': 0, 'best_model_path': "" # 실험 B를 위해 최고 모델 저장
}

def run_command(cmd, capture=True):
    if capture:
        # [수정] stderr=subprocess.STDOUT 을 추가하여 에러/로그 출력도 모두 캡처
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if res.returncode != 0:
            # 에러가 나도 일단 출력은 리턴해서 디버깅 가능하게 함
            print(f"Command failed: {cmd}\nError output: {res.stdout[:200]}...") 
            return None, res.stdout
        return res.stdout, None
    else:
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0:
            return None, "Process Failed"
        return "", None

def update_yaml(file_path, **kwargs):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    data.update(kwargs)
    with open(file_path, 'w') as f:
        yaml.dump(data, f)

def get_latest_checkpoint(dir="saved/"):
    if not os.path.exists(dir): return None
    files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pth')]
    return max(files, key=os.path.getctime) if files else None
def parse_result(output):
    if not output: return 0.0, 0.0
    
    # [전략 1] evaluate.py가 뱉어주는 '치트키'(__PARSE_RESULT__)를 최우선으로 찾음
    # 형식: __PARSE_RESULT__:recall@10,recall@20,time
    # 예시: __PARSE_RESULT__:0.2034,0.3110,1.54
    match = re.search(r"__PARSE_RESULT__:([\d.,eE\-]+)", output)
    
    if match:
        try:
            content = match.group(1)
            parts = content.split(',')
            # evaluate.py 코드 순서: [0]=r10, [1]=r20, [2]=time
            r20 = float(parts[1]) 
            t_inf = float(parts[2])
            return r20, t_inf
        except (ValueError, IndexError):
            pass # 파싱 실패시 아래로 넘어감

    # [전략 2] 치트키를 못 찾았을 경우를 대비한 기존 로직 (백업)
    # ANSI 색상 제거
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_output = ansi_escape.sub('', output)

    # 패턴 찾기
    r20_match = re.search(r"recall@20\s*[:=]\s*([\d.]+)", clean_output, re.IGNORECASE)
    t_match = re.search(r"test result.*?(\d+\.\d+)s", clean_output, re.S | re.IGNORECASE)

    r20_val = float(r20_match.group(1)) if r20_match else 0.0
    t_val = float(t_match.group(1)) if t_match else 0.0
    
    return r20_val, t_val
# =========================================================
# 2. [실험 A] N 변화에 따른 성능 비교
# =========================================================
print("\n" + "="*60)
print(" PHASE A: Performance Comparison (DiffCF vs FlowCF)")
print(f" Target Steps: {COMPARISON_STEPS}")
print("="*60)

global_best_r20 = -1.0

for i, n in enumerate(COMPARISON_STEPS):
    print(f"\n>>> [A-{i+1}/{len(COMPARISON_STEPS)}] Experimenting at N = {n}")
    
    # --- [A-1] DiffCF ---
    print(f"   [DiffCF] Training...")
    update_yaml('diffcf.yaml', n_steps=n, s_steps=None, epochs=COMMON_EPOCHS, stopping_step=STOPPING_STEP)
    
    # 학습
    out, _ = run_command("python run.py --config diffcf.yaml", capture=False)
    
    # 평가
    diff_val = 0.0
    if out is not None:
        ckpt = get_latest_checkpoint()
        out_eval, _ = run_command(f"python evaluate.py --config diffcf.yaml --checkpoint {ckpt}", capture=True)
        diff_val, diff_t = parse_result(out_eval)
        print(f"      -> DiffCF Result: R@20={diff_val:.4f}")
    
    results['diff_r20'].append(diff_val)

    # --- [A-2] FlowCF ---
    print(f"   [FlowCF] Training...")
    update_yaml('flowcf.yaml', n_steps=n, s_steps=1, epochs=COMMON_EPOCHS, stopping_step=STOPPING_STEP)
    
    out, _ = run_command("python run.py --config flowcf.yaml --act leakyrelu", capture=False)
    
    flow_best_val = 0.0
    flow_best_s = 1
    flow_best_t = 0.0
    
    if out is not None:
        ckpt_flow = get_latest_checkpoint()
        
        # [스마트 탐색 전략] N 크기에 따라 탐색할 S 후보군 결정
        if n <= 20:
            s_candidates = list(range(1, n + 1)) # N이 작으면 전수 조사
        else:
            # N이 크면 효율적 조사 (초반 정밀 + 후반 Stride)
            base = list(range(1, 11)) # 1~10은 무조건
            stride = max(10, n // 10)
            sparse = list(range(stride, n, stride))
            s_candidates = sorted(list(set(base + sparse)))
            if s_candidates[-1] != n: s_candidates.append(n) # S=N 포함
            
        print(f"      -> Searching Best S in: {s_candidates}")
        
        for s in s_candidates:
            update_yaml('flowcf.yaml', n_steps=n, s_steps=s)
            out_eval, _ = run_command(f"python evaluate.py --config flowcf.yaml --checkpoint {ckpt_flow}", capture=True)
            curr_r20, curr_t = parse_result(out_eval)
            
            if curr_r20 > flow_best_val:
                flow_best_val = curr_r20
                flow_best_s = s
                flow_best_t = curr_t
        
        print(f"      -> FlowCF Best: R@20={flow_best_val:.4f} (at S={flow_best_s})")
        
        # [중요] 실험 B를 위해 가장 성능 좋은 모델 저장
        if flow_best_val > global_best_r20:
            global_best_r20 = flow_best_val
            results['best_model_n'] = n
            results['best_model_path'] = ckpt_flow
            print(f"      *** New Best Model Found! (N={n}) ***")

    results['flow_best_r20'].append(flow_best_val)
    results['flow_best_s'].append(flow_best_s)
    results['flow_best_time'].append(flow_best_t)

# [A번 이미지 생성]
try: plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('ggplot')

plt.figure(figsize=(10, 6))
plt.plot(COMPARISON_STEPS, results['diff_r20'], 'o--', label='DiffCF', color='blue', alpha=0.7)
plt.plot(COMPARISON_STEPS, results['flow_best_r20'], 's-', label='FlowCF (Best S)', color='red', linewidth=2)
plt.xscale('log')
plt.xlabel('Training Steps (N) [Log Scale]')
plt.ylabel('Recall@20')
plt.title('Experiment A: Performance Comparison')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig('result_A_performance.png')
print("\n[Saved] result_A_performance.png")


# =========================================================
# 3. [실험 B] 최고 모델의 S-Step 정밀 분석
# =========================================================
best_n = results['best_model_n']
best_ckpt = results['best_model_path']

if best_n > 0 and os.path.exists(best_ckpt):
    print("\n" + "="*60)
    print(f" PHASE B: Efficiency Analysis on Best Model (N={best_n})")
    print(f" Checkpoint: {best_ckpt}")
    print("="*60)
    
    b_s_steps = []
    b_r20s = []
    b_times = []
    
    # [전수 조사] 1부터 N까지 모든 S에 대해 평가
    print(f"   -> Scanning ALL steps from 1 to {best_n}...")
    full_scan_steps = list(range(1, best_n + 1))
    
    # 진행률 표시를 위해 간단히 구현
    for idx, s in enumerate(full_scan_steps):
        if idx % 10 == 0: print(f"      Processing S={s}...", end="\r")
        
        update_yaml('flowcf.yaml', n_steps=best_n, s_steps=s)
        out_eval, _ = run_command(f"python evaluate.py --config flowcf.yaml --checkpoint {best_ckpt}", capture=True)
        r20, t_inf = parse_result(out_eval)
        
        b_s_steps.append(s)
        b_r20s.append(r20)
        b_times.append(t_inf)
    
    print(f"      Done.                                ")

    # [B번 이미지 생성]
    plt.figure(figsize=(10, 6))
    plt.plot(b_s_steps, b_r20s, '-', color='green', linewidth=2)
    plt.scatter(b_s_steps, b_r20s, color='green', s=20, alpha=0.6)
    
    # 최고점 표시
    max_y = max(b_r20s)
    max_x = b_s_steps[b_r20s.index(max_y)]
    plt.plot(max_x, max_y, 'r*', markersize=15, label=f'Peak (S={max_x})')
    
    plt.xlabel(f'Inference Steps (S) [1 ~ {best_n}]')
    plt.ylabel('Recall@20')
    plt.title(f'Experiment B: Efficiency Trade-off (Best Model N={best_n})')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_B_efficiency.png')
    print("\n[Saved] result_B_efficiency.png")

else:
    print("\n[Skip] Experiment B skipped (No valid FlowCF model found).")

print("\nAll Experiments Completed.")