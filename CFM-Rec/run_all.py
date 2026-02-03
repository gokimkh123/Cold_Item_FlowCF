# run_all.py
import os

# [수정] 여기서 'popularity' 또는 'noise'를 선택하세요.
PRIOR_TYPE = 'noise' 

# 실험할 스텝 리스트
step_list = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60, 70,80,90,100,200,300]

#step_list = [1 ]
print(f"🚀 [{PRIOR_TYPE.upper()} 실험] CFM-Rec 및 diffusion 실험을 시작합니다.")

for step in step_list:
    
    # --- 1. CFM-Rec (Flow) 실행 ---
    print(f"\n[Flow - {PRIOR_TYPE}] Running with steps = {step} ...")
    # f-string을 이용해 PRIOR_TYPE 변수를 전달합니다.
    flow_cmd = f"python train.py --steps {step} --prior_type {PRIOR_TYPE}"
    os.system(flow_cmd)
    
    # --- 2. diffusion (DDPM) 실행 ---
    print(f"\n[Diffusion - {PRIOR_TYPE}] Running with steps = {step} ...")
    ddpm_cmd = f"python -m src_ddpm.train_ddpm --steps {step} --prior_type {PRIOR_TYPE}"
    os.system(ddpm_cmd)

print(f"\n모든 {PRIOR_TYPE} 실험이 완료되었습니다!")