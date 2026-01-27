# evaluate.py
import argparse
import yaml
import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich import box

# Custom Modules
from src.data_loader import ColdStartDataLoader
from src.model import FlowModel
from src.flow_logic import BernoulliFlow

console = Console()

def compute_metrics(top_k_items, ground_truth_items, k_list=[10, 20]):
    """
    User-to-Item Metrics Calculation
    Returns: Recall, NDCG, Precision, Hit Rate
    """
    gt_set = set(ground_truth_items)
    
    # 정답이 없으면 모든 지표 0
    if not gt_set:
        return {
            f"Recall@{k}": 0.0 for k in k_list
        } | {
            f"NDCG@{k}": 0.0 for k in k_list
        } | {
            f"Precision@{k}": 0.0 for k in k_list
        } | {
            f"Hit@{k}": 0.0 for k in k_list
        }

    # Hit 여부 리스트 (1 or 0)
    hit_list = []
    for item_id in top_k_items:
        if item_id in gt_set:
            hit_list.append(1)
        else:
            hit_list.append(0)

    results = {}
    for k in k_list:
        # k개까지만 잘라서 계산
        k_hit_list = hit_list[:k]
        k_hits = sum(k_hit_list)
        
        # 1. Recall
        results[f"Recall@{k}"] = k_hits / len(gt_set)
        
        # 2. NDCG
        k_sum_r = sum([1.0 / np.log2(i + 2) for i, is_hit in enumerate(k_hit_list) if is_hit])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_set), k))])
        results[f"NDCG@{k}"] = k_sum_r / idcg if idcg > 0 else 0.0
        
        # 3. Precision (맞춘 개수 / 추천 개수 K)
        results[f"Precision@{k}"] = k_hits / k
        
        # 4. Hit Rate (하나라도 맞췄으면 1)
        results[f"Hit@{k}"] = 1.0 if k_hits > 0 else 0.0

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="FlowCF Evaluation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--s_step', type=int, default=None, help='Override inference steps')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    return parser.parse_args()

def main():
    console.print(Panel.fit(
        "[bold cyan]FlowCF Evaluation (User-to-Item)[/bold cyan]", 
        subtitle="[dim]Metrics: Recall, NDCG, Precision, Hit Rate[/dim]",
        border_style="blue"
    ))

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inference_steps = args.s_step if args.s_step is not None else config.get('inference_steps', 10)
    
    config_table = Table(title="Configuration", box=box.SIMPLE)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_row("Inference Steps", str(inference_steps))
    console.print(config_table)
    console.print()

    # 1. Load Data & Model
    with console.status("[bold green]Loading Data & Model...", spinner="dots"):
        loader = ColdStartDataLoader(config)
        num_items, num_users = loader.build()
        
        model = FlowModel(config['dims_mlp'] + [num_users], config['time_embedding_size'])
        _ = model(tf.zeros((1, num_users)), tf.zeros((1, loader.side_dim)), tf.zeros((1, 1)), training=False)
        
        checkpoint_path = "saved_model/best_flow_model"
        try:
            model.load_weights(checkpoint_path).expect_partial()
            console.print(f"[bold blue]Best Model Loaded:[/bold blue] {checkpoint_path}")
        except:
            latest = tf.train.latest_checkpoint("saved_model/")
            if latest:
                model.load_weights(latest).expect_partial()
                console.print(f"[bold yellow]Loaded latest checkpoint:[/bold yellow] {latest}")
            else:
                console.print("[bold red] No model found![/bold red]")
                return

    flow = BernoulliFlow(loader.user_activity)
    
    # 2. Load Test Data
    test_path = os.path.join(config['data_path'], config['test_file'])
    test_df = pd.read_csv(test_path, sep='\t', dtype={
        f"{config['entity_field']}:token": str, 
        f"{config['target_field']}:token": str
    })
    test_df.columns = [col.split(':')[0] for col in test_df.columns]
    
    user_ground_truth = test_df.groupby(config['target_field'])[config['entity_field']].apply(set).to_dict()
    test_item_tokens = test_df[config['entity_field']].unique()
    valid_test_items = [t for t in test_item_tokens if t in loader.entity2id]
    
    console.print(f"Total Test Users: {len(user_ground_truth)}")
    console.print(f"Total Test Items (Cold): {len(valid_test_items)}")

    # 3. Score Matrix Construction
    score_rows = []
    item_idx_to_token = {} 
    dt = 1.0 / inference_steps
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        
        task_infer = progress.add_task("[cyan]Predicting Scores for Cold Items...", total=len(valid_test_items))
        
        for i, token_str in enumerate(valid_test_items):
            eid = loader.entity2id[token_str]
            cond = tf.expand_dims(loader.side_emb[eid], 0)
            
            x_t = flow.get_prior_sample(1)
            for step in range(inference_steps):
                t_val = step * dt
                t = tf.fill([1, 1], t_val)
                t = tf.cast(t, tf.float32)
                pred = model(x_t, cond, t, training=False)
                x_t = flow.inference_step(x_t, pred, t_val, dt)
            
            scores = x_t.numpy().flatten()
            score_rows.append(scores)
            item_idx_to_token[i] = token_str
            progress.advance(task_infer)

    score_matrix = np.vstack(score_rows) 
    
    # 4. User-wise Ranking & Evaluation
    # 모든 지표를 저장할 딕셔너리
    metric_accumulator = {
        'Recall@10': [], 'Recall@20': [],
        'NDCG@10': [], 'NDCG@20': [],
        'Precision@10': [], 'Precision@20': [],
        'Hit@10': [], 'Hit@20': []
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        
        task_eval = progress.add_task("[magenta]Ranking & Evaluating Users...", total=len(user_ground_truth))
        
        for user_token, true_item_set in user_ground_truth.items():
            if user_token not in loader.target2id:
                progress.advance(task_eval)
                continue
                
            uid_internal = loader.target2id[user_token]
            user_scores = score_matrix[:, uid_internal]
            
            top_indices = np.argsort(user_scores)[-20:][::-1]
            top_item_tokens = [item_idx_to_token[idx] for idx in top_indices]
            
            metrics = compute_metrics(top_item_tokens, true_item_set, k_list=[10, 20])
            
            for k, v in metrics.items():
                metric_accumulator[k].append(v)
            
            progress.advance(task_eval)

    # 5. Final Report Table
    final_metrics = {k: np.mean(v) for k, v in metric_accumulator.items()}
    
    res_table = Table(title="🏆 Final User-to-Item Evaluation Results", box=box.DOUBLE_EDGE)
    res_table.add_column("Metric", justify="center", style="cyan", no_wrap=True)
    res_table.add_column("@10", justify="center", style="magenta")
    res_table.add_column("@20", justify="center", style="bold green")

    # Recall
    res_table.add_row("Recall", f"{final_metrics['Recall@10']:.4f}", f"{final_metrics['Recall@20']:.4f}")
    # NDCG
    res_table.add_row("NDCG", f"{final_metrics['NDCG@10']:.4f}", f"{final_metrics['NDCG@20']:.4f}")
    # Precision (추가됨)
    res_table.add_row("Precision", f"{final_metrics['Precision@10']:.4f}", f"{final_metrics['Precision@20']:.4f}")
    # Hit Rate (추가됨)
    res_table.add_row("Hit Rate", f"{final_metrics['Hit@10']:.4f}", f"{final_metrics['Hit@20']:.4f}")

    console.print()
    console.print(Panel(res_table, border_style="green"))
    console.print("[dim]Evaluation Complete.[/dim]")

if __name__ == "__main__":
    main()