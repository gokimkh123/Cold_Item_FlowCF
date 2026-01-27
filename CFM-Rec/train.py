# train.py
import tensorflow as tf
import yaml
import os
import glob
import numpy as np
import time
import datetime # [추가] 시간별 로그 폴더 생성을 위해

# Rich UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn, 
    TaskProgressColumn
)
from rich import box

# Custom Modules
from src.data_loader import ColdStartDataLoader
from src.model import FlowModel
from src.flow_logic import BernoulliFlow
from src.metrics import compute_metrics

console = Console()

def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@tf.function
def train_step(model, optimizer, x_1, cond, t, x_0):
    mask = tf.cast(tf.random.uniform(tf.shape(x_1)) < t, tf.float32)
    x_t = mask * x_1 + (1.0 - mask) * x_0
    
    with tf.GradientTape() as tape:
        pred = model(x_t, cond, t, training=True)
        loss = tf.reduce_mean(tf.square(x_1 - pred))
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train():
    # --- Header ---
    console.print(Panel.fit(
        "[bold cyan]FlowCF Training Pipeline[/bold cyan]", 
        subtitle="[dim]with TensorBoard Monitoring[/dim]",
        border_style="blue"
    ))

    # 0. Clean previous checkpoints
    console.print("[dim]>>> Cleaning previous best model checkpoints...[/dim]")
    for f in glob.glob("saved_model/best_flow_model*"):
        try: os.remove(f)
        except OSError: pass

    # 1. Load Data
    with console.status("[bold green]Loading Data & Building Matrix...", spinner="dots"):
        config = load_config()
        loader = ColdStartDataLoader(config)
        num_items, num_users = loader.build()
        
        train_ds = loader.get_dataset(mode='train')
        vali_ds = loader.get_dataset(mode='vali')
        
        user_activity = tf.convert_to_tensor(loader.user_activity, dtype=tf.float32)
        flow = BernoulliFlow(loader.user_activity)

    # 2. Initialize Model
    model_dims = config['dims_mlp'] + [num_users]
    model = FlowModel(model_dims, config['time_embedding_size'], config.get('dropout', 0.0))
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    # ---------------- [TensorBoard 설정 추가] ----------------
    # 로그를 저장할 폴더 이름 (시간별로 구분)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/gradient_tape/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    console.print(f"[bold yellow]📊 TensorBoard Log Dir:[/bold yellow] {log_dir}")
    # ---------------------------------------------------------

    # Parameters
    n_step = config.get('n_step', 100)
    inference_steps = config.get('inference_steps', 10)
    eval_step = config.get('eval_step', 1)
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    config_table = Table(title="Training Configuration", box=box.SIMPLE, show_header=True)
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="bold green")
    
    config_table.add_row("Epochs", str(epochs))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Log Directory", log_dir)
    
    console.print(config_table)
    console.print()

    best_recall = -1.0
    patience_cnt = 0
    steps_per_epoch = int(np.ceil(loader.num_entities / batch_size))

    # --- Training Loop ---
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[info]}"),
        console=console
    )

    with progress:
        overall_task = progress.add_task("[bold magenta]Total Progress", total=epochs, info="")
        epoch_task = progress.add_task("[cyan]Current Epoch", total=steps_per_epoch, info="Loss: N/A")

        for epoch in range(epochs):
            progress.reset(epoch_task)
            progress.update(epoch_task, description=f"[cyan]Epoch {epoch+1}/{epochs}")
            
            train_loss = 0
            train_steps = 0
            
            # 1. Train
            for x_1, cond in train_ds:
                current_batch_size = tf.shape(x_1)[0]
                
                t_indices = tf.random.uniform((current_batch_size, 1), minval=1, maxval=n_step+1, dtype=tf.int32)
                t = tf.cast(t_indices, tf.float32) / n_step
                
                activity_probs = tf.tile(tf.expand_dims(user_activity, 0), [current_batch_size, 1])
                x_0 = tf.cast(tf.random.uniform(tf.shape(x_1)) < activity_probs, tf.float32)
                
                loss = train_step(model, optimizer, x_1, cond, t, x_0)
                
                loss_val = loss.numpy()
                train_loss += loss_val
                train_steps += 1
                
                progress.update(epoch_task, advance=1, info=f"Loss: {loss_val:.4f}")
            
            avg_train_loss = train_loss / train_steps
            
            # -------- [TensorBoard] Epoch별 Loss 기록 --------
            with summary_writer.as_default():
                tf.summary.scalar('Loss/train', avg_train_loss, step=epoch)
            # ----------------------------------------------------
            
            # 2. Validation
            if (epoch + 1) % eval_step == 0:
                progress.update(epoch_task, description=f"[bold yellow]Validating (s={inference_steps})...", info="")
                
                results = {'Recall@20': [], 'NDCG@20': [], 'Precision@20': [], 'Hit@20': []}
                
                for x_1, cond in vali_ds:
                    batch_bs = tf.shape(x_1)[0]
                    curr_x = flow.get_prior_sample(batch_bs)
                    dt = 1.0 / inference_steps
                    
                    for i in range(inference_steps):
                        t_val = i * dt
                        t = tf.fill([batch_bs, 1], t_val)
                        t = tf.cast(t, tf.float32)
                        pred = model(curr_x, cond, t, training=False)
                        curr_x = flow.inference_step(curr_x, pred, t_val, dt)
                    
                    preds = curr_x.numpy()
                    targets = x_1.numpy()
                    
                    for i in range(batch_bs):
                        gt_users = np.where(targets[i] > 0.5)[0]
                        if len(gt_users) == 0: continue
                        top_indices = np.argsort(preds[i])[-20:][::-1]
                        
                        m = compute_metrics(top_indices, gt_users, k_list=[20])
                        results['Recall@20'].append(m['Recall@20'])
                        results['NDCG@20'].append(m['NDCG@20'])
                        results['Precision@20'].append(m['Precision@20'])
                        results['Hit@20'].append(m['Hit@20'])

                def safe_mean(l): return np.mean(l) if l else 0.0
                r20_avg = safe_mean(results['Recall@20'])
                n20_avg = safe_mean(results['NDCG@20'])
                p20_avg = safe_mean(results['Precision@20'])
                h20_avg = safe_mean(results['Hit@20'])

                # -------- [TensorBoard] Validation 지표 기록 --------
                with summary_writer.as_default():
                    tf.summary.scalar('Metrics/Recall@20', r20_avg, step=epoch)
                    tf.summary.scalar('Metrics/NDCG@20', n20_avg, step=epoch)
                    tf.summary.scalar('Metrics/Precision@20', p20_avg, step=epoch)
                    tf.summary.scalar('Metrics/Hit@20', h20_avg, step=epoch)
                # --------------------------------------------------------
                
                log_msg = (
                    f"[bold white]Epoch {epoch+1:03d}[/] | "
                    f"Loss: [red]{avg_train_loss:.4f}[/] | "
                    f"R@20: [cyan]{r20_avg:.4f}[/] | "
                    f"N@20: [blue]{n20_avg:.4f}[/] | "
                    f"P@20: [magenta]{p20_avg:.4f}[/] | "
                    f"H@20: [green]{h20_avg:.4f}[/]"
                )

                if r20_avg > best_recall:
                    best_recall = r20_avg
                    patience_cnt = 0
                    model.save_weights("saved_model/best_flow_model")
                    log_msg += " | [bold green]★ Best[/]"
                else:
                    patience_cnt += 1
                    if patience_cnt >= config.get('patience', 10):
                        console.print(log_msg + " | [bold red][Early Stop][/]")
                        break
                
                console.print(log_msg)

            else:
                console.print(f"[dim]Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} (Vali Skip)[/dim]")

            progress.update(overall_task, advance=1)

    console.print()
    console.print(Panel(
        f"Training Finished.\nBest Recall@20: [bold green]{best_recall:.4f}[/bold green]\nRun 'tensorboard --logdir logs/gradient_tape' to view.",
        title="Training Complete",
        border_style="green"
    ))

if __name__ == "__main__":
    if not os.path.exists("saved_model"):
        os.makedirs("saved_model")
    train()