import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def read_json_lines(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")
    return records

def average_per_epoch(records, key):
    epoch_to_values = defaultdict(list)
    for r in records:
        if key in r and 'epoch' in r:
            epoch_to_values[r['epoch']].append(r[key])
    epochs = sorted(epoch_to_values.keys())
    avg_values = [sum(epoch_to_values[e]) / len(epoch_to_values[e]) for e in epochs]
    return epochs, avg_values

def smooth_curve(values, window=3):
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i+1]
        smoothed.append(sum(window_vals) / len(window_vals))
    return smoothed

def plot_curve(x, y, title, ylabel, save_path):
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel('Epoch' if all(isinstance(i,int) for i in x) else 'Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()
    
def extract_metric(records, key):
    steps = []
    values = []
    for r in records:
        if key in r:
            steps.append(r.get('step', len(steps)))
            values.append(r[key])
    return steps, values

def process_log(json_file, prefix, output_dir):
    print(f"Processing {json_file} ...")
    records = read_json_lines(json_file)
    if not records:
        print(f"No valid records found in {json_file}")
        return
    
    # 训练 loss 曲线（按 epoch 平均+平滑）
    epochs, avg_loss = average_per_epoch(records, 'loss')
    if avg_loss:
        smooth_loss = smooth_curve(avg_loss, window=3)
        plot_curve(
            epochs, smooth_loss,
            f'Training Loss Curve (Smoothed) - {prefix}',
            'Loss',
            os.path.join(output_dir, f'{prefix}_training_loss.png')
        )
    else:
        print(f"No 'loss' key found in {json_file}")

    # 验证 mAP 曲线（step vs mAP）
    map_steps, map_values = extract_metric(records, 'pascal_voc/mAP')
    if map_values:
        plot_curve(
            map_steps, map_values,
            f'Validation mAP Curve - {prefix}',
            'mAP',
            os.path.join(output_dir, f'{prefix}_validation_map.png')
        )
    else:
        print(f"No 'pascal_voc/mAP' key found in {json_file}")

def main():
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)

    logs = [
        ("/root/autodl-tmp/mmdetection-main/work_dirs/sparse_rcnn_voc_1x/20250528_121748/vis_data/scalars.json", "sparse_rcnn_voc"),
        ("/root/autodl-tmp/mmdetection-main/work_dirs/mask_rcnn_voc_1x/20250527_160710/vis_data/20250527_160710.json", "mask_rcnn_voc"),
    ]

    for json_file, prefix in logs:
        process_log(json_file, prefix, output_dir)

if __name__ == "__main__":
    main()
