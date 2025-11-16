# 3D重建质量评估指南
## 简介
本脚本用于评估3D重建质量，计算以下三个关键指标：
- **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，衡量图像质量，单位为dB，值越高越好
- **SSIM (Structural Similarity Index)**: 结构相似性指数，范围0-1，值越高越好
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 感知相似度，值越低越好
## 使用方法
### 基本用法
```bash
# 评估指定模型路径和迭代次数
python evaluate_metrics.py -m ./output/figurines --iteration 30000
# 使用最新的checkpoint（自动查找）
python evaluate_metrics.py -m ./output/figurines --iteration -1
# 指定数据源路径
python evaluate_metrics.py -m ./output/figurines -s ./data/figurines --iteration 30000
```
### 高级选项
```bash
# 保存结果到指定JSON文件
python evaluate_metrics.py -m ./output/figurines --iteration 30000 \
    --output_json ./results/figurines_eval.json
# 指定白色背景
python evaluate_metrics.py -m ./output/figurines --iteration 30000 \
    --white_background
```
### 完整参数列表
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m`, `--model_path` | 模型输出路径 | 必需 |
| `-s`, `--source_path` | 数据源路径 | 从model_path推断 |
| `--iteration` | 评估的迭代次数（-1表示最新） | -1 |
| `--output_json` | 保存结果的JSON文件路径 | None |
| `--white_background` | 使用白色背景 | False |
| `--quiet` | 静默模式 | False |
## 输出说明
### 控制台输出
脚本会在控制台显示：
```
======================================================================
3D Reconstruction Quality Evaluation
======================================================================
Model path: ./output/figurines
Iteration: 30000
======================================================================
Evaluating 100 images...
100%|██████████| 100/100 [00:30<00:00,  3.33it/s]
======================================================================
EVALUATION RESULTS
======================================================================
Number of images: 100
----------------------------------------------------------------------
PSNR:  28.5432 ± 2.1234 dB
SSIM:  0.9234 ± 0.0123
LPIPS: 0.0456 ± 0.0089
======================================================================
```
### JSON输出文件
结果会自动保存到两个位置：
1. **模型目录下**：`{model_path}/evaluation_results/metrics_iter_{iteration}.json`
2. **自定义路径**（如果指定了`--output_json`）
JSON文件格式：
```json
{
  "model_path": "./output/figurines",
  "iteration": 30000,
  "average_metrics": {
    "psnr_avg": 28.5432,
    "psnr_std": 2.1234,
    "ssim_avg": 0.9234,
    "ssim_std": 0.0123,
    "lpips_avg": 0.0456,
    "lpips_std": 0.0089,
    "num_images": 100
  },
  "per_image_metrics": [
    {
      "image_idx": 0,
      "image_name": "image_00000",
      "psnr": 29.1234,
      "ssim": 0.9345,
      "lpips": 0.0423
    },
    ...
  ]
}
```
---
## 📊 指标解释
### PSNR (峰值信噪比)
- **单位**: dB（分贝）
- **范围**: 通常 20-40 dB
- **评价标准**:
  - ✅ **> 30 dB**: 优秀
  - ⚠️ **25-30 dB**: 良好  
  - ❌ **< 25 dB**: 需要改进
### SSIM (结构相似性指数)
- **范围**: 0-1
- **评价标准**:
  - ✅ **> 0.95**: 优秀
  - ⚠️ **0.90-0.95**: 良好
  - ❌ **< 0.90**: 需要改进
### LPIPS (感知相似度)
- **范围**: 0-1（越小越好）
- **评价标准**:
  - ✅ **< 0.05**: 优秀
  - ⚠️ **0.05-0.10**: 良好
  - ❌ **> 0.10**: 需要改进
---
## 批量评估示例
```bash
#!/bin/bash
# 批量评估多个场景
SCENES=("figurines" "robot" "desk")
ITERATIONS=(10000 20000 30000)
for scene in "${SCENES[@]}"; do
    for iter in "${ITERATIONS[@]}"; do
        echo "Evaluating $scene at iteration $iter"
        python evaluate_metrics.py \
            -m ./output/$scene \
            --iteration $iter \
            --output_json ./results/${scene}_iter${iter}.json
    done
done
```
## 故障排除
### 问题：找不到测试相机
**解决方案**：脚本会自动使用训练相机进行评估
### 问题：CUDA内存不足
**解决方案**：减少测试图像数量，或在评估时使用较小的图像分辨率
### 问题：LPIPS模型下载失败
**解决方案**：手动下载LPIPS权重文件到`lpipsPyTorch/weights/`目录
## 与其他工具集成
### 可视化结果
```python
import json
import matplotlib.pyplot as plt
# 读取评估结果
with open('results/figurines_eval.json', 'r') as f:
    data = json.load(f)
# 绘制每张图片的PSNR
psnr_values = [img['psnr'] for img in data['per_image_metrics']]
plt.plot(psnr_values)
plt.xlabel('Image Index')
plt.ylabel('PSNR (dB)')
plt.title('Per-Image PSNR')
plt.savefig('psnr_plot.png')
```
### 比较多个模型
```python
import json
import pandas as pd
# 读取多个模型的结果
models = ['model_a', 'model_b', 'model_c']
results = []
for model in models:
    with open(f'results/{model}_eval.json', 'r') as f:
        data = json.load(f)
        results.append({
            'Model': model,
            'PSNR': data['average_metrics']['psnr_avg'],
            'SSIM': data['average_metrics']['ssim_avg'],
            'LPIPS': data['average_metrics']['lpips_avg']
        })
df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```
## 参考资料
- PSNR: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
- SSIM: https://en.wikipedia.org/wiki/Structural_similarity
- LPIPS: https://arxiv.org/abs/1801.03924
