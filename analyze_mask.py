#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析和可视化 precomputed_mask.pt 文件

这个脚本用于详细分析 SegAnyGaussians 项目中的分割掩码文件
"""

import torch
import os
import numpy as np
from plyfile import PlyData

def analyze_mask(mask_path, point_cloud_path=None):
    """分析 mask 文件的详细信息"""
    
    print("=" * 80)
    print(f"分析文件: {mask_path}")
    print("=" * 80)
    
    # 加载 mask
    if not os.path.exists(mask_path):
        print(f"错误: 文件不存在 - {mask_path}")
        return
    
    mask = torch.load(mask_path)
    
    # 基本信息
    print("\n【1. 基本信息】")
    print(f"  数据类型: {type(mask)}")
    print(f"  Tensor 形状: {mask.shape}")
    print(f"  Tensor dtype: {mask.dtype}")
    print(f"  设备: {mask.device}")
    print(f"  总元素数: {mask.numel():,}")
    print(f"  内存占用: {mask.element_size() * mask.numel() / 1024:.2f} KB")
    
    # 统计信息
    print("\n【2. 数值统计】")
    if mask.dtype == torch.bool:
        true_count = torch.sum(mask).item()
        false_count = mask.numel() - true_count
        print(f"  True 数量: {true_count:,} ({100.0 * true_count / mask.numel():.2f}%)")
        print(f"  False 数量: {false_count:,} ({100.0 * false_count / mask.numel():.2f}%)")
        print(f"  唯一值: {torch.unique(mask).tolist()}")
    else:
        mask_float = mask.float()
        print(f"  最小值: {mask_float.min().item():.6f}")
        print(f"  最大值: {mask_float.max().item():.6f}")
        print(f"  平均值: {mask_float.mean().item():.6f}")
        print(f"  标准差: {mask_float.std().item():.6f}")
        unique_count = len(torch.unique(mask))
        print(f"  唯一值数量: {unique_count}")
        if unique_count <= 20:
            print(f"  唯一值列表: {torch.unique(mask).tolist()}")
    
    # 与点云的关系
    if point_cloud_path and os.path.exists(point_cloud_path):
        print("\n【3. 与3D点云的对应关系】")
        plydata = PlyData.read(point_cloud_path)
        vertices = plydata['vertex']
        num_points = len(vertices)
        
        print(f"  点云文件: {point_cloud_path}")
        print(f"  点云数量: {num_points:,}")
        print(f"  Mask 元素数: {mask.shape[0]:,}")
        
        if mask.shape[0] == num_points:
            print(f"  ✓ Mask 与点云完美匹配！")
            print(f"  → 每个 mask 元素对应一个 3D Gaussian 点")
        else:
            print(f"  ✗ Mask 与点云不匹配！")
            print(f"  → 差异: {abs(mask.shape[0] - num_points):,} 个元素")
    
    # 物理含义解释
    print("\n【4. 物理含义】")
    print("  这是一个 3D Gaussian Splatting 分割掩码:")
    print("  • mask[i] 表示第 i 个 3D Gaussian 点的选择状态")
    print("  • True/1.0  → 该点被选中，属于目标分割区域")
    print("  • False/0.0 → 该点未被选中，不属于目标分割区域")
    
    # 使用示例
    print("\n【5. 在代码中的使用】")
    print("  位置: gaussian_renderer/__init__.py::render_mask()")
    print("  ")
    print("  处理流程:")
    print("  1. 加载: mask = torch.load('precomputed_mask.pt')")
    print("  2. 转换: mask = mask.float()  # bool → float")
    print("  3. 扩展: mask = mask.unsqueeze(-1).repeat([1,3])  # [N] → [N,3]")
    print("  4. 渲染: 作为 colors_precomp 传入 GaussianRasterizer")
    print("  5. 输出: 生成 2D 分割图像")
    
    # 分割语义
    print("\n【6. 分割语义】")
    if mask.dtype == torch.bool:
        true_ratio = torch.sum(mask).item() / mask.numel()
        if true_ratio == 1.0:
            print("  当前状态: 全选 (100% True)")
            print("  可能含义:")
            print("    • 初始化的默认 mask")
            print("    • 整个场景作为单一对象")
            print("    • 用于渲染完整场景")
        elif true_ratio == 0.0:
            print("  当前状态: 全不选 (100% False)")
            print("  可能含义:")
            print("    • 空分割结果")
            print("    • 需要重新运行分割算法")
        else:
            print(f"  当前状态: 部分选择 ({true_ratio*100:.2f}% True)")
            print("  可能含义:")
            print("    • 已识别出特定对象")
            print(f"    • 选中了 {torch.sum(mask).item():,} 个点")
            print(f"    • 未选中 {(~mask).sum().item():,} 个点")
    
    # 生成方式
    print("\n【7. 生成方式】")
    print("  方法1: 交互式GUI分割")
    print("    命令: python saga_gui.py")
    print("    说明: 通过GUI界面进行交互式分割，点击'save'保存mask")
    print("  ")
    print("  方法2: 自动分割脚本")
    print("    命令: python extract_segment_everything_masks.py")
    print("    说明: 使用SAM等模型自动生成分割mask")
    
    print("\n" + "=" * 80)


def compare_masks(mask_paths):
    """比较多个 mask 文件的差异"""
    
    print("\n" + "=" * 80)
    print("多个 Mask 文件对比")
    print("=" * 80)
    
    masks = {}
    for path in mask_paths:
        if os.path.exists(path):
            masks[path] = torch.load(path)
    
    if len(masks) == 0:
        print("没有找到任何 mask 文件")
        return
    
    print(f"\n找到 {len(masks)} 个 mask 文件:\n")
    
    # 打印对比表格
    print(f"{'文件名':<35} {'形状':<15} {'类型':<10} {'True%':<10} {'说明'}")
    print("-" * 95)
    
    for path, mask in masks.items():
        filename = os.path.basename(path)
        shape_str = str(mask.shape)
        dtype_str = str(mask.dtype).replace('torch.', '')
        
        if mask.dtype == torch.bool:
            true_ratio = f"{100.0 * torch.sum(mask).item() / mask.numel():.1f}%"
            description = "全选" if torch.all(mask) else ("全不选" if not torch.any(mask) else "部分选择")
        else:
            true_ratio = "N/A"
            description = f"值域:[{mask.min():.2f},{mask.max():.2f}]"
        
        print(f"{filename:<35} {shape_str:<15} {dtype_str:<10} {true_ratio:<10} {description}")
    
    # 检查是否完全相同
    if len(masks) > 1:
        print("\n差异分析:")
        mask_list = list(masks.values())
        all_same = True
        for i in range(1, len(mask_list)):
            if not torch.equal(mask_list[0], mask_list[i]):
                all_same = False
                print(f"  • {list(masks.keys())[0]} 与 {list(masks.keys())[i]} 不同")
                diff_count = torch.sum(mask_list[0] != mask_list[i]).item()
                print(f"    差异点数: {diff_count:,}")
        
        if all_same:
            print("  ✓ 所有 mask 文件内容完全相同")
    
    print("=" * 80)


if __name__ == "__main__":
    # 分析 precomputed_mask.pt
    mask_path = "./segmentation_res/precomputed_mask.pt"
    point_cloud_path = "./output/513864c4-1/point_cloud/iteration_30000/scene_point_cloud.ply"
    
    analyze_mask(mask_path, point_cloud_path)
    
    # 对比所有 mask 文件
    mask_files = [
        "./segmentation_res/precomputed_mask.pt",
        "./segmentation_res/precomputed_mask_2d.pt",
        "./segmentation_res/precomputed_mask233.pt"
    ]
    
    compare_masks(mask_files)
    
    print("\n提示:")
    print("  • 使用 saga_gui.py 可以交互式编辑和保存新的 mask")
    print("  • 使用 render.py --precomputed_mask <mask.pt> 可以渲染分割结果")
    print("  • Mask 文件可以用于场景编辑、对象提取、选择性渲染等应用")
