"""
3D Reconstruction Quality Evaluation Script
Measures PSNR, SSIM, and LPIPS metrics for rendered images vs ground truth
"""
import torch
import torch.nn.functional as F
from scene import Scene, GaussianModel, FeatureGaussianModel
import os
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS
import json
from pathlib import Path
import numpy as np
from utils.image_utils import psnr as psnr_fn
class MetricsEvaluator:
    """Evaluator for 3D reconstruction metrics"""
    def __init__(self, device='cuda'):
        self.device = device
        # Initialize LPIPS model
        self.lpips_fn = LPIPS(net_type='alex').to(device)
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    def calculate_psnr(self, img1, img2):
        """
        Calculate PSNR between two images
        Args:
            img1, img2: torch.Tensor, shape [C, H, W], range [0, 1]
        Returns:
            float: PSNR value in dB
        """
        # Use existing psnr function, which expects shape [B, C, H, W]
        return psnr_fn(img1.unsqueeze(0), img2.unsqueeze(0))[0]
    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images
        Args:
            img1, img2: torch.Tensor, shape [C, H, W], range [0, 1]
        Returns:
            float: SSIM value
        """
        return ssim(img1.unsqueeze(0), img2.unsqueeze(0))
    def calculate_lpips(self, img1, img2):
        """
        Calculate LPIPS between two images
        Args:
            img1, img2: torch.Tensor, shape [C, H, W], range [0, 1]
        Returns:
            float: LPIPS value
        """
        # LPIPS expects shape [B, C, H, W]
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        # LPIPS expects range [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1, img2)
        return lpips_value.item()
    def evaluate_all(self, img1, img2):
        """
        Calculate all metrics (PSNR, SSIM, LPIPS)
        Args:
            img1, img2: torch.Tensor, shape [C, H, W], range [0, 1]
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {}
        metrics['psnr'] = self.calculate_psnr(img1, img2).item()
        metrics['ssim'] = self.calculate_ssim(img1, img2).item()
        metrics['lpips'] = self.calculate_lpips(img1, img2)
        return metrics
def evaluate_reconstruction(model_path, iteration, dataset, pipeline, evaluator):
    """
    Evaluate reconstruction quality on test set
    Args:
        model_path: Path to the model
        iteration: Iteration number to evaluate
        dataset: Dataset parameters
        pipeline: Pipeline parameters
        evaluator: MetricsEvaluator instance
    Returns:
        dict: Average metrics across all test images
    """
    # Load Gaussian model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, mode='eval', target='scene')
    # Set background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # Get test cameras
    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("Warning: No test cameras found, using training cameras instead")
        test_cameras = scene.getTrainCameras()
    # Storage for metrics
    all_metrics = {
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    per_image_metrics = []
    print(f"\nEvaluating {len(test_cameras)} images...")
    # Evaluate each test view
    with torch.no_grad():
        for idx, viewpoint in enumerate(tqdm(test_cameras, desc="Evaluating views")):
            # Render image
            render_pkg = render(viewpoint, gaussians, pipeline, background)
            rendered_image = render_pkg["render"]
            # Get ground truth
            gt_image = viewpoint.original_image[0:3, :, :].cuda()
            # Calculate metrics
            metrics = evaluator.evaluate_all(rendered_image, gt_image)
            # Store metrics
            all_metrics['psnr'].append(metrics['psnr'])
            all_metrics['ssim'].append(metrics['ssim'])
            all_metrics['lpips'].append(metrics['lpips'])
            per_image_metrics.append({
                'image_idx': idx,
                'image_name': viewpoint.image_name if hasattr(viewpoint, 'image_name') else f"image_{idx:05d}",
                **metrics
            })
    # Calculate average metrics
    avg_metrics = {
        'psnr_avg': np.mean(all_metrics['psnr']),
        'psnr_std': np.std(all_metrics['psnr']),
        'ssim_avg': np.mean(all_metrics['ssim']),
        'ssim_std': np.std(all_metrics['ssim']),
        'lpips_avg': np.mean(all_metrics['lpips']),
        'lpips_std': np.std(all_metrics['lpips']),
        'num_images': len(test_cameras)
    }
    return avg_metrics, per_image_metrics
def main():
    # Parse arguments
    parser = ArgumentParser(description="Evaluation script for 3D reconstruction metrics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to evaluate")
    parser.add_argument("--skip_train", action="store_true", help="Skip training set evaluation")
    parser.add_argument("--skip_test", action="store_true", help="Skip test set evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save results as JSON")
    parser.add_argument('--target', default='scene', const='scene', nargs='?',
                       choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature', 'xyz'],
                       help="Target for evaluation (default: scene)")
    args = get_combined_args(parser)

    # Ensure output_json attribute exists (get_combined_args may not include it if None)
    if not hasattr(args, 'output_json'):
        args.output_json = None

    print("=" * 70)
    print("3D Reconstruction Quality Evaluation")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print("=" * 70)
    # Initialize evaluator
    evaluator = MetricsEvaluator(device='cuda')
    # Evaluate
    avg_metrics, per_image_metrics = evaluate_reconstruction(
        args.model_path,
        args.iteration,
        model.extract(args),
        pipeline.extract(args),
        evaluator
    )
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Number of images: {avg_metrics['num_images']}")
    print("-" * 70)
    print(f"PSNR:  {avg_metrics['psnr_avg']:.4f} ± {avg_metrics['psnr_std']:.4f} dB")
    print(f"SSIM:  {avg_metrics['ssim_avg']:.4f} ± {avg_metrics['ssim_std']:.4f}")
    print(f"LPIPS: {avg_metrics['lpips_avg']:.4f} ± {avg_metrics['lpips_std']:.4f}")
    print("=" * 70)
    # Save results to JSON if specified
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            'model_path': args.model_path,
            'iteration': args.iteration,
            'average_metrics': avg_metrics,
            'per_image_metrics': per_image_metrics
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    # Save to default location in model folder
    default_output_dir = Path(args.model_path) / "evaluation_results"
    default_output_dir.mkdir(parents=True, exist_ok=True)
    iteration_str = str(args.iteration) if args.iteration != -1 else "latest"
    default_output_path = default_output_dir / f"metrics_iter_{iteration_str}.json"
    results = {
        'model_path': args.model_path,
        'iteration': args.iteration,
        'average_metrics': avg_metrics,
        'per_image_metrics': per_image_metrics
    }
    with open(default_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results also saved to: {default_output_path}")
    return avg_metrics
if __name__ == "__main__":
    main()

