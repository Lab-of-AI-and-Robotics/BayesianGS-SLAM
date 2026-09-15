""" This module includes the Logger class, which is responsible for logging for both Mapper and the Tracker """
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


class Logger(object):

    def __init__(self, output_path: Union[Path, str], use_wandb=False, timing_log_path=None) -> None:
        self.output_path = Path(output_path)
        (self.output_path / "mapping_vis").mkdir(exist_ok=True, parents=True)
        self.use_wandb = use_wandb
        self.timing_log_path = timing_log_path  # Path to main timing log file
    
    def log_and_print(self, message, end='\n'):
        """
        Write message to both main timing log file (if available) and print to console.
        
        Args:
            message: Message to log
            end: End character for the message (default: newline)
        """
        if self.timing_log_path:
            # Append to main timing log file
            try:
                with open(self.timing_log_path, 'a') as f:
                    f.write(message + end)
            except Exception as e:
                # If file writing fails, at least print
                pass
        print(message, end=end)

    def log_tracking_iteration(self, frame_id, cur_pose, gt_quat, gt_trans, total_loss,
                               color_loss, depth_loss, iter, num_iters,
                               wandb_output=False, print_output=False) -> None:
        """ Logs tracking iteration metrics including pose error, losses, and optionally reports to Weights & Biases.
        Logs the error between the current pose estimate and ground truth quaternion and translation,
        as well as various loss metrics. Can output to wandb if enabled and specified, and print to console.
        Args:
            frame_id: Identifier for the current frame.
            cur_pose: The current estimated pose as a tensor (quaternion + translation).
            gt_quat: Ground truth quaternion.
            gt_trans: Ground truth translation.
            total_loss: Total computed loss for the current iteration.
            color_loss: Computed color loss for the current iteration.
            depth_loss: Computed depth loss for the current iteration.
            iter: The current iteration number.
            num_iters: The total number of iterations planned.
            wandb_output: Whether to output the log to wandb.
            print_output: Whether to print the log output.
        """

        quad_err = torch.abs(cur_pose[:4] - gt_quat).mean().item()
        trans_err = torch.abs(cur_pose[4:] - gt_trans).mean().item()
        if self.use_wandb and wandb_output:
            wandb.log(
                {
                    "Tracking/idx": frame_id,
                    "Tracking/cam_quad_err": quad_err,
                    "Tracking/cam_position_err": trans_err,
                    "Tracking/total_loss": total_loss.item(),
                    "Tracking/color_loss": color_loss.item(),
                    "Tracking/depth_loss": depth_loss.item(),
                    "Tracking/num_iters": num_iters,
                })
        if iter == num_iters - 1:
            msg = f"frame_id: {frame_id}, cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f} "
        else:
            msg = f"iter: {iter}, color_loss: {color_loss.item():.5f}, depth_loss: {depth_loss.item():.5f} "
        msg = msg + f", cam_quad_err: {quad_err:.5f}, cam_trans_err: {trans_err:.5f}"
        if print_output:
            self.log_and_print(msg)

    def log_mapping_iteration(self, frame_id, new_pts_num, model_size, iter_opt_time, opt_dict: dict) -> None:
        """ Logs mapping iteration metrics including the number of new points, model size, and optimization times,
        and optionally reports to Weights & Biases (wandb).
        Args:
            frame_id: Identifier for the current frame.
            new_pts_num: The number of new points added in the current mapping iteration.
            model_size: The total size of the model after the current mapping iteration.
            iter_opt_time: Time taken per optimization iteration.
            opt_dict: A dictionary containing optimization metrics such as PSNR, color loss, and depth loss.
        """
        if self.use_wandb:
            wandb.log({"Mapping/idx": frame_id,
                       "Mapping/num_total_gs": model_size,
                       "Mapping/num_new_gs": new_pts_num,
                       "Mapping/per_iteration_time": iter_opt_time,
                       "Mapping/psnr_render": opt_dict["psnr_render"],
                       "Mapping/color_loss": opt_dict[frame_id]["color_loss"],
                       "Mapping/depth_loss": opt_dict[frame_id]["depth_loss"]})

    def vis_mapping_iteration(self, frame_id, iter, color, depth, gt_color, gt_depth, seeding_mask=None, interval=10) -> None:
        """
        Visualization of depth, color images and save to file.

        Args:
            frame_id (int): current frame index.
            iter (int): the iteration number.
            save_rendered_image (bool): whether to save the rgb image in separate folder
            img_dir (str): the directory to save the visualization.
            seeding_mask: used in mapper when adding gaussians, if not none.
        """
        if frame_id % interval != 0:
            return
        gt_depth_np = gt_depth.cpu().numpy()
        gt_color_np = gt_color.cpu().numpy()

        depth_np = depth.detach().cpu().numpy()
        color = torch.round(color * 255.0) / 255.0
        color_np = color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        # make errors >=5cm noticeable
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[np.squeeze(gt_depth_np == 0.0)] = 0.0

        # Determine Aspect Ratio and Figure Size
        aspect_ratio = color.shape[1] / color.shape[0]
        fig_height = 8
        # Adjust the multiplier as needed for better spacing
        fig_width = fig_height * aspect_ratio * 1.2

        fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 0].set_title('Input Depth', fontsize=16)
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
        axs[0, 1].set_title('Rendered Depth', fontsize=16)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma")
        axs[0, 2].set_title('Depth Residual', fontsize=16)
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB', fontsize=16)
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Rendered RGB', fontsize=16)
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        if seeding_mask is not None:
            axs[1, 2].imshow(seeding_mask, cmap="gray")
            axs[1, 2].set_title('Densification Mask', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])
        else:
            axs[1, 2].imshow(color_residual, cmap="plasma")
            axs[1, 2].set_title('RGB Residual', fontsize=16)
            axs[1, 2].set_xticks([])
            axs[1, 2].set_yticks([])

        for ax in axs.flatten():
            ax.axis('off')
        fig.tight_layout()
        plt.subplots_adjust(top=0.90)  # Adjust top margin
        fig_name = str(self.output_path / "mapping_vis" / f'{frame_id:04d}_{iter:04d}.jpg')
        fig_title = f"Mapper Color/Depth at frame {frame_id:04d} iters {iter:04d}"
        plt.suptitle(fig_title, y=0.98, fontsize=20)
        plt.savefig(fig_name, dpi=250, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            log_title = "Mapping_vis/" + f'{frame_id:04d}_{iter:04d}'
            wandb.log({log_title: [wandb.Image(fig_name)]})
        self.log_and_print(f"Saved rendering vis of color/depth at {frame_id:04d}_{iter:04d}.jpg")

    def vis_tracking_iteration(self, frame_id: int, color: torch.Tensor, depth: torch.Tensor,
                               gt_color: torch.Tensor, gt_depth: torch.Tensor,
                               color_var: torch.Tensor = None, depth_var: torch.Tensor = None,
                               tracking_mask: torch.Tensor = None, interval: int = 10) -> None:
        """Visualization of tracking result (color/depth/residual + optional uncertainty) and save to file.

        Args:
            frame_id: current frame index.
            color: rendered color [3, H, W].
            depth: rendered depth [1, H, W] or [H, W].
            gt_color: ground truth color [3, H, W].
            gt_depth: ground truth depth [H, W].
            color_var: total color uncertainty [3, H, W], optional.
            depth_var: total depth uncertainty [1, H, W], optional.
            interval: save every N frames.
        """
        if frame_id % interval != 0:
            return

        (self.output_path / "tracking_vis").mkdir(exist_ok=True, parents=True)

        # [3,H,W] → [H,W,3]
        gt_color_np  = np.clip(gt_color.cpu().numpy().transpose(1, 2, 0), 0, 1)
        color_np     = np.clip(torch.round(color * 255.0).detach().cpu().numpy().transpose(1, 2, 0) / 255.0, 0, 1)
        gt_depth_np  = gt_depth.squeeze().cpu().numpy()
        depth_np     = depth.squeeze().detach().cpu().numpy()

        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0

        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        depth_residual = np.clip(depth_residual, 0.0, 0.05)

        has_var  = color_var is not None and depth_var is not None
        has_mask = tracking_mask is not None
        n_rows   = 3 if has_var else 2
        aspect_ratio = gt_color_np.shape[1] / gt_color_np.shape[0]
        fig, axs = plt.subplots(n_rows, 3, figsize=(fig_height := 8, fig_height) if False
                                else (8 * aspect_ratio * 1.2, 4 * n_rows))

        titles = [['GT Color', 'Rendered Color', 'Color Residual'],
                  ['GT Depth', 'Rendered Depth', 'Depth Residual']]
        data   = [[gt_color_np,  color_np,    color_residual],
                  [gt_depth_np,  depth_np,    depth_residual]]
        cmaps  = [['plasma', 'plasma', 'plasma'],
                  ['jet',    'jet',    'plasma']]
        vranges = [[None, None, None],
                   [(0, 6), (0, 6), None]]

        for r in range(2):
            for c in range(3):
                kwargs = {'cmap': cmaps[r][c]}
                if vranges[r][c]:
                    kwargs['vmin'], kwargs['vmax'] = vranges[r][c]
                axs[r, c].imshow(data[r][c], **kwargs)
                axs[r, c].set_title(titles[r][c], fontsize=14)
                axs[r, c].axis('off')

        if has_var:
            color_var_np = color_var.mean(dim=0).detach().cpu().numpy()   # [H,W]
            depth_var_np = depth_var.squeeze().detach().cpu().numpy()     # [H,W]
            axs[2, 0].imshow(np.log10(color_var_np + 1e-8), cmap='turbo')
            axs[2, 0].set_title('Color Uncertainty (log10)', fontsize=14)
            axs[2, 0].axis('off')
            axs[2, 1].imshow(np.log10(depth_var_np + 1e-8), cmap='turbo')
            axs[2, 1].set_title('Depth Uncertainty (log10)', fontsize=14)
            axs[2, 1].axis('off')
            if has_mask:
                mask_np = tracking_mask.squeeze().cpu().numpy().astype(np.float32)
                axs[2, 2].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
                axs[2, 2].set_title('Tracking Mask', fontsize=14)
            axs[2, 2].axis('off')

        plt.suptitle(f"Tracker Color/Depth at frame {frame_id:04d}", y=0.98, fontsize=18)
        plt.tight_layout()
        fig_name = str(self.output_path / "tracking_vis" / f'{frame_id:04d}.jpg')
        plt.savefig(fig_name, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()
        if self.use_wandb:
            wandb.log({f"Tracking_vis/{frame_id:04d}": [wandb.Image(fig_name)]})

    def vis_lc_registration(self, src_id: int, tgt_id: int,
                             renders: list) -> None:
        """Save rasterizer renders and loss curves for a loop closure registration attempt.

        Args:
            src_id: source submap id.
            tgt_id: target submap id.
            renders: list of dicts with keys 'image','depth','gt_image','gt_depth','color_var','depth_var'.
            loss_logs: list of loss-per-iteration lists, one per viewpoint.
        """
        lc_vis_dir = self.output_path / "loop_vis"
        lc_vis_dir.mkdir(exist_ok=True, parents=True)

        prefix = f"{src_id:05d}_sub{src_id}_to_{tgt_id}"

        # --- per-viewpoint render comparisons ---
        for vi, r in enumerate(renders):
            img   = r.get("image")    # [3,H,W]
            depth = r.get("depth")    # [1,H,W]
            gt_img   = r.get("gt_image")  # [3,H,W] or None
            gt_depth = r.get("gt_depth")  # [H,W] or None
            cvar  = r.get("color_var")    # [3,H,W] or None
            dvar  = r.get("depth_var")    # [1,H,W] or None

            if img is None:
                continue

            img_np   = np.clip(img.cpu().numpy().transpose(1, 2, 0), 0, 1)
            depth_np = depth.squeeze().cpu().numpy()

            has_gt    = gt_img is not None and gt_depth is not None
            has_var   = cvar is not None and dvar is not None
            n_rows    = 1 + int(has_gt) + int(has_var)

            aspect = img_np.shape[1] / img_np.shape[0]
            fig, axs = plt.subplots(n_rows, 3, figsize=(8 * aspect * 1.2, 4 * n_rows))
            if n_rows == 1:
                axs = axs[np.newaxis, :]

            # row 0: rendered color / rendered depth / (residual or blank)
            axs[0, 0].imshow(img_np, cmap="plasma")
            axs[0, 0].set_title("Rendered Color", fontsize=12)
            axs[0, 1].imshow(depth_np, cmap="jet", vmin=0, vmax=6)
            axs[0, 1].set_title("Rendered Depth", fontsize=12)
            axs[0, 2].axis("off")

            row = 1
            if has_gt:
                gt_img_np   = np.clip(gt_img.cpu().numpy().transpose(1, 2, 0), 0, 1)
                gt_depth_np = gt_depth.squeeze().cpu().numpy()
                color_res   = np.clip(np.abs(gt_img_np - img_np), 0, 1)
                depth_res   = np.clip(np.abs(gt_depth_np - depth_np), 0, 0.05)
                depth_res[gt_depth_np == 0] = 0

                axs[row, 0].imshow(gt_img_np, cmap="plasma")
                axs[row, 0].set_title("GT Color", fontsize=12)
                axs[row, 1].imshow(gt_depth_np, cmap="jet", vmin=0, vmax=6)
                axs[row, 1].set_title("GT Depth", fontsize=12)
                axs[row, 2].imshow(color_res, cmap="plasma")
                axs[row, 2].set_title("Color Residual", fontsize=12)
                row += 1

            if has_var:
                cvar_np = cvar.mean(dim=0).cpu().numpy()
                dvar_np = dvar.squeeze().cpu().numpy()
                axs[row, 0].imshow(np.log10(cvar_np + 1e-8), cmap="turbo")
                axs[row, 0].set_title("Color Var (log10)", fontsize=12)
                axs[row, 1].imshow(np.log10(dvar_np + 1e-8), cmap="turbo")
                axs[row, 1].set_title("Depth Var (log10)", fontsize=12)
                axs[row, 2].axis("off")

            for ax in axs.flatten():
                ax.axis("off")
            plt.suptitle(f"LC submap {src_id}→{tgt_id}  view {vi}", y=0.98, fontsize=14)
            plt.tight_layout()
            fig_name = str(lc_vis_dir / f"{prefix}_view{vi:02d}.jpg")
            plt.savefig(fig_name, dpi=200, bbox_inches="tight")
            plt.clf()
            plt.close()

    def vis_tracking_loss_curve(self, frame_id: int, color_losses: list, depth_losses: list,
                                interval: int = 1) -> None:
        """Save a plot of color/depth loss convergence over tracking iterations for a single frame.

        Args:
            frame_id: current frame index.
            color_losses: color loss value at each iteration.
            depth_losses: depth loss value at each iteration.
            interval: save every N frames.
        """
        if frame_id % interval != 0:
            return

        (self.output_path / "tracking_vis").mkdir(exist_ok=True, parents=True)

        iters = list(range(len(color_losses)))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(iters, color_losses, color='tab:blue')
        ax1.set_title(f'Color Loss (frame {frame_id:04d})', fontsize=13)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(iters, depth_losses, color='tab:orange')
        ax2.set_title(f'Depth Loss (frame {frame_id:04d})', fontsize=13)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.grid(True)

        plt.tight_layout()
        fig_name = str(self.output_path / "tracking_vis" / f'loss_curve_{frame_id:04d}.jpg')
        plt.savefig(fig_name, dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close()
