import torch, roma
import numpy as np
import copy

from src.gsr.renderer import render
from src.gsr.loss import get_loss_tracking, get_loss_tracking_var
from src.gsr.overlap import compute_overlap_gaussians
from src.utils.pose_utils import update_pose


class CustomPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False

def viewpoint_localizer(viewpoint, gaussians, base_lr: float=1e-3, use_variance: bool=True, config_var: dict=None): # True    False
    """Localize a single viewpoint in a 3DGS

    Args:
        viewpoint (Camera): Camera instance
        gaussians (Gaussians): 3D Gaussians to locate the viewpoint
        base_lr (float, optional). Defaults to 1e-3.

    Returns:
        _type_: _description_
    """
    opt_params = []
    pipe = CustomPipeline()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
    config = {
        'Training': {
            'monocular': False,
            "rgb_boundary_threshold": 0.01,
        },
        'var': config_var if config_var is not None else {'use_nlldepth': False},
    }

    init_T = viewpoint.get_T.detach()
    
    opt_params.append(
        {
            "params": [viewpoint.cam_rot_delta],
            "lr": 3*base_lr,
            "name": "rot_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.cam_trans_delta],
            "lr": base_lr,
            "name": "trans_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_a],
            "lr": 0.01,
            "name": "exposure_a_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_b],
            "lr": 0.01,
            "name": "exposure_b_{}".format(viewpoint.uid),
        }
    )
    optimizer = torch.optim.Adam(opt_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.98, patience=5) #, verbose=False

    # Re-orthogonalize R to prevent singular matrix errors from numerical drift or degenerate initialization
    with torch.no_grad():
        U, _, Vh = torch.linalg.svd(viewpoint.R.float())
        viewpoint.R = (U @ Vh).to(dtype=torch.float32, device=viewpoint.device)

    loss_log = []
    opt_iterations = 100
    for tracking_itr in range(opt_iterations):
        optimizer.zero_grad()
        render_pkg = render(
            viewpoint, gaussians, pipe, bg_color
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        
        if use_variance:
            # variance = render_pkg["variance"].detach()
            # loss = get_loss_tracking_var(config, image, depth, opacity, viewpoint, variance, tau_registration=tau_registration, w_limit=w_limit)
            
            # sensor_uncert = self.sensor_uncertainty_MLP(gt_depth).detach()
            # rendered_color_var, rendered_depth_var = render_pkg["render_color_variance"], render_pkg["render_depth_variance"]

            # loss = get_loss_tracking_var(config, image, depth, opacity, viewpoint, None, rendered_color_var=rendered_color_var, rendered_depth_var=rendered_depth_var)
            loss = get_loss_tracking(config, image, depth, opacity, viewpoint)
        else:
            loss = get_loss_tracking(config, image, depth, opacity, viewpoint)
        loss.backward()
        loss_log.append(loss.item())
        
        with torch.no_grad():
            optimizer.step()
            scheduler.step(loss)
            converged = update_pose(viewpoint)
        
        if converged:
            break
    
    rel_tsfm = (init_T.inverse() @ viewpoint.get_T).inverse()
    loss_residual = loss.item()

    with torch.no_grad():
        final_pkg = render(viewpoint, gaussians, pipe, bg_color)
        gt_image = viewpoint.original_image.cuda().detach() if viewpoint.original_image is not None else None
        gt_depth = torch.from_numpy(viewpoint.depth).float().cuda() if viewpoint.depth is not None else None
        final_render = {
            "image": final_pkg["render"].detach(),
            "depth": final_pkg["depth"].detach(),
            "color_var": final_pkg["render_color_variance"].detach() if final_pkg.get("render_color_variance") is not None else None,
            "depth_var": final_pkg["render_depth_variance"].detach() if final_pkg.get("render_depth_variance") is not None else None,
            "gt_image": gt_image,
            "gt_depth": gt_depth,
        }

    return converged, rel_tsfm, loss_residual, loss_log, final_render

def gaussian_registration(src_dict, tgt_dict, config: dict, visualize=False, config_var: dict=None):
    """_summary_

    Args:
        src_dict (dict): dictionary of source gaussians and its keyframes
        tgt_dict (dict): dictionary of target gaussians and its keyframes
        base_lr (float, optional): the base learning rate for optimization. Defaults to 5e-3.

    Returns:
        dict: dictionary of registration result
    """

    init_overlap = compute_overlap_gaussians(src_dict['gaussians'], tgt_dict['gaussians'], 0.1)
    if init_overlap< 0.2:
        print("Initial overlap between two submaps are too small, skipping ...")
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item(),
        }
    
    src_3dgs, src_view_list = copy.deepcopy(src_dict['gaussians']), copy.deepcopy(src_dict['cameras'])
    tgt_3dgs, tgt_view_list = copy.deepcopy(tgt_dict['gaussians']), copy.deepcopy(tgt_dict['cameras'])
    
    # compute gt tsfm
    src_keyframe= src_dict['cameras'][0].get_T.detach()
    src_gt= src_dict['cameras'][0].get_T_gt.detach()
    tgt_keyframe= tgt_dict['cameras'][0].get_T.detach()
    tgt_gt= tgt_dict['cameras'][0].get_T_gt.detach()
    delta_src = src_gt.inverse() @ src_keyframe
    delta_tgt = tgt_gt.inverse() @ tgt_keyframe
    gt_tsfm = delta_tgt.inverse() @ delta_src
    
    # similarity choosing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src_desc, tgt_desc = src_dict['kf_desc'], tgt_dict['kf_desc']
    
    score_cross = torch.einsum("id,jd->ij", src_desc.to(device), tgt_desc.to(device))
    score_best_src, _ = score_cross.topk(1)
    _, ii = score_best_src.view(-1).topk(2)
    
    score_best_tgt, _ = score_cross.T.topk(1)
    _, jj = score_best_tgt.view(-1).topk(2)
    
    src_view_list = [src_view_list[i.item()] for i in ii]
    tgt_view_list = [tgt_view_list[j.item()] for j in jj]
    
    pred_list, residual_list, converged_list, loss_log_list, render_list = [], [], [], [], []

    pipe = CustomPipeline()
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda", requires_grad=False)
    # per-cam
    for viewpoint in src_view_list:

        # use rendered image as target not the raw observation
        if config["use_render"]:
            render_pkg = render(viewpoint, src_3dgs, pipe, bg_color)
            viewpoint.load_rgb(render_pkg['render'].detach())
            viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
        else:
            viewpoint.load_rgb()
        converged, pred_tsfm, residual, loss_log, final_render = viewpoint_localizer(viewpoint, tgt_3dgs, config["base_lr"], config_var=config_var)
        pred_list.append(pred_tsfm)
        residual_list.append(residual)
        converged_list.append(converged)
        loss_log_list.append(loss_log)
        render_list.append(final_render)

    for viewpoint in tgt_view_list:
        if config["use_render"]:
            render_pkg = render(viewpoint, tgt_3dgs, pipe, bg_color)
            viewpoint.load_rgb(render_pkg['render'].detach())
            viewpoint.depth = render_pkg['depth'].squeeze().detach().cpu().numpy()
        else:
            viewpoint.load_rgb()
        converged, pred_tsfm, residual, loss_log, final_render = viewpoint_localizer(viewpoint, src_3dgs, config["base_lr"], config_var=config_var)
        pred_list.append(pred_tsfm.inverse())
        residual_list.append(residual)
        converged_list.append(converged)
        loss_log_list.append(loss_log)
        render_list.append(final_render)
    
    
    pred_tsfms = torch.stack(pred_list)
    residuals = torch.Tensor(residual_list).cuda().float()
    # probability based on residuals
    prob = 1/residuals / (1/residuals).sum()
    
    M = torch.sum(prob[:, None, None] * pred_tsfms[:,:3,:3], dim=0)
    try:
        R_w = roma.special_procrustes(M)
    except Exception as e:
        print(f"Error in roma.special_procrustes: {e}")
        return {
            'successful': False,
            "pred_tsfm": torch.eye(4).cuda(),
            "gt_tsfm": torch.eye(4).cuda(),
            "overlap": init_overlap.item()
        }
    t_w = torch.sum(prob[:, None] * pred_tsfms[:,:3, 3], dim=0)
    
    best_tsfm = torch.eye(4).cuda().float()
    best_tsfm[:3,:3]  = R_w
    best_tsfm[:3, 3]  = t_w
    
    result_dict = {
        "gt_tsfm": gt_tsfm,
        "pred_tsfm": best_tsfm,
        "successful": True,
        "best_viewpoint": src_view_list[0].get_T,
        "renders": render_list,
        "loss_logs": loss_log_list,
    }
    
    if visualize:
        import matplotlib
        import matplotlib.pyplot as plt
        from src.gsr.utils import visualize_registration
        matplotlib.use('TkAgg')
        plt.figure(figsize=(10, 6))
        for log in loss_log_list:
            plt.plot(log)

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves of 10 Independent Optimizations')
        plt.legend([f'Run {i+1}' for i in range(len(loss_log_list))], loc='upper right')
        plt.grid(True)
        plt.show()
        
        visualize_registration(src_3dgs, tgt_3dgs, best_tsfm, gt_tsfm)
    
    del src_3dgs, src_view_list, tgt_3dgs, tgt_view_list
    return result_dict



