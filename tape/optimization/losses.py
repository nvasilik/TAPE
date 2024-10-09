import torch
import torch.nn as nn
from typing import Dict, Callable
from tape.utils.geometry import rotation_matrix_to_angle_axis,aa_to_rotmat, batch_adv_disc_l2_loss, batch_encoder_disc_l2_loss, perspective_projection

from tape.utils.geometry import perspective_projection
def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
def gmof(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Geman-McClure error function.
    Args:
        x (torch.Tensor): Raw error signal.
        sigma (float): Robustness hyperparameter
    Returns:
        torch.Tensor: Robust error signal
    """
    x_squared =  x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)

def keypoint_fitting_loss(smpl_params: Dict,
                          model_joints: torch.Tensor,
                          camera_translation: torch.Tensor,
                          camera_center: torch.Tensor,
                          img_size: torch.Tensor,
                          joints_2d: torch.Tensor,
                          joints_conf: torch.Tensor,
                          pose_prior: Callable,
                          focal_length: torch.Tensor,
                          sigma: float = 100.0,
                          pose_prior_weight: float = 4.0, #4.78,#4.0,
                          shape_prior_weight: float = 6.0) -> torch.Tensor:
    """
    Loss function for fitting the SMPL model on 2D keypoints.
    Args:
        model_joints (torch.Tensor): Tensor of shape (B, N, 3) containing the SMPL 3D joint locations.
        camera_translation (torch.Tensor): Tensor of shape (B, 3) containing the camera translation.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        img_size (torch.Tensor): Tensor of shape (B, 2) containing the image size in pixels (height, width).
        joints_2d (torch.Tensor): Tensor of shape (B, N, 2) containing the target 2D joint locations.
        joints_conf (torch.Tensor): Tensor of shape (B, N, 1) containing the target 2D joint confidences.
        pose_prior (Callable): Returns the pose prior value.
        focal_length (float): Focal length value in pixels.
        pose_prior_weight (float): Pose prior loss weight.
        shape_prior_weight (float): Shape prior loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    betas = smpl_params['betas']
    batch_size = int(betas.shape[0]/16)
    img_size = img_size.max(dim=-1)[0]

    # Heuristic for scaling data_weight with resolution used in SMPLify-X
    data_weight = (1000. / img_size).reshape(-1, 1, 1).repeat(1, 1, 2)

    # Project 3D model joints
    projected_joints = perspective_projection(model_joints, camera_translation, focal_length, camera_center=camera_center)

    # Compute robust reprojection loss
    reprojection_error = gmof(projected_joints - joints_2d, sigma)
    reprojection_loss = ((data_weight ** 2) * (joints_conf ** 2) * reprojection_error).sum()
    #pose = torch.zeros(batch_size,69).to(smpl_params['betas'].device)
    #for i in range(batch_size):
    #    pose[i]= rotation_matrix_to_angle_axis(smpl_params['body_pose'][i]).reshape(69)
        #
    # Compute pose prior loss
    #pose_prior_loss = (pose_prior_weight ** 2) * (pose_prior(pose,betas))#**2).sum()
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior().sum()
    # Compute shape prior loss
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum()
    
    #angle_prior_loss =(15.2 ** 2) *  (angle_prior(pose)).sum(dim=-1)
    
    # Smooth loss\
    #print('Smooth')
    flag=False
    if flag:
        projected_joints=projected_joints.reshape(batch_size,16,44,-1)

        smooth2d_loss=2*(projected_joints[:,1:,:]-projected_joints[:,:-1,:]).abs().sum()#0.5
        betas=betas.reshape(batch_size,16,-1)
        #smooth_shape_loos=15*(betas[1:,:]-betas[:-1,:]).abs().sum()#0.2
        smooth_shape_loos=15*(betas[:,1:,:]-betas[:,:-1,:]).abs().sum()#0.2
        pose = smpl_params['body_pose'].reshape(batch_size,16,23,3,3)
        #smooth_pose_loos=70*(pose[:,1:,:]-pose[:,:-1,:]).abs().sum()#0.2
        smooth_pose_loos=70*(pose[:,1:,:,:]-pose[:,:-1,:,:]).abs().sum()#0.2

    #print(smooth2d_loss)
    #print(smooth_shape_loos)


    # Add up all losses
    total_loss = reprojection_loss + pose_prior_loss + shape_prior_loss#+angle_prior_loss#+smooth_pose_loos+smooth_shape_loos#+smooth2d_loss

    return total_loss.sum()

def multiview_loss(smpl_params: Dict,
                   pose_prior: Callable,
                   pose_prior_weight: float = 1.0,
                   consistency_weight: float = 300.0):
    """
    Loss function for multiple view refinement (Eq. 12)
    Args:
        smpl_params (Dict): Dictionary containing the SMPL model parameters.
        pose_prior (Callable): Returns the pose prior value.
        pose_prior_weight (float): Pose prior loss weight.
        shape_prior_weight (float): Shape prior loss weight.
    Returns:
        torch.Tensor: Total loss value.
    """
    body_pose = smpl_params['body_pose']

    # Compute pose consistency loss.
    mean_pose = body_pose.mean(dim=0).unsqueeze(dim=0)
    pose_diff = ((body_pose - mean_pose) ** 2).sum(dim=-1)
    consistency_loss = consistency_weight ** 2 * pose_diff.sum()

    # Compute pose prior loss
    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior().sum()

    # Add up all losses
    total_loss = consistency_loss + pose_prior_loss

    return total_loss
