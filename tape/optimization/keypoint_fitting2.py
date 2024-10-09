from cv2 import KeyPoint
import torch
import torch.nn as nn
from typing import Dict
import numpy as np
import constants
import os
import pickle
from .optimization_task import OptimizationTask, rel_change
from .losses import keypoint_fitting_loss
from tape.utils.geometry import batch_rodrigues
from tape.utils.render_openpose import render_openpose
from tape.utils.geometry import rotation_matrix_to_angle_axis,aa_to_rotmat, batch_adv_disc_l2_loss, batch_encoder_disc_l2_loss, perspective_projection

class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=torch.float32, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)

class KeypointFitting(OptimizationTask):

    def __call__(self,
                 flow_net: nn.Module,
                 regression_output: Dict,
                 data: Dict,
                 full_frame: bool = True,
                 use_hips: bool = False) -> Dict:
        """
        Fit SMPL to 2D keypoint data.
        Args:
            flow_net (nn.Module): Pretrained Conditional Normalizing Flows network.
            regression_output (Dict): Output of TAPE for the given input images.
            data (Dict): Dictionary containing images and their corresponding annotations.
            full_frame (bool): If True, perform fitting in the original image. Otherwise fit in the cropped box.
            use_hips (bool): If True, use hip keypoints for fitting. Hips are usually problematic.
        Returns:
            Dict: Optimization output containing SMPL parameters, camera, vertices and model joints.
        """
        pred_cam = regression_output['pred_cam'][:, 0]
        import ipdb
        ipdb.set_trace()
        batch_size = pred_cam.shape[0]
        '''
        
        for k in data.keys():
            if k == 'instance_id'or k=='features' or k == 'smpl_params_is_axis_angle' or k=='imgname' or k=='img_name' or k=='bbox' or k=='idx'or k=='keypoints_2d' or k=='orig_keypoints_2d' or k=='keypoints_3d' :
                continue
            if k == 'smpl_params':# or k == 'has_smpl_params':
                for v in data[k].keys():
                    data[k][v] = data[k][v].reshape(batch_size,-1)
                continue
            if k == 'has_smpl_params':
                for v in data[k].keys():
                    data[k][v] = data[k][v].reshape(batch_size)
                continue
            data[k] = data[k].reshape(batch_size,-1)
        '''
        # Differentiating between fitting on the cropped box or the original image coordinates
        if full_frame:
            # Compute initial camera translation
            box_center = data['box_center']
            box_size = data['box_size']
            img_size = data['img_size']
            camera_center = 0.5 * img_size
            depth = 2 * self.cfg.EXTRA.FOCAL_LENGTH / (box_size.reshape(batch_size, 1) * pred_cam[:,0].reshape(batch_size, 1) + 1e-9)
            init_cam_t = torch.zeros_like(pred_cam)
            init_cam_t[:, :2] = pred_cam[:, 1:] + (box_center - camera_center) * depth / self.cfg.EXTRA.FOCAL_LENGTH
            init_cam_t[:, -1] = depth.reshape(batch_size)
            keypoints_2d = data['orig_keypoints_2d'].reshape((batch_size,44,3))
        else:
            # Translation has been already computed in the forward pass
            init_cam_t = regression_output['pred_cam_t'][:, 0]
            keypoints_2d = data['keypoints_2d'].reshape((batch_size,44,3))
            keypoints_2d[:, :, :-1] = self.cfg.MODEL.IMAGE_SIZE * (keypoints_2d[:, :, :-1] + 0.5)
            img_size = torch.tensor([self.cfg.MODEL.IMAGE_SIZE, self.cfg.MODEL.IMAGE_SIZE], device=pred_cam.device, dtype=pred_cam.dtype).reshape(1, 2).repeat(batch_size, 1)
            camera_center = 0.5 * img_size

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.detach().clone()

        # Get detected joints and their confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, [-1]]
        if not use_hips:
            joints_conf[:, [8, 9, 12, 25+2, 25+3, 25+14]] *= 0.0


        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones_like(camera_center)

        # Get predicted betas
        betas = regression_output['pred_smpl_params']['betas'][:,0].detach().clone()
        
        #pose = regression_output['pred_smpl_params']['body_pose'][:,0].detach().clone()
        
        
        #pose = torch.zeros(batch_size,69).to(pred_cam.device)
        #for i in range(batch_size):
        #    pose[i]= rotation_matrix_to_angle_axis(regression_output['pred_smpl_params']['body_pose'][i,0].detach().clone()).reshape(69)
        # Initialize latent to 0 (mode of the regressed distribution)
        z = torch.zeros(batch_size, 144, requires_grad=True, device=pred_cam.device)

        # Make z, betas and camera_translation optimizable
        #pose.requires_grad=True
        z.requires_grad=True
        betas.requires_grad=True
        camera_translation.requires_grad = True
        # Setup optimizer
        opt_params = [z, betas, camera_translation]
        optimizer = torch.optim.LBFGS(opt_params, lr=1.0, line_search_fn='strong_wolfe')
        #pose_prior=MaxMixturePrior(prior_folder='data',
        #                                  num_gaussians=8,
        #                                  dtype=torch.float32).to(pred_cam.device)
        # As explained in Section 3.6 of the paper the pose prior reduces to ||z||_2^2
        def pose_prior():
            return (z ** 2).sum(dim=1)
        '''
        
        def angle_prior(pose):
            """
            Angle prior that penalizes unnatural bending of the knees and elbows
            """
            # We subtract 3 because pose does not include the global rotation of the model
            return torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
        '''
        # Define fitting closure
        def closure():
            optimizer.zero_grad()
            smpl_params, _, _, _, _ = flow_net(z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1) for k,v in smpl_params.items()}
            # Override regression betas with the optimizable variable
            smpl_params['betas'] = betas
            #smpl_params['body_pose']=pose #aa_to_rotmat(pose.reshape)
            smpl_output = self.smpl(**smpl_params, pose2rot=False)
            model_joints = smpl_output.joints

            loss = keypoint_fitting_loss(smpl_params, model_joints,
                                        camera_translation, camera_center, img_size,
                                        joints_2d, joints_conf, pose_prior,
                                        focal_length)
            loss.backward()
            return loss

        # Run fitting until convergence
        prev_loss = None
        for i in range(self.max_iters):
            loss = optimizer.step(closure)
            if i > 0:
                loss_rel_change = rel_change(prev_loss, loss.item())
                if loss_rel_change < self.ftol:
                    break
            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in opt_params if var.grad is not None]):
                break
            prev_loss = loss.item()

        # Get and save final parameter values
        opt_output = {}
        with torch.no_grad():
            smpl_params, _, _, _, _ = flow_net(z.unsqueeze(1))
            smpl_params = {k: v.squeeze(dim=1) for k,v in smpl_params.items()}
            smpl_params['betas'] = betas
            smpl_output = self.smpl(**smpl_params, pose2rot=False)
            model_joints = smpl_output.joints
            vertices = smpl_output.vertices
        opt_output['smpl_params'] = smpl_params
        opt_output['model_joints'] = model_joints
        opt_output['vertices'] = vertices
        opt_output['camera_translation'] = camera_translation.detach()

        return opt_output
