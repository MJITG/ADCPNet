import torch.nn.functional as F


def model_loss(weights, disp_ests, disp_gt, mask, training=True):
    all_losses = []
    if training:
        all_losses.append(weights[0] * F.smooth_l1_loss(disp_ests[0][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[1] * F.smooth_l1_loss(disp_ests[1][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[2] * F.smooth_l1_loss(disp_ests[2][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[3] * F.smooth_l1_loss(disp_ests[4][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[4] * F.smooth_l1_loss(disp_ests[5][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[5] * F.smooth_l1_loss(disp_ests[6][mask], disp_gt[mask], size_average=True))
    else:
        all_losses.append(weights[3] * F.smooth_l1_loss(disp_ests[0][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[4] * F.smooth_l1_loss(disp_ests[1][mask], disp_gt[mask], size_average=True))
        all_losses.append(weights[5] * F.smooth_l1_loss(disp_ests[2][mask], disp_gt[mask], size_average=True))
    # [print('loss[{}]: {}'.format(i, all_losses[i] / weights[i])) for i, loss in enumerate(all_losses)]
    return all_losses
