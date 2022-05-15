import torch
import torch.nn.functional as F

def iou_binary(preds, targets):
    intersection = torch.sum(torch.logical_and(preds, targets), dim=(1,2))
    union = torch.sum(torch.logical_or(preds, targets), dim=(1,2))
    iou = (intersection + 1) / (union + 1)
    return iou


def iou(preds, targets, threshold=0.5):
    preds_prob = F.softmax(preds, 1)
    preds_map = preds_prob > threshold
    intersection = torch.sum(torch.logical_and(preds_map[:, 1, ...], targets), dim=(1,2))
    union = torch.sum(torch.logical_or(preds_map[:, 1, ...], targets), dim=(1,2))
    iou = (intersection + 1) / (union + 1)
    return iou

def dice_values(preds, targets, threshold=0.5):
    preds_prob = F.softmax(preds, dim=1)
    preds_map = preds_prob > threshold
    intersection = torch.sum(targets * preds_map[:,1,...],dim=(1,2))
    pred_size = torch.sum(preds_map[:,1,...], dim=(1,2))
    target_size = torch.sum(targets, dim=(1,2))
    dice_coeff = ((2 * intersection + 1)/ (pred_size + target_size + 1))
    return dice_coeff

def soft_dice_values(preds, targets):
    preds_prob = F.softmax(preds, dim=1)
    intersection = torch.sum(targets * preds_prob[:,1,...],dim=(1,2))
    pred_size = torch.sum(preds_prob[:,1,...], dim=(1,2))
    target_size = torch.sum(targets, dim=(1,2))
    dice_coeff = ((2 * intersection + 1)/ (pred_size + target_size + 1))
    return dice_coeff

def soft_dice_loss(preds, targets):
    dice_coeffs = soft_dice_values(preds, targets)
    return torch.sum(1 - dice_coeffs)

def dice_loss(preds, targets):
    dice_coeffs = dice_values(preds, targets)
    return torch.sum(1 - dice_coeffs)
