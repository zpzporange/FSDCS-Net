import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score


class CombinedLoss_average(nn.Module):
    def __init__(self, size_average=True):
        super(CombinedLoss_average, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, max_weight_ratio=6, weight_dice=0.5, weight_bce=0.5, weight_pos_last=0.1,
                weight_neg_last=0.1, smooth_coeff=0.95, smooth=1e-8):
        # 把张量形状变为一致
        inputs = inputs.squeeze(1)
        # 动态计算方式
        num_pos = torch.sum(targets == torch.max(targets)).item()
        num_neg = torch.sum(targets == torch.min(targets)).item()
        total = num_pos + num_neg
        weight_pos = num_neg / total
        weight_neg = num_pos / total
        # 使用权重缩放和平滑系数
        # max_weight_ratio 是正样本权重与负样本权重的最大允许比例
        max_weight_ratio = max_weight_ratio
        # smooth_coeff 是平滑系数，用于减少批次间的权重波动
        smooth_coeff = smooth_coeff
        # 计算平滑后的权重
        weight_pos_smooth = smooth_coeff * weight_pos + (1 - smooth_coeff) * weight_pos_last
        weight_neg_smooth = smooth_coeff * weight_neg + (1 - smooth_coeff) * weight_neg_last
        # 防止 weight_pos_smooth 和 weight_neg_smooth 为零，将它们封装在一个张量中，使用 torch.clamp 来限制这两个数值
        weight_values = torch.tensor([weight_pos_smooth, weight_neg_smooth])
        weight_values = torch.clamp(weight_values, min=1e-8)
        # 应用权重缩放
        if weight_values[0] / weight_values[1] > max_weight_ratio:
            weight_values[0] = max_weight_ratio * weight_values[1]
        elif weight_values[1] / weight_values[0] > max_weight_ratio:
            weight_values[1] = max_weight_ratio * weight_values[0]

        # BCEWithLogitsLoss
        pos_weight = torch.tensor([weight_values[0] / weight_values[1]], dtype=torch.float32).to(inputs.device)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)

        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = torch.clamp((inputs_flat * targets_flat).sum(), min=smooth)
        # 防止分母为零
        denominator = torch.clamp(inputs_flat.sum() + targets_flat.sum(), min=smooth)
        dice_loss = 1 - (2. * intersection + smooth) / denominator

        bce_loss = torch.clamp(weight_bce * bce_loss, min=1e-8)
        dice_loss = torch.clamp(weight_dice * dice_loss, min=1e-8)
        # Combined Loss
        # # 调和平均数
        # combined_loss = 2 * weight_dice * dice_loss * weight_bce * bce_loss / (weight_dice * dice_loss + weight_bce * bce_loss)
        # 几何平均数
        combined_loss = torch.sqrt(dice_loss * bce_loss)

        # # 打印各个损失和权重
        # print(f"pos_weight: {pos_weight.item():.6f}")
        # print(f"dice_loss: {dice_loss.item():.6f}")
        # print(f"bce_loss: {bce_loss.item():.6f}")

        return combined_loss, weight_pos, weight_neg


def get_accuracy(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    outputs = outputs > threshold
    labels = labels == torch.max(labels)
    corr = torch.sum(outputs == labels)
    tensor_size = outputs.numel()
    acc = float(corr) / float(tensor_size)

    return acc


def get_sensitivity(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    # Sensitivity == Recall
    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    # TP : True Positive
    # FN : False Negative
    TP = torch.sum((outputs == True) & (labels == True)).item()
    FN = torch.sum((outputs == False) & (labels == True)).item()

    SE = float(TP) / (float(TP + FN) + 1e-7)

    return SE


def get_specificity(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    # TN : True Negative
    # FP : False Positive
    TN = torch.sum((outputs == False) & (labels == False)).item()
    FP = torch.sum((outputs == True) & (labels == False)).item()

    SP = float(TN) / (float(TN + FP) + 1e-7)

    return SP


def get_precision(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    # TP : True Positive
    # FP : False Positive
    TP = torch.sum((outputs == True) & (labels == True)).item()
    FP = torch.sum((outputs == True) & (labels == False)).item()

    PC = float(TP) / (float(TP + FP) + 1e-7)

    return PC


def get_F1(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    # Sensitivity == Recall
    SE = get_sensitivity(outputs, labels, threshold=threshold)
    PC = get_precision(outputs, labels, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-10)

    return F1


def get_JS(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    # JS : Jaccard similarity
    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    Inter = torch.sum(outputs & labels)
    Union = torch.sum(outputs | labels)

    JS = float(Inter) / (float(Union) + 1e-10)

    return JS


def get_DC(outputs, labels, threshold=0.5):
    # 确保outputs和labels形状一致
    outputs = outputs.squeeze(1)  # 移除outputs的通道维度
    labels = labels.float()  # 确保labels是浮点张量以便比较

    # DC : Dice Coefficient
    outputs = outputs > threshold
    labels = labels == torch.max(labels)

    Inter = torch.sum(outputs & labels)
    DC = float(2 * Inter) / (float(torch.sum(outputs) + torch.sum(labels)) + 1e-10)

    return DC


def get_AUROC(outputs, labels, threshold=0.5):
    # Apply sigmoid to the outputs to get the probability
    outputs_probs = torch.sigmoid(outputs).squeeze(1)  # Remove the channel dimension and apply sigmoid
    labels = labels.float()  # Ensure labels is a float tensor for comparison

    # Flatten the tensors to convert them into 1D arrays
    outputs_probs_flat = outputs_probs.view(-1).cpu().detach().numpy()
    labels_flat = labels.view(-1).cpu().detach().numpy()

    # Convert smoothed labels to binary labels
    labels_binary = (labels_flat > threshold).astype(int)

    # Calculate AUROC
    AUROC = roc_auc_score(labels_binary, outputs_probs_flat)

    return AUROC


def get_AUPRC(outputs, labels, threshold=0.5):
    # Apply sigmoid to the outputs to get the probability
    outputs_probs = outputs.squeeze(1)  # Remove the channel dimension and apply sigmoid
    labels = labels.float()  # Ensure labels is a float tensor for comparison

    # Flatten the tensors to convert them into 1D arrays
    outputs_probs_flat = outputs_probs.view(-1).cpu().detach().numpy()
    labels_flat = labels.view(-1).cpu().detach().numpy()

    # Convert smoothed labels to binary labels
    labels_binary = (labels_flat > threshold).astype(int)

    # Calculate AUPRC
    AUPRC = average_precision_score(labels_binary, outputs_probs_flat)

    return AUPRC
