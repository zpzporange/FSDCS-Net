import math
import os
from datetime import datetime

import numpy as np
from imblearn.under_sampling import NearMiss
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
from sklearn.svm import OneClassSVM
from torch import optim
from torch.backends import cudnn

from FSDCS_Net import FSDCS_Net, init_weights
from data_loader import get_loader
from evaluation import *


class Solver(object):
    def __init__(self, config, train_loader, valid_loader):
        # Top classifier
        self.classifier = config.classifier

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models
        self.model = None
        self.init_type = config.init_type
        self.gain = config.gain
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = config.criterion
        self.max_weight_ratio = config.max_weight_ratio
        self.weight_dice = config.weight_dice
        self.weight_bce = config.weight_bce
        self.smooth_coeff = config.smooth_coeff
        self.max_norm = config.max_norm
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.warmup_epochs = config.warmup_epochs
        self.patience = config.patience
        self.batch_size = config.batch_size

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.t = config.t
        self.build_model()

    def build_model(self):
        model = FSDCS_Net(batch_size=self.batch_size, t=self.t)
        init_weights(model, init_type=self.init_type, gain=self.gain)
        self.model = model

        self.optimizer = optim.AdamW(list(self.model.parameters()),
                                     lr=self.lr,
                                     betas=(self.beta1, self.beta2),
                                     weight_decay=1e-2)

        self.model.to(self.device)

    # Warmup + Cosine Annealing的学习率调整策略
    def warmup_cosine_annealing_scheduler(self, optimizer, warmup_epochs, total_epochs, init_lr, max_lr):

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                alpha = float(epoch) / float(max(1, warmup_epochs))
                alpha_tensor = torch.tensor(alpha).float()
                return init_lr + 0.5 * (max_lr - init_lr) * (1 + torch.sigmoid(-5 * (alpha_tensor - 0.5)))
            else:
                # 将计算结果转换为张量
                cos_input = torch.tensor(
                    (epoch - warmup_epochs) / (total_epochs - warmup_epochs)).float() * torch.tensor(torch.pi)
                return max_lr - 0.5 * (max_lr - init_lr) * (1 + torch.cos(cos_input))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def reset_grad(self, optimizer):
        """Zero the gradient buffers using the provided optimizer."""
        optimizer.zero_grad()

    # 划分同频子图
    def dct_cfsi(self, dct):
        # Co-frequency sub-images
        batch_size, _, col, row = dct.shape
        subdct = torch.zeros((batch_size, batch_size * 64, col // 8, row // 8), dtype=dct.dtype, device=dct.device)

        for i in range(batch_size):
            for j in range(64):
                subdct[i, j + i * 64, :, :] = dct[i, :, j // 8:col:8, j % 8:row:8]

        return subdct

    # 同频子图重组
    def dct_cfsi_re(self, subdct):
        # Co-frequency sub-images
        batch_size, _, subcol, subrow = subdct.shape
        col, row = subcol * 8, subrow * 8
        dct = torch.zeros((batch_size, 1, col, row), dtype=subdct.dtype, device=subdct.device)

        for i in range(batch_size):
            for m in range(8):
                for n in range(8):
                    dct[i, :, m:col:8, n:col:8] = subdct[i, m * 8 + n, :, :]

        return dct

    # 损失归一化，不会改变单调性
    def loss_normalization(self, loss):
        # 使用反正切函数转换loss
        atan_loss = math.atan(loss)

        # 由于arctan函数的输出范围是(-π/2, π/2)，我们可以直接使用这些值作为最小值和最大值
        min_atan_loss = -math.pi / 2
        max_atan_loss = math.pi / 2

        # 应用最小-最大归一化
        normalized_loss = (atan_loss - min_atan_loss) / (max_atan_loss - min_atan_loss)

        return normalized_loss

    # 神经网络训练及保存
    def train_val_model(self):
        torch.backends.cudnn.benchmark = True

        patience = self.patience  # 早停机制的耐心值，即连续多少个 epoch 没有提升就停止
        counter = 0  # 计数器，用于跟踪没有提升的 epoch 数

        scheduler = self.warmup_cosine_annealing_scheduler(self.optimizer, self.warmup_epochs, self.num_epochs, self.lr,
                                                           self.lr * 2)

        best_score = 0.

        for epoch in range(self.num_epochs):

            train_loss = 0.
            val_loss = 0.
            # 定义包含所有评估指标和对应函数的字典
            train_metrics = {
                'accuracy': 0.,  # Accuracy
                'sensitivity': 0.,  # Sensitivity (Recall)
                'specificity': 0.,  # Specificity
                'precision': 0.,  # Precision
                'F1': 0.,  # F1 Score
                'auroc': 0.,  # Area Under ROC Curve (AUROC)
                'auprc': 0.  # Area Under Precision-Recall Curve (AUPRC)
            }
            val_metrics = {
                'accuracy': 0.,  # Accuracy
                'sensitivity': 0.,  # Sensitivity (Recall)
                'specificity': 0.,  # Specificity
                'precision': 0.,  # Precision
                'F1': 0.,  # F1 Score
                'auroc': 0.,  # Area Under ROC Curve (AUROC)
                'auprc': 0.  # Area Under Precision-Recall Curve (AUPRC)
            }
            metric_functions = {
                'accuracy': get_accuracy,
                'sensitivity': get_sensitivity,
                'specificity': get_specificity,
                'precision': get_precision,
                'F1': get_F1,
                'auroc': get_AUROC,
                'auprc': get_AUPRC,
            }

            # 模型训练
            # 初始化记录批次数量的变量，以及上个批次的类别权重
            train_batch_count = 0
            weight_pos_last = 0.
            weight_neg_last = 0.

            self.model.train(True)

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    outputs = self.dct_cfsi_re(outputs)
                    loss, weight_pos_last, weight_neg_last = self.criterion(outputs, labels,
                                                                            max_weight_ratio=self.max_weight_ratio,
                                                                            weight_dice=self.weight_dice,
                                                                            weight_bce=self.weight_bce,
                                                                            weight_pos_last=weight_pos_last,
                                                                            weight_neg_last=weight_neg_last,
                                                                            smooth_coeff=self.smooth_coeff)
                    train_loss += loss.item()
                # Backprop + optimize
                self.optimizer.zero_grad()  # Clear gradients
                loss.backward()
                # Clip gradients to avoid explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_norm)  # Set max_norm as needed
                self.optimizer.step()

                # 为了使用混合精度，必须使用BCEWithLogitsLoss
                # 而BCEWithLogitsLoss要求用模型的原始输出来计算损失，因此sigmoid要放在后面
                # sigmoid缩放到01之间，配合阈值计算各种指标
                outputs = torch.sigmoid(outputs)
                for metric in train_metrics:
                    train_metrics[metric] += metric_functions[metric](outputs, labels)
                train_batch_count += 1

            # 模型验证
            val_batch_count = 0
            weight_pos_last = 0.
            weight_neg_last = 0.

            self.model.eval()

            for i, (images, labels) in enumerate(self.valid_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    outputs = self.dct_cfsi_re(outputs)
                    loss, weight_pos_last, weight_neg_last = self.criterion(outputs, labels,
                                                                            max_weight_ratio=self.max_weight_ratio,
                                                                            weight_dice=self.weight_dice,
                                                                            weight_bce=self.weight_bce,
                                                                            weight_pos_last=weight_pos_last,
                                                                            weight_neg_last=weight_neg_last,
                                                                            smooth_coeff=self.smooth_coeff)
                    val_loss += loss.item()

                # 为了使用混合精度，必须使用BCEWithLogitsLoss
                # 而BCEWithLogitsLoss要求用模型的原始输出来计算损失，因此sigmoid要放在后面
                # sigmoid缩放到01之间，配合阈值计算各种指标
                outputs = torch.sigmoid(outputs)
                for metric in val_metrics:
                    val_metrics[metric] += metric_functions[metric](outputs, labels)
                val_batch_count += 1

            # 计算训练和验证每个指标的平均值
            for metric in train_metrics:
                train_metrics[metric] = train_metrics[metric] / train_batch_count
            for metric in val_metrics:
                val_metrics[metric] = val_metrics[metric] / val_batch_count
            train_loss = train_loss / train_batch_count
            val_loss = val_loss / val_batch_count
            # 模型的得分用recall+auprc再减去归一化之后的损失
            model_score = val_metrics["sensitivity"] + val_metrics["auprc"] - self.loss_normalization(val_loss)
            # Print the log info
            print(f'Epoch [{epoch + 1}/{self.num_epochs}]')
            print(f'Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}')
            print(
                f'Train Acc: {train_metrics["accuracy"] * 100:.6f}%  |  Val Acc: {val_metrics["accuracy"] * 100:.6f}%')
            print(
                f'Train Rec: {train_metrics["sensitivity"] * 100:.6f}%  |  Val Rec: {val_metrics["sensitivity"] * 100:.6f}%')
            print(
                f'Train Spec: {train_metrics["specificity"] * 100:.6f}%  |  Val Spec: {val_metrics["specificity"] * 100:.6f}%')
            print(
                f'Train Prec: {train_metrics["precision"] * 100:.6f}%  |  Val Prec: {val_metrics["precision"] * 100:.6f}%')
            print(f'Train F1: {train_metrics["F1"] * 100:.6f}%  |  Val F1: {val_metrics["F1"] * 100:.6f}%')
            print(f'Train AUROC: {train_metrics["auroc"]:.6f}  |  Val AUROC: {val_metrics["auroc"]:.6f}')
            print(f'Train AUPRC: {train_metrics["auprc"]:.6f}  |  Val AUPRC: {val_metrics["auprc"]:.6f}')

            # Save Best model
            if model_score > (best_score + 1e-12):
                best_score = model_score
                best_model = self.model.state_dict()
                best_epoch = epoch + 1
                best_lr = self.optimizer.param_groups[0]['lr']
                print(f'Best model score : {best_score:.6f} at epoch {best_epoch}')
                model_path = os.path.join(self.model_path,
                                          f"{self.model_type}-{best_epoch}-{best_lr:.8f}.pkl")
                torch.save(best_model, model_path)
                counter = 0
            else:
                counter += 1

            # 如果计数器达到耐心值，则触发早停
            if counter > patience and epoch > self.warmup_epochs:
                print(f"Early stopping after {epoch + 1} epochs with no improvement.")
                break

            # Decay learning rate
            scheduler.step()

            # 在每个epoch结束后调用torch.cuda.empty_cache()，节约显存用
            torch.cuda.empty_cache()

    # 转换数据加载器为np数据格式
    def convert_loader_to_data(self, model, loader, use_nearmiss=False):
        num_batches = len(loader)
        batch_size = loader.batch_size
        # 预先分配足够大的NumPy数组
        datas = np.zeros((num_batches * batch_size * 512 * 512, 1), dtype=np.float16)
        labels = np.zeros((num_batches * batch_size * 512 * 512, 1), dtype=np.float16)

        # 用于跟踪当前填充位置的索引
        current_index = 0

        for images, label in loader:
            images = images.to(self.device)
            features = model(images)
            features = self.dct_cfsi_re(features)

            # 直接将数据复制到预分配的数组中
            datas[current_index:current_index + batch_size * 512 * 512] = features.detach().cpu().numpy().reshape(-1, 1,
                                                                                                                  order='F')
            labels[current_index:current_index + batch_size * 512 * 512] = label.numpy().reshape(-1, 1, order='F')

            # 更新索引
            current_index += batch_size * 512 * 512

        labels = labels == max(labels)

        # 如果启用NearMiss-3
        if use_nearmiss:
            # 初始化 NearMiss-3
            nm3 = NearMiss(version=3, n_neighbors_ver3=3, sampling_strategy={0: 1, 1: 200}, n_jobs=-1)
            # 应用 NearMiss-3
            datas, labels = nm3.fit_resample(datas[:current_index], labels[:current_index].ravel())

        return datas, labels

    def train_classifier(self):
        model = FSDCS_Net().to(self.device)
        # 指定模型文件的路径
        model_path = './models/FSDCS_Net_pretrained.pkl'
        # 加载模型状态字典
        state_dict = torch.load(model_path)

        # 应用状态字典到你的模型实例
        model.load_state_dict(state_dict)
        # 测试阶段不更新模型参数
        model.eval()

        # 转换训练和验证数据加载器
        train_data, train_labels = self.convert_loader_to_data(model, self.train_loader, use_nearmiss=False)

        # 训练并保存模型
        self.classifier.fit(train_data, train_labels)
        # self.classifier.fit(train_data)
        # 获取当前日期和时间
        current_time = datetime.now()
        # 格式化日期和时间为字符串
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # 将分类器保存到包含日期和时间的文件名中
        dump(self.classifier, f'./models/SVM_{time_str}.joblib', compress=3)

    def test(self):
        model = FSDCS_Net().to(self.device)
        # 指定模型文件的路径
        model_path = './models/FSDCS_Net_pretrained.pkl'
        # 加载模型状态字典
        state_dict = torch.load(model_path)
        # 应用状态字典到你的模型实例
        model.load_state_dict(state_dict)
        # 测试阶段不更新模型参数
        model.eval()

        classifier = load('./models/SVM.joblib')

        # 用于存储所有嵌入率下的测试指标
        all_test_metrics = {}

        # 遍历所有嵌入率
        for embedding_rate in ['0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']:
            test_loader = get_loader(image_path=config.data_path, batch_size=1, num_workers=2, mode='test',
                                     augmentation_prob=0., dct_domain=True, embedding_rate=embedding_rate,
                                     sample_ratio=0.1)
            test_metrics = {
                'accuracy': 0.,  # Accuracy
                'Recall': 0.,  # Sensitivity (Recall)
                'specificity': 0.,  # Specificity
                'precision': 0.,  # Precision
                'F1': 0.,  # F1 Score
                'auroc': 0.,  # Area Under ROC Curve (AUROC)
                'auprc': 0.,  # Area Under Precision-Recall Curve (AUPRC)
            }

            test_data, test_labels = self.convert_loader_to_data(model, test_loader)
            test_labels = (test_labels == max(test_labels)).astype(int)

            pred_labels = classifier.predict(test_data)
            pred_labels = (pred_labels > 0.5).astype(int)

            test_metrics['accuracy'] += accuracy_score(test_labels, pred_labels)
            test_metrics['Recall'] += recall_score(test_labels, pred_labels)
            test_metrics['specificity'] += recall_score(test_labels, pred_labels, pos_label=0)
            test_metrics['precision'] += precision_score(test_labels, pred_labels)
            test_metrics['F1'] += f1_score(test_labels, pred_labels)
            test_metrics['auroc'] += roc_auc_score(test_labels, pred_labels)
            test_metrics['auprc'] += average_precision_score(test_labels, pred_labels)

            # 将当前嵌入率的测试指标存储起来
            all_test_metrics[embedding_rate] = test_metrics

            # 打印当前嵌入率的测试结果
            print(f"Embedding Rate: {embedding_rate} - Test Metrics: {test_metrics}")

        return all_test_metrics


class Config:
    classifier = OneClassSVM(nu=1 / 18000, kernel='poly', degree=2, gamma=1, coef0=1.0, shrinking=True)

    # model hyper-parameters
    t = 3  # t for Recurrent step
    init_type = 'xavier'
    gain = 0.01
    criterion = CombinedLoss_average()
    max_weight_ratio = 4.0
    weight_dice = 0.57
    weight_bce = 1. - weight_dice
    smooth_coeff = 0.85
    max_norm = 0.57

    # training hyper-parameters
    img_ch = 1
    output_ch = 1
    num_epochs = 25
    warmup_epochs = 5
    patience = 5
    batch_size = 1
    num_workers = 2
    lr = 6.225274857608964e-08
    beta1 = 0.5
    beta2 = 0.95
    augmentation_prob = 0.

    # misc
    mode = None
    model_path = './models'
    data_path = '../StegoDataSets/JstegM'
    result_path = './result/'

    cuda_idx = 1


def main(config):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    train_loader = get_loader(image_path=config.data_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=0.,
                              sample_ratio=0.055,
                              dct_domain=True)
    valid_loader = get_loader(image_path=config.data_path,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='val',
                              augmentation_prob=0.,
                              dct_domain=True)

    solver = Solver(config, train_loader, valid_loader)

    # Train and sample the images
    if config.mode == 'train_model':
        solver.train_val_model()
    elif config.mode == 'train_classifier':
        solver.train_classifier()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    config = Config()
    # config.mode = 'train_model'
    # config.mode = 'train_classifier'
    config.mode = 'test'

    main(config)
