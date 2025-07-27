import os
import numpy as np
import cv2

from mmengine.config import Config
from mmengine.runner import Runner
from utils.compute_confusion_matrix import ConfusionMatrix



def eval(output_dir=None,
         gt_path=None,
         num_classes=2
         ):
    iou, dice, precision, recall, acc, fscore, mae = [], [], [], [], [], [], []
    pred_list = sorted(os.listdir(output_dir))
    gt_list = sorted(os.listdir(gt_path))
    cm = []
    for i in range(num_classes):
        cm.append([0, 0, 0, 0])
        iou.append(0)
        dice.append(0)
        precision.append(0)
        recall.append(0)
        acc.append(0)
        fscore.append(0)
    for pred_video, gt_video in zip(pred_list, gt_list):
        pred_list = sorted(os.listdir(os.path.join(output_dir, pred_video)))
        gt_list = sorted(os.listdir(os.path.join(gt_path, gt_video)))
        for pred_img, gt_img in zip(pred_list, gt_list):
            pred = cv2.imread(os.path.join(output_dir, pred_video, pred_img), cv2.IMREAD_GRAYSCALE)
            pred[np.where(pred!=0)] = 1
            gt = cv2.imread(os.path.join(gt_path, gt_video, gt_img), cv2.IMREAD_GRAYSCALE)
            if not pred.shape == gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                print(pred_video, pred_img)
            pred1, gt1 = pred.astype(np.int16), gt.astype(np.int16)
            mae.append(np.mean(np.abs(pred1-gt1)))
            a = np.abs(pred1 - gt1)
            for i in range(num_classes):
                confusion_matrix = ConfusionMatrix(test=pred, reference=gt, class_name=i)
                cur_tp, cur_fp, cur_tn, cur_fn = confusion_matrix.get_matrix()
                cm[i][0] += cur_tp
                cm[i][1] += cur_fp
                cm[i][2] += cur_tn
                cm[i][3] += cur_fn
    for i in range(num_classes):
        tp, fp, tn, fn = cm[i][0], cm[i][1], cm[i][2], cm[i][3]
        iou[i] = tp / (tp + fp + fn)
        dice[i] = 2 * tp / (2 * tp + fp + fn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        acc[i] = (tp + tn) / (tp + tn + fn + tn)
        fscore[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        print('________________________________________________________________________________________________________')
        print(
            f'第{i}类的iou为{iou[i]:.4f}，dice为{dice[i]:.4f}，precision为{precision[i]:.4f}，recall为{recall[i]:.4f}, acc为{acc[i]:.4f}, fscore为{fscore[i]:.4f}')
    miou = sum(iou) / len(iou)
    mdice = sum(dice) / len(dice)
    mrecall = sum(recall) / len(recall)
    macc = sum(acc) / len(acc)
    mprecision = sum(precision) / len(precision)
    mfscore = sum(fscore) / len(fscore)
    mmae = sum(mae) / len(mae)
    print('________________________________________________MEAN____________________________________________________')
    print(
        f'第miou为{miou:.4f}，mdice为{mdice:.4f}，mprecision为{mprecision:.4f}，mrecall为{mrecall:.4f}, macc为{macc:.4f}, mfscore为{mfscore:.4f}, mae为{mmae:.4f}')
    print('________________________________________________________________________________________________________')

if __name__ == '__main__':
    data_root = '' # path to vide
    gt_path = os.path.join(data_root, 'ann_dir', 'test')  # path to ground truth mask
    config_path = ''  # path to config files
    checkpoint_file = ''  # path to checkpoints
    output_dir = ''  # path to results
    config = Config.fromfile(config_path)
    config.val_evaluator['output_dir'] = output_dir
    config.load_from = checkpoint_file
    config.work_dir = output_dir
    # config.data_root = data_root
    runner = Runner.from_cfg(config)
    runner.val()
    eval(output_dir=output_dir, gt_path=gt_path)