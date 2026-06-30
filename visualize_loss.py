#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv11训练损失和PR曲线可视化工具

此脚本用于解析YOLOv11训练日志文件，提取并可视化：
1. 训练损失函数的变化趋势：box_loss、cls_loss、dfl_loss
2. 精确率-召回率(PR)曲线

作者: AI助手
日期: 2024
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号


class YOLOv11Visualizer:
    """YOLOv11训练损失和PR曲线可视化器类"""

    def __init__(self):
        """初始化可视化器"""
        self.loss_data = {
            'epoch': [],
            'box_loss': [],
            'cls_loss': [],
            'dfl_loss': []
        }
        self.metrics_data = {
            'epoch': [],
            'mAP50': [],
            'mAP50_95': []
        }
        self.pr_data = {
            'recall': [],
            'precision': []
        }

    def parse_log_file(self, log_file_path):
        """
        解析YOLO训练日志文件，提取损失数据
        """
        if not os.path.exists(log_file_path):
            print(f"错误: 日志文件 '{log_file_path}' 不存在")
            return False

        print(f"正在解析日志文件: {log_file_path}")

        # 用于匹配训练日志中的损失值的正则表达式
        train_pattern = re.compile(
            r'\s*\[\s*(\d+)\/\d+\]\s+\w+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
        )

        # 用于匹配验证指标的正则表达式
        val_pattern = re.compile(
            r'\s*\[\s*(\d+)\/\d+\]\s+val\s+mAP50: ([\d.]+)\s+mAP50-95: ([\d.]+)'
        )

        # 读取并解析日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_epoch = 0
        for line in lines:
            # 尝试匹配训练损失
            train_match = train_pattern.search(line)
            if train_match:
                epoch = int(train_match.group(1))
                box_val = float(train_match.group(2))
                obj_val = float(train_match.group(3))
                cls_val = float(train_match.group(4))

                # 这里假设obj_val就是dfl_loss
                dfl_val = obj_val

                # 记录每个epoch的最后一个值
                if epoch > current_epoch:
                    self.loss_data['epoch'].append(epoch)
                    self.loss_data['box_loss'].append(box_val)
                    self.loss_data['cls_loss'].append(cls_val)
                    self.loss_data['dfl_loss'].append(dfl_val)
                    current_epoch = epoch

            # 尝试匹配验证指标
            val_match = val_pattern.search(line)
            if val_match:
                epoch = int(val_match.group(1))
                mAP50 = float(val_match.group(2))
                mAP50_95 = float(val_match.group(3))

                self.metrics_data['epoch'].append(epoch)
                self.metrics_data['mAP50'].append(mAP50)
                self.metrics_data['mAP50_95'].append(mAP50_95)

        if not self.loss_data['epoch']:
            print("警告: 未能从日志文件中提取损失数据")
            return False

        print(f"成功提取 {len(self.loss_data['epoch'])} 个epoch的损失数据")
        return True

    def parse_ultralytics_results(self, results_dir):
        """
        解析Ultralytics库生成的results.csv文件
        """
        try:
            import pandas as pd
        except ImportError:
            print("错误: 请先安装pandas库 'pip install pandas'")
            return False

        results_path = os.path.join(results_dir, 'results.csv')
        if not os.path.exists(results_path):
            print(f"错误: 结果文件 '{results_path}' 不存在")
            return False

        print(f"正在解析结果文件: {results_path}")

        try:
            # 读取CSV文件
            df = pd.read_csv(results_path)

            # 提取损失数据
            if 'epoch' in df.columns:
                self.loss_data['epoch'] = df['epoch'].values.tolist()

                # 尝试提取各种损失值
                if 'box_loss' in df.columns:
                    self.loss_data['box_loss'] = df['box_loss'].values.tolist()
                elif 'box' in df.columns:
                    self.loss_data['box_loss'] = df['box'].values.tolist()

                if 'cls_loss' in df.columns:
                    self.loss_data['cls_loss'] = df['cls_loss'].values.tolist()
                elif 'cls' in df.columns:
                    self.loss_data['cls_loss'] = df['cls'].values.tolist()

                if 'dfl_loss' in df.columns:
                    self.loss_data['dfl_loss'] = df['dfl_loss'].values.tolist()
                elif 'dfl' in df.columns:
                    self.loss_data['dfl_loss'] = df['dfl'].values.tolist()
                elif 'obj' in df.columns:
                    print("警告: 未找到dfl_loss列，使用obj_loss代替")
                    self.loss_data['dfl_loss'] = df['obj'].values.tolist()

                # 提取评估指标
                if 'metrics/mAP50(B)' in df.columns:
                    self.metrics_data['mAP50'] = df['metrics/mAP50(B)'].values.tolist()
                    self.metrics_data['epoch'] = df['epoch'].values.tolist()
                if 'metrics/mAP50-95(B)' in df.columns:
                    self.metrics_data['mAP50_95'] = df['metrics/mAP50-95(B)'].values.tolist()

            return len(self.loss_data['epoch']) > 0

        except Exception as e:
            print(f"解析结果文件时出错: {str(e)}")
            return False

    def generate_synthetic_data(self, epochs=100):
        """
        生成合成的损失数据和PR数据用于演示
        """
        print(f"正在生成{epochs}个epoch的合成数据...")

        # 生成模拟的损失数据
        epochs_list = list(range(1, epochs + 1))

        # box_loss: 从0.05开始，指数下降到0.005
        box_loss = 0.05 * np.exp(-0.04 * np.array(epochs_list)) + 0.005 + 0.002 * np.random.randn(epochs)
        box_loss = np.maximum(box_loss, 0.001)  # 确保值为正

        # cls_loss: 从0.5开始，指数下降到0.05
        cls_loss = 0.5 * np.exp(-0.03 * np.array(epochs_list)) + 0.05 + 0.02 * np.random.randn(epochs)
        cls_loss = np.maximum(cls_loss, 0.01)  # 确保值为正

        # dfl_loss: 从0.3开始，指数下降到0.03
        dfl_loss = 0.3 * np.exp(-0.035 * np.array(epochs_list)) + 0.03 + 0.01 * np.random.randn(epochs)
        dfl_loss = np.maximum(dfl_loss, 0.005)  # 确保值为正

        # 保存生成的数据
        self.loss_data = {
            'epoch': epochs_list,
            'box_loss': box_loss.tolist(),
            'cls_loss': cls_loss.tolist(),
            'dfl_loss': dfl_loss.tolist()
        }

        # 生成模拟的mAP数据
        mAP50 = 0.3 + 0.6 * (1 - np.exp(-0.03 * np.array(epochs_list))) + 0.02 * np.random.randn(epochs)
        mAP50 = np.minimum(np.maximum(mAP50, 0.3), 0.95)  # 限制在0.3-0.95之间

        mAP50_95 = 0.1 + 0.4 * (1 - np.exp(-0.025 * np.array(epochs_list))) + 0.015 * np.random.randn(epochs)
        mAP50_95 = np.minimum(np.maximum(mAP50_95, 0.1), 0.6)  # 限制在0.1-0.6之间

        self.metrics_data = {
            'epoch': epochs_list,
            'mAP50': mAP50.tolist(),
            'mAP50_95': mAP50_95.tolist()
        }

        # 生成PR曲线数据 - 模拟更真实的曲线形状
        recall_points = np.linspace(0, 1, 100)
        # 生成一个阶梯状的PR曲线，更符合实际检测结果
        precision_points = np.ones_like(recall_points)

        # 创建阶梯式下降的PR曲线
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        precision_values = [0.95, 0.90, 0.80, 0.60, 0.30, 0.1]

        for i, (thresh, prec) in enumerate(zip(thresholds, precision_values)):
            mask = recall_points >= thresh
            precision_points[mask] = prec

        # 添加少量噪声使曲线更自然
        precision_points += 0.02 * np.random.randn(len(recall_points))
        precision_points = np.minimum(np.maximum(precision_points, 0), 1)

        self.pr_data = {
            'recall': recall_points.tolist(),
            'precision': precision_points.tolist()
        }

        return True

    def plot_pr_curve(self, output_dir='./output', show=True, save=True):
        """
        绘制精确率-召回率(PR)曲线
        """
        if not self.pr_data['precision'] or not self.pr_data['recall']:
            print("警告: 没有PR曲线数据，跳过PR曲线绘制")
            return False

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # 绘制PR曲线
        ax.plot(self.pr_data['recall'], self.pr_data['precision'], 'b-', linewidth=2)

        # 计算平均精确率 (mAP0.5)
        mAP = np.trapz(self.pr_data['precision'], self.pr_data['recall'])

        # 设置图表属性
        ax.set_title(f'Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加mAP标签
        ax.text(0.5, 0.95, f'mAP@0.5: {mAP:.3f}', transform=ax.transAxes,
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()

        # 保存图像
        if save:
            output_path = os.path.join(output_dir, 'yolov11_pr_curve.png')
            plt.savefig(output_path, dpi=300)
            print(f"PR曲线已保存至: {output_path}")

        # 显示图像
        if show:
            plt.show()

        return True

    def plot_loss_curves(self, output_dir='./output', show=True, save=True):
        """
        绘制损失函数曲线图，简化版本
        """
        # 检查是否有数据
        if not self.loss_data['epoch']:
            print("错误: 没有可绘制的损失数据")
            return False

        # 创建输出目录
        if save:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # 设置绘图风格
        plt.style.use('ggplot')

        # 创建单个图表，简化布局
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        # 绘制损失曲线
        ax.plot(self.loss_data['epoch'], self.loss_data['box_loss'],
                'b-', linewidth=2, label='box_loss')
        ax.plot(self.loss_data['epoch'], self.loss_data['cls_loss'],
                'r-', linewidth=2, label='cls_loss')
        ax.plot(self.loss_data['epoch'], self.loss_data['dfl_loss'],
                'g-', linewidth=2, label='dfl_loss')

        # 设置图表属性，简化标题和说明
        ax.set_title('YOLOv11 Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()

        # 保存图像
        if save:
            output_path = os.path.join(output_dir, 'yolov11_loss_curves.png')
            plt.savefig(output_path, dpi=300)
            print(f"损失曲线图已保存至: {output_path}")

        # 显示图像
        if show:
            plt.show()

        return True

    def save_loss_data(self, output_dir='./output'):
        """
        保存提取的损失数据和PR数据到CSV文件
        """
        try:
            import pandas as pd
        except ImportError:
            print("警告: 未安装pandas，跳过数据保存")
            return False

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建损失数据DataFrame
        loss_df = pd.DataFrame({
            'epoch': self.loss_data['epoch'],
            'box_loss': self.loss_data['box_loss'],
            'cls_loss': self.loss_data['cls_loss'],
            'dfl_loss': self.loss_data['dfl_loss']
        })

        # 保存损失数据
        loss_output_path = os.path.join(output_dir, 'yolov11_loss_data.csv')
        loss_df.to_csv(loss_output_path, index=False)
        print(f"损失数据已保存至: {loss_output_path}")

        # 保存PR数据
        if self.pr_data['precision'] and self.pr_data['recall']:
            pr_df = pd.DataFrame({
                'recall': self.pr_data['recall'],
                'precision': self.pr_data['precision']
            })
            pr_output_path = os.path.join(output_dir, 'yolov11_pr_data.csv')
            pr_df.to_csv(pr_output_path, index=False)
            print(f"PR曲线数据已保存至: {pr_output_path}")

        return True


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv11训练损失和PR曲线可视化工具')
    parser.add_argument('--log-file', type=str, help='训练日志文件路径')
    parser.add_argument('--results-dir', type=str, help='Ultralytics results.csv所在目录')
    parser.add_argument('--output-dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='生成合成数据时的epoch数量')
    parser.add_argument('--no-show', action='store_true', help='不显示图像')
    parser.add_argument('--no-save', action='store_true', help='不保存图像')
    parser.add_argument('--pr-only', action='store_true', help='只绘制PR曲线')
    parser.add_argument('--loss-only', action='store_true', help='只绘制损失曲线')
    args = parser.parse_args()

    # 创建可视化器实例
    visualizer = YOLOv11Visualizer()

    # 获取损失数据
    success = False

    # 首先尝试从日志文件中解析
    if args.log_file:
        success = visualizer.parse_log_file(args.log_file)

    # 如果日志文件解析失败，尝试从results.csv中解析
    if not success and args.results_dir:
        success = visualizer.parse_ultralytics_results(args.results_dir)

    # 如果都失败了，生成合成数据用于演示
    if not success:
        print("\n未找到有效的训练日志文件，将生成合成数据用于演示...")
        success = visualizer.generate_synthetic_data(args.epochs)

    # 绘制并保存图表
    if success:
        # 创建并提示runs文件夹信息
        runs_dir = './runs'
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)

        # 根据参数决定绘制内容
        if not args.pr_only:
            print("\n开始绘制损失曲线图...")
            visualizer.plot_loss_curves(
                output_dir=args.output_dir,
                show=not args.no_show,
                save=not args.no_save
            )

        if not args.loss_only:
            print("\n开始绘制PR曲线...")
            visualizer.plot_pr_curve(
                output_dir=args.output_dir,
                show=not args.no_show,
                save=not args.no_save
            )

        # 保存数据到CSV文件
        visualizer.save_loss_data(args.output_dir)

        print(f"\n注意: runs文件夹已创建，可用于存放训练结果和模型权重")
        print("\n=== 可视化完成 ===")


if __name__ == '__main__':
    main()