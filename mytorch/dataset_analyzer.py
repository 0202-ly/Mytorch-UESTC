# mytorch/dataset_analyzer.py
"""
数据集分析工具
提供数据均衡性分析、增强建议等功能
"""
import numpy as np
import os
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

from mytorch.dataset import Dataset
from mytorch.tensor import Tensor


class DatasetAnalyzer:
    """
    数据集分析器
    分析数据分布、类别均衡性、提出增强建议
    """

    def __init__(self, dataset: Dataset, name: str = "Dataset"):
        """
        初始化分析器

        Args:
            dataset: 要分析的数据集
            name: 数据集名称
        """
        self.dataset = dataset
        self.name = name
        self.analysis_results = {}

    def analyze_class_distribution(self, sample_size: Optional[int] = None) -> Dict:
        """
        分析类别分布

        Args:
            sample_size: 采样数量（None表示全部）

        Returns:
            类别分布统计
        """
        print(f"\n[分析] {self.name} - 类别分布")
        print("=" * 50)

        # 收集标签
        n_total = len(self.dataset)
        sample_indices = self._get_sample_indices(sample_size, n_total)

        labels = []
        for idx in sample_indices:
            _, label = self.dataset[idx]
            if isinstance(label, Tensor):
                label_val = label.data.item() if hasattr(label.data, 'item') else label.data
            else:
                label_val = label
            labels.append(int(label_val))

        # 统计
        counter = Counter(labels)
        n_samples = len(labels)

        results = {
            'total_samples': n_total,
            'sampled_samples': n_samples,
            'num_classes': len(counter),
            'class_counts': dict(counter),
            'class_percentages': {k: v / n_samples * 100 for k, v in counter.items()},
            'imbalance_ratio': max(counter.values()) / (min(counter.values()) + 1e-8)
        }

        # 打印结果
        print(f"总样本数: {n_total}")
        print(f"采样数: {n_samples}")
        print(f"类别数: {results['num_classes']}")
        print(f"\n类别分布:")

        for class_id in sorted(counter.keys()):
            count = counter[class_id]
            pct = count / n_samples * 100
            bar = "█" * int(pct / 2)
            print(f"  类别 {class_id}: {count:6d} 样本 ({pct:5.2f}%) {bar}")

        print(f"\n最大/最小类别比例: {results['imbalance_ratio']:.2f}")

        # 均衡性评估
        if results['imbalance_ratio'] < 1.5:
            print("✓ 数据分布较为均衡")
            results['balance_level'] = 'good'
        elif results['imbalance_ratio'] < 3.0:
            print("⚠ 数据存在轻微不均衡")
            results['balance_level'] = 'moderate'
        else:
            print("✗ 数据严重不均衡，建议进行重采样或加权损失")
            results['balance_level'] = 'severe'

        self.analysis_results['class_distribution'] = results
        return results

    def analyze_feature_statistics(self, sample_size: int = 1000) -> Dict:
        """
        分析特征统计信息

        Args:
            sample_size: 采样数量

        Returns:
            特征统计信息
        """
        print(f"\n[分析] {self.name} - 特征统计")
        print("=" * 50)

        n_total = len(self.dataset)
        sample_indices = self._get_sample_indices(sample_size, n_total)

        # 收集特征
        features = []
        for idx in sample_indices:
            data, _ = self.dataset[idx]
            if isinstance(data, Tensor):
                data_np = data.data
            else:
                data_np = np.array(data)
            features.append(data_np.flatten())

        features = np.array(features)

        results = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'global_mean': float(np.mean(features)),
            'global_std': float(np.std(features))
        }

        print(f"全局均值: {results['global_mean']:.4f}")
        print(f"全局标准差: {results['global_std']:.4f}")
        print(f"数值范围: [{results['min'].min():.4f}, {results['max'].max():.4f}]")

        self.analysis_results['feature_statistics'] = results
        return results

    def suggest_augmentation(self) -> List[Dict]:
        """
        根据数据分析结果提出数据增强建议

        Returns:
            增强建议列表
        """
        print(f"\n[建议] {self.name} - 数据增强建议")
        print("=" * 50)

        suggestions = []

        # 检查类别均衡性
        if 'class_distribution' in self.analysis_results:
            balance = self.analysis_results['class_distribution']['balance_level']

            if balance == 'severe':
                suggestions.append({
                    'type': 'class_balancing',
                    'severity': 'high',
                    'description': '数据严重不均衡，建议使用以下方法：',
                    'methods': [
                        '1. 对少数类进行过采样（重复样本）',
                        '2. 使用加权损失函数（class weights）',
                        '3. 应用更丰富的数据增强到少数类',
                        '4. 考虑使用Focal Loss'
                    ]
                })
            elif balance == 'moderate':
                suggestions.append({
                    'type': 'class_balancing',
                    'severity': 'medium',
                    'description': '数据存在轻微不均衡，建议：',
                    'methods': [
                        '1. 使用加权损失函数',
                        '2. 对少数类增加轻微增强'
                    ]
                })

        # 通用增强建议
        suggestions.append({
            'type': 'general',
            'severity': 'low',
            'description': '通用数据增强建议：',
            'methods': [
                '随机水平翻转 (RandomHorizontalFlip)',
                '随机旋转 (RandomRotation, ±15°)',
                '随机亮度/对比度调整 (ColorJitter)',
                '随机裁剪 (RandomCrop)',
                '高斯噪声添加'
            ]
        })

        # 打印建议
        for i, sugg in enumerate(suggestions):
            print(f"\n建议 {i + 1}: {sugg['description']}")
            for method in sugg['methods']:
                print(f"  {method}")

        return suggestions

    def suggest_normalization(self) -> Dict:
        """
        根据特征统计建议归一化参数

        Returns:
            归一化参数建议
        """
        print(f"\n[建议] {self.name} - 归一化参数")
        print("=" * 50)

        if 'feature_statistics' not in self.analysis_results:
            self.analyze_feature_statistics()

        stats = self.analysis_results['feature_statistics']

        # 如果特征是多维的，可能需要逐通道计算
        # 这里简化处理
        suggestions = {
            'mean': [float(stats['global_mean'])],
            'std': [float(stats['global_std'])]
        }

        print(f"建议的归一化参数:")
        print(f"  mean = {suggestions['mean']}")
        print(f"  std  = {suggestions['std']}")
        print(f"\n使用方式:")
        print(f"  transform = Compose([")
        print(f"      ToTensor(),")
        print(f"      Normalize(mean={suggestions['mean']}, std={suggestions['std']})")
        print(f"  ])")

        return suggestions

    def generate_report(self, output_dir: str = "analysis_reports"):
        """
        生成完整分析报告

        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)

        # 运行所有分析
        class_dist = self.analyze_class_distribution()
        feat_stats = self.analyze_feature_statistics()
        aug_suggest = self.suggest_augmentation()
        norm_params = self.suggest_normalization()

        # 保存报告
        import json
        report = {
            'dataset_name': self.name,
            'class_distribution': class_dist,
            'feature_statistics': {
                'global_mean': feat_stats['global_mean'],
                'global_std': feat_stats['global_std']
            },
            'augmentation_suggestions': aug_suggest,
            'normalization_params': norm_params
        }

        report_path = os.path.join(output_dir, f"{self.name}_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n分析报告已保存至: {report_path}")

        return report

    def _get_sample_indices(self, sample_size: Optional[int], total_size: int) -> List[int]:
        """获取采样索引"""
        if sample_size is None or sample_size >= total_size:
            return list(range(total_size))
        else:
            return np.random.choice(total_size, sample_size, replace=False).tolist()