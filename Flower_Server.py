# fl_server.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Dict, Optional
import numpy as np
import json
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """自定义加权平均指标计算"""
    # 按样本数量加权平均
    mses = [num_examples * m["mse"] for num_examples, m in metrics]
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    r2s = [num_examples * m["r2"] for num_examples, m in metrics]
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    # 收集各个client的指标
    client_metrics = {}
    for num_examples, m in metrics:
        client_id = m.get("location_id", "unknown")
        client_metrics[client_id] = {
            "mse": m["mse"],
            "mae": m["mae"], 
            "r2": m["r2"],
            "samples": num_examples
        }
    
    return {
        "mse": sum(mses) / total_examples,
        "mae": sum(maes) / total_examples, 
        "r2": sum(r2s) / total_examples,
        "client_metrics": client_metrics  # 保存各个client的详细指标
    }

class SaveMetricsStrategy(FedAvg):
    """自定义策略，保存训练指标"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            'round': [],
            'mse': [], 
            'mae': [],
            'r2': [],
            'client_metrics': []
        }
    
    def aggregate_evaluate(self, server_round, results, failures):
        """聚合评估结果并保存"""
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated_metrics is not None and len(results) > 0:
            # 计算加权指标
            weighted_metrics = weighted_average([(r.num_examples, r.metrics) for r in results])
            
            # 保存历史
            self.history['round'].append(server_round)
            self.history['mse'].append(weighted_metrics['mse'])
            self.history['mae'].append(weighted_metrics['mae'])
            self.history['r2'].append(weighted_metrics['r2'])
            self.history['client_metrics'].append(weighted_metrics['client_metrics'])
            
            print(f"\n=== Round {server_round} Results ===")
            print(f"Global MSE: {weighted_metrics['mse']:.4f}")
            print(f"Global MAE: {weighted_metrics['mae']:.4f}")
            print(f"Global R²: {weighted_metrics['r2']:.4f}")
            
            # 保存指标到文件
            with open('federated_metrics.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # 每5轮绘制一次图表
            if server_round % 5 == 0:
                self.plot_training_history()
        
        return aggregated_metrics
    
    def plot_training_history(self):
        """绘制训练历史图表"""
        plt.figure(figsize=(12, 8))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['round'], self.history['mse'], 'b-o', label='MSE')
        plt.xlabel('Round')
        plt.ylabel('MSE')
        plt.title('Federated Learning - MSE')
        plt.grid(True, alpha=0.3)
        
        # MAE曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.history['round'], self.history['mae'], 'r-o', label='MAE')
        plt.xlabel('Round')
        plt.ylabel('MAE')
        plt.title('Federated Learning - MAE')
        plt.grid(True, alpha=0.3)
        
        # R²曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.history['round'], self.history['r2'], 'g-o', label='R²')
        plt.xlabel('Round')
        plt.ylabel('R²')
        plt.title('Federated Learning - R² Score')
        plt.grid(True, alpha=0.3)
        
        # 各个client的最终MAE
        plt.subplot(2, 2, 4)
        if self.history['client_metrics']:
            last_round_metrics = self.history['client_metrics'][-1]
            client_ids = list(last_round_metrics.keys())
            client_maes = [last_round_metrics[cid]['mae'] for cid in client_ids]
            
            plt.bar(range(len(client_ids)), client_maes, alpha=0.7)
            plt.xticks(range(len(client_ids)), client_ids, rotation=45)
            plt.xlabel('Client ID')
            plt.ylabel('MAE')
            plt.title('Client-wise MAE (Latest Round)')
        
        plt.tight_layout()
        plt.savefig('federated_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # 配置联邦学习策略
    strategy = SaveMetricsStrategy(
        fraction_fit=1.0,      # 100%的client参与训练
        fraction_evaluate=1.0, # 100%的client参与评估
        min_fit_clients=2,     # 最少2个client才开始训练
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # 启动服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),  # 更多训练轮次
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,  # 1GB最大消息长度
    )