import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from models import (GATModel, GCNModel, BasicGNN, CNNModel,
                    LightweightDGCNN, AttentionDGCNN, HighwayDGCNN, GlueEdgeDGCNN)
import time
import statistics
from graphCons import UAVDataset
from configure import Config


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, min_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.epoch = 0

    def __call__(self, val_loss):
        self.epoch += 1
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.epoch > self.min_epochs:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def measure_latency(model, data_loader, num_warmup=5, num_repeats=20):
    model = model.to(Config.device)
    model.eval()
    batch = next(iter(data_loader))
    samples = batch.to_data_list()
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(batch)
    latencies = []
    for _ in range(num_repeats):
        for sample in samples:
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(sample)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
    return {'mean': statistics.mean(latencies)}

def evaluate_model(model, train_loader, val_loader, test_loader, model_name):
    model = model.to(Config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, min_epochs=20)

    metrics = {
        'train_acc': [], 'train_loss': [],
        'val_acc': [], 'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_acc = 0
    best_metrics = {}
    best_model_state = None

    for epoch in range(1, Config.epochs + 1):
        model.train()
        train_correct = train_total = train_loss = 0
        for data in train_loader:
            data = data.to(Config.device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            train_correct += pred.eq(data.y).sum().item()
            train_total += data.y.size(0)
            train_loss += loss.item() * data.y.size(0)

        train_acc = train_correct / train_total
        metrics['train_acc'].append(train_acc)
        metrics['train_loss'].append(train_loss / train_total)

        model.eval()
        val_correct = val_total = val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(Config.device)
                out = model(data)
                val_loss += F.nll_loss(out, data.y, reduction='sum').item()
                pred = out.argmax(dim=1)
                val_correct += pred.eq(data.y).sum().item()
                val_total += data.y.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(data.y.cpu().numpy())

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(avg_val_loss)
        metrics['val_precision'].append(precision_score(val_targets, val_preds, average='weighted', zero_division=0))
        metrics['val_recall'].append(recall_score(val_targets, val_preds, average='weighted', zero_division=0))
        metrics['val_f1'].append(f1_score(val_targets, val_preds, average='weighted', zero_division=0))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = {
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss / train_total,
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'val_precision': metrics['val_precision'][-1],
                'val_recall': metrics['val_recall'][-1],
                'val_f1': metrics['val_f1'][-1],
            }
            best_model_state = model.state_dict()

        print(f'\n{model_name} - Epoch {epoch}:')
        print(f'Train Loss: {metrics["train_loss"][-1]:.4f} | Acc: {metrics["train_acc"][-1]:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f}')

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    test_correct = test_total = test_loss = 0
    test_preds, test_targets = [], []
    route_stats = {}

    with torch.no_grad():
        for data in test_loader:
            data = data.to(Config.device)
            out = model(data)
            test_loss += F.nll_loss(out, data.y, reduction='sum').item()
            pred = out.argmax(dim=1)
            test_correct += pred.eq(data.y).sum().item()
            test_total += data.y.size(0)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(data.y.cpu().numpy())

            batch_routes = data.route_id.cpu().numpy()
            batch_labels = data.y.cpu().numpy()
            batch_correct = pred.eq(data.y).cpu().numpy()
            for route_num, label, correct in zip(batch_routes, batch_labels, batch_correct):
                route_num = route_num.item()
                if route_num not in route_stats:
                    route_stats[route_num] = {'correct': 0, 'total': 0, 'label': label.item()}
                route_stats[route_num]['correct'] += int(correct)
                route_stats[route_num]['total'] += 1

    test_acc = test_correct / test_total
    test_precision = precision_score(test_targets, test_preds, average='binary', zero_division=0)
    test_recall = recall_score(test_targets, test_preds, average='binary', zero_division=0)
    test_f1 = f1_score(test_targets, test_preds, average='binary', zero_division=0)
    best_metrics['test_acc'] = test_acc
    best_metrics['test_f1'] = test_f1
    best_metrics['test_precision'] = test_precision
    best_metrics['test_recall'] = test_recall
    best_metrics['test_loss'] = test_loss / test_total

    print(f'\n=== Best Validation Performance at Epoch {best_metrics["epoch"]} ===')
    print(f"Train Acc: {best_metrics['train_acc']:.4f}")
    print(f"Val Acc: {best_metrics['val_acc']:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    return {
        'best_metrics': best_metrics
    }

def main():

    test_df = pd.read_csv('TEST_DATA')

    train_dataset = UAVDataset('TRAIN_DATA', train=True, test_data=test_df)
    test_dataset = UAVDataset('TEST_DATA', scaler=train_dataset.scaler, train=False)
    val_dataset = UAVDataset('VAL_DATA', scaler=train_dataset.scaler, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    print(len(test_dataset))
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    latency_subset = Subset(train_dataset, indices=range(Config.batch_size))
    latency_loader = DataLoader(latency_subset, batch_size=Config.batch_size)

    models = {
        'GAT': GATModel(len(train_dataset.features)),
        'GCN': GCNModel(len(train_dataset.features)),
        'BasicGNN': BasicGNN(len(train_dataset.features)),
        'CNN': CNNModel(len(train_dataset.features)),
        'DGCNN': LightweightDGCNN(len(train_dataset.features)),
        'AttentionDGCNN': AttentionDGCNN(len(train_dataset.features)),
        'HighwayDGCNN': HighwayDGCNN(len(train_dataset.features)),
        'GLUedgeConv': GlueEdgeDGCNN(len(train_dataset.features))
    }
    
    print("\n=== Model Latency Testing (CPU) ===")
    print(f"Testing with batch size: {Config.batch_size}")
    print("Warmup runs: 5, Measurement runs: 20\n")
    
    latency_results = {}
    for name, model in models.items():
        print(f"Measuring {name}...")
        results = measure_latency(model, latency_loader)
        latency_results[name] = results
        
        print(f"{name} Latency (ms per sample):")
        print(f"  Mean: {results['mean']:.2f} ± {results['std']:.2f}")
        print("-" * 50)
    
    results = {}
    for name, model in models.items():
        print(f"\n=== Evaluating {name} ===")
        results[name] = evaluate_model(model, train_loader, val_loader,test_loader, name)
    
    print("\n=== Final Model Comparison ===")
    print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8}".format(
        "Model", "Acc", "F1", "Prec", "Rec", "Loss"))
    print("-" * 60)

    for name, result in sorted(results.items(), 
                            key=lambda x: -x[1]['best_metrics']['test_acc']):
        metrics = result['best_metrics']
        print("{:<10} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}".format(
            name,
            metrics['test_acc'],
            metrics['test_f1'],
            metrics['test_precision'],
            metrics['test_recall'],
            metrics['test_loss']))

    print("\n=== Training Metrics ===")
    print("{:<10} {:<8} {:<8}".format("Model", "Acc", "Loss"))
    print("-" * 30)
    for name, result in results.items():
        metrics = result['best_metrics']
        print("{:<10} {:<8.4f} {:<8.4f}".format(
            name, metrics['train_acc'], metrics['train_loss']))

if __name__ == '__main__':
    main()
