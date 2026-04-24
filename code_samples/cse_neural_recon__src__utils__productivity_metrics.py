"""
DGX Spark Productivity Metrics Collection System

Collects quantitative metrics demonstrating the value of DGX Spark
for neural 3D reconstruction research productivity.

Key Metric Categories:
1. Training Throughput - samples/second, epochs/hour
2. GPU Utilization - compute %, memory usage, power efficiency  
3. Memory Capacity - batch sizes enabled, model sizes possible
4. Experiment Iteration Speed - time to result, experiments/day
5. Cost Efficiency - samples per watt, TFLOPS utilization
"""

import os
import sys
import time
import json
import threading
import subprocess
from datetime import datetime

import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from collections import deque

import torch
import numpy as np


@dataclass
class GPUSnapshot:
    """Single GPU metrics snapshot."""
    timestamp: float
    gpu_utilization: float  # %
    memory_used: float  # MB
    memory_total: float  # MB
    memory_utilization: float  # %
    power_draw: float  # Watts
    temperature: float  # Celsius
    sm_clock: float  # MHz
    memory_clock: float  # MHz


@dataclass  
class TrainingSnapshot:
    """Training metrics at a point in time."""
    timestamp: float
    epoch: int
    batch_idx: int
    samples_processed: int
    loss: float
    batch_time: float  # seconds
    data_load_time: float  # seconds
    forward_time: float  # seconds
    backward_time: float  # seconds


@dataclass
class ProductivityReport:
    """Complete productivity metrics report."""
    
    # System Info
    gpu_name: str = ""
    gpu_architecture: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    pytorch_version: str = ""
    
    # Training Configuration
    model_name: str = ""
    model_parameters: int = 0
    batch_size: int = 0
    dataset_size: int = 0
    
    # Throughput Metrics
    total_training_time: float = 0.0  # seconds
    total_epochs: int = 0
    total_samples_processed: int = 0
    samples_per_second: float = 0.0
    batches_per_second: float = 0.0
    epochs_per_hour: float = 0.0
    
    # GPU Utilization (averages)
    avg_gpu_utilization: float = 0.0
    peak_gpu_utilization: float = 0.0
    avg_memory_used: float = 0.0  # MB
    peak_memory_used: float = 0.0  # MB
    memory_utilization: float = 0.0  # % of total
    avg_power_draw: float = 0.0  # Watts
    peak_power_draw: float = 0.0  # Watts
    
    # Efficiency Metrics
    samples_per_watt: float = 0.0
    samples_per_gb_memory: float = 0.0
    gpu_efficiency: float = 0.0  # actual util / theoretical max
    
    # Memory Capacity Benefits
    max_batch_size_tested: int = 0
    memory_headroom: float = 0.0  # MB available
    model_fits_in_memory: bool = True
    
    # Experiment Iteration Speed
    time_to_first_result: float = 0.0  # seconds to first viz
    checkpoint_save_time: float = 0.0  # seconds
    visualization_gen_time: float = 0.0  # seconds
    estimated_experiments_per_day: float = 0.0
    
    # Quality Metrics (if available)
    final_loss: float = 0.0
    final_chamfer_distance: float = 0.0
    final_accuracy_5cm: float = 0.0
    
    # Comparison Baselines (for context)
    theoretical_peak_tflops: float = 0.0
    achieved_tflops_estimate: float = 0.0
    
    # Timestamps
    experiment_start: str = ""
    experiment_end: str = ""
    report_generated: str = ""


class GPUMetricsCollector:
    """
    Background collector for GPU metrics using nvidia-smi.
    
    Collects metrics at regular intervals without impacting training.
    """
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: List[GPUSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start background collection."""
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop background collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _collect_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                snapshot = self._get_gpu_snapshot()
                if snapshot:
                    self.snapshots.append(snapshot)
            except Exception:
                pass  # Don't crash on collection errors
            time.sleep(self.interval)
            
    def _get_gpu_snapshot(self) -> Optional[GPUSnapshot]:
        """Query nvidia-smi for current GPU state, with PyTorch fallback for memory."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,'
                 'power.draw,temperature.gpu,clocks.sm,clocks.mem',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return None
                
            parts = result.stdout.strip().split(',')
            if len(parts) < 7:
                return None
                
            # Parse values, handling [N/A] gracefully
            def parse_float(s, default=0.0):
                s = s.strip()
                if s == '[N/A]' or s == 'N/A' or not s:
                    return default
                try:
                    return float(s)
                except ValueError:
                    return default
            
            memory_used = parse_float(parts[1])
            memory_total = parse_float(parts[2], 1.0)  # Avoid div by zero
            
            # If nvidia-smi returns 0 for memory, try PyTorch's memory tracking
            if memory_used == 0 and torch.cuda.is_available():
                try:
                    memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
                    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                except Exception:
                    pass
            
            return GPUSnapshot(
                timestamp=time.time(),
                gpu_utilization=parse_float(parts[0]),
                memory_used=memory_used,
                memory_total=memory_total,
                memory_utilization=100 * memory_used / memory_total if memory_total > 0 else 0,
                power_draw=parse_float(parts[3]),
                temperature=parse_float(parts[4]),
                sm_clock=parse_float(parts[5]),
                memory_clock=parse_float(parts[6]),
            )
        except Exception:
            return None
            
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from collected snapshots."""
        if not self.snapshots:
            return {}
            
        utils = [s.gpu_utilization for s in self.snapshots]
        mems = [s.memory_used for s in self.snapshots]
        powers = [s.power_draw for s in self.snapshots if s.power_draw > 0]
        temps = [s.temperature for s in self.snapshots if s.temperature > 0]
        
        return {
            'avg_gpu_utilization': np.mean(utils) if utils else 0,
            'peak_gpu_utilization': np.max(utils) if utils else 0,
            'min_gpu_utilization': np.min(utils) if utils else 0,
            'avg_memory_used': np.mean(mems) if mems else 0,
            'peak_memory_used': np.max(mems) if mems else 0,
            'memory_total': self.snapshots[0].memory_total if self.snapshots else 0,
            'avg_power_draw': np.mean(powers) if powers else 0,
            'peak_power_draw': np.max(powers) if powers else 0,
            'avg_temperature': np.mean(temps) if temps else 0,
            'num_samples': len(self.snapshots),
        }


class TrainingProfiler:
    """
    Profiles training loop timing breakdown.
    
    Measures:
    - Data loading time
    - Forward pass time
    - Backward pass time
    - Optimizer step time
    """
    
    def __init__(self):
        self.snapshots: List[TrainingSnapshot] = []
        self._batch_start: float = 0
        self._phase_times: Dict[str, float] = {}
        self._current_phase: str = ""
        self._phase_start: float = 0
        
    def start_batch(self):
        """Mark start of a batch."""
        self._batch_start = time.time()
        self._phase_times = {}
        
    def start_phase(self, phase: str):
        """Start timing a phase (data_load, forward, backward, optimizer)."""
        if self._current_phase:
            self.end_phase()
        self._current_phase = phase
        self._phase_start = time.time()
        
    def end_phase(self):
        """End current phase timing."""
        if self._current_phase:
            self._phase_times[self._current_phase] = time.time() - self._phase_start
            self._current_phase = ""
            
    def end_batch(self, epoch: int, batch_idx: int, samples: int, loss: float):
        """Record completed batch."""
        if self._current_phase:
            self.end_phase()
            
        batch_time = time.time() - self._batch_start
        
        snapshot = TrainingSnapshot(
            timestamp=time.time(),
            epoch=epoch,
            batch_idx=batch_idx,
            samples_processed=samples,
            loss=loss,
            batch_time=batch_time,
            data_load_time=self._phase_times.get('data_load', 0),
            forward_time=self._phase_times.get('forward', 0),
            backward_time=self._phase_times.get('backward', 0),
        )
        self.snapshots.append(snapshot)
        
    def get_summary(self) -> Dict[str, float]:
        """Get timing summary statistics."""
        if not self.snapshots:
            return {}
            
        batch_times = [s.batch_time for s in self.snapshots]
        data_times = [s.data_load_time for s in self.snapshots]
        forward_times = [s.forward_time for s in self.snapshots]
        backward_times = [s.backward_time for s in self.snapshots]
        
        total_samples = sum(s.samples_processed for s in self.snapshots)
        total_time = sum(batch_times)
        
        return {
            'total_batches': len(self.snapshots),
            'total_samples': total_samples,
            'total_time': total_time,
            'samples_per_second': total_samples / total_time if total_time > 0 else 0,
            'avg_batch_time': np.mean(batch_times),
            'avg_data_load_time': np.mean(data_times) if data_times else 0,
            'avg_forward_time': np.mean(forward_times) if forward_times else 0,
            'avg_backward_time': np.mean(backward_times) if backward_times else 0,
            'data_load_fraction': sum(data_times) / total_time if total_time > 0 else 0,
            'compute_fraction': (sum(forward_times) + sum(backward_times)) / total_time if total_time > 0 else 0,
        }


class ProductivityMetricsCollector:
    """
    Main productivity metrics collection and reporting system.
    
    Collects detailed training metrics for cross-device comparison.
    Enabled by default to support benchmarking across different hardware.
    
    Usage:
        collector = ProductivityMetricsCollector(output_dir='output/metrics')
        collector.start_experiment(model, config)
        
        for epoch in range(epochs):
            for batch in dataloader:
                collector.start_batch()
                # ... training ...
                collector.end_batch(epoch, batch_idx, batch_size, loss)
                
        collector.end_experiment()
        report = collector.generate_report()
    """
    
    def __init__(
        self, 
        output_dir: str = 'output/metrics', 
        gpu_sample_interval: float = 1.0,
        sample_rate: int = 10  # Record batch timing every N batches
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.gpu_collector = GPUMetricsCollector(interval=gpu_sample_interval)
        self.training_profiler = TrainingProfiler()
        
        self.experiment_start: Optional[float] = None
        self.experiment_end: Optional[float] = None
        
        self.model_info: Dict[str, Any] = {}
        self.config_info: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, float] = {}
        self.timing_events: Dict[str, float] = {}
        
        # Device identification for cross-device comparison
        self.device_info: Dict[str, Any] = self._collect_device_info()
        
        self._current_epoch = 0
        self._epoch_start_time = 0
        self._epoch_times: List[float] = []
        self._batch_count = 0
        
    def _collect_device_info(self) -> Dict[str, Any]:
        """Collect comprehensive device information for comparison."""
        info = {
            'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown',
            'platform': os.uname().sysname if hasattr(os, 'uname') else 'unknown',
            'python_version': sys.version.split()[0],
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update({
                'gpu_name': props.name,
                'gpu_compute_capability': f"{props.major}.{props.minor}",
                'gpu_total_memory_gb': props.total_memory / (1024**3),
                'gpu_multiprocessor_count': props.multi_processor_count,
                'cuda_version': torch.version.cuda or 'unknown',
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A',
            })
            
            # Try to get driver version
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    info['driver_version'] = result.stdout.strip()
            except Exception:
                pass
                
        return info
        
    def start_experiment(
        self, 
        model: torch.nn.Module,
        config: Optional[Dict] = None,
        model_name: str = "NeuralSDF"
    ):
        """Start metrics collection for an experiment."""
        self.experiment_start = time.time()
        
        # Collect model info
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model_info = {
            'name': model_name,
            'total_parameters': num_params,
            'trainable_parameters': trainable_params,
            'model_class': model.__class__.__name__,
        }
        
        # Collect config info
        if config:
            self.config_info = {
                'batch_size': config.get('training', {}).get('batch_size', 0),
                'learning_rate': config.get('training', {}).get('learning_rate', 0),
                'epochs': config.get('training', {}).get('epochs', 0),
            }
            
        # Start GPU monitoring
        self.gpu_collector.start()
        
        print(f"[ProductivityMetrics] Started experiment tracking")
        print(f"  Model: {model_name} ({num_params:,} parameters)")
        
    def start_epoch(self, epoch: int):
        """Mark start of an epoch."""
        self._current_epoch = epoch
        self._epoch_start_time = time.time()
        
    def end_epoch(self, epoch: int):
        """Mark end of an epoch."""
        epoch_time = time.time() - self._epoch_start_time
        self._epoch_times.append(epoch_time)
        
    def start_batch(self):
        """Mark start of a training batch."""
        self.training_profiler.start_batch()
        
    def start_phase(self, phase: str):
        """Start timing a training phase."""
        self.training_profiler.start_phase(phase)
        
    def end_batch(self, epoch: int, batch_idx: int, batch_size: int, loss: float):
        """Record completed batch."""
        self.training_profiler.end_batch(epoch, batch_idx, batch_size, loss)
        
    def record_event(self, event_name: str, duration: float):
        """Record a timed event (checkpoint save, visualization, etc.)."""
        if event_name not in self.timing_events:
            self.timing_events[event_name] = []
        self.timing_events[event_name].append(duration)
        
    def record_quality_metric(self, name: str, value: float):
        """Record a quality metric (loss, chamfer distance, etc.)."""
        self.quality_metrics[name] = value
        
    def end_experiment(self):
        """End metrics collection."""
        self.experiment_end = time.time()
        self.gpu_collector.stop()
        
        print(f"[ProductivityMetrics] Experiment completed")
        total_time = self.experiment_end - self.experiment_start
        print(f"  Total time: {total_time/3600:.2f} hours")
        
    def generate_report(self) -> ProductivityReport:
        """Generate comprehensive productivity report."""
        report = ProductivityReport()
        
        # System info
        report.gpu_name = self._get_gpu_name()
        report.gpu_architecture = "Blackwell"  # DGX Spark
        report.cuda_version = torch.version.cuda or "Unknown"
        report.pytorch_version = torch.__version__
        report.driver_version = self._get_driver_version()
        
        # Model info
        report.model_name = self.model_info.get('name', '')
        report.model_parameters = self.model_info.get('total_parameters', 0)
        report.batch_size = self.config_info.get('batch_size', 0)
        
        # Training throughput
        training_summary = self.training_profiler.get_summary()
        if self.experiment_start and self.experiment_end:
            report.total_training_time = self.experiment_end - self.experiment_start
        report.total_epochs = len(self._epoch_times)
        report.total_samples_processed = training_summary.get('total_samples', 0)
        report.samples_per_second = training_summary.get('samples_per_second', 0)
        
        if report.total_training_time > 0:
            report.batches_per_second = training_summary.get('total_batches', 0) / report.total_training_time
            report.epochs_per_hour = report.total_epochs / (report.total_training_time / 3600)
            
        # GPU utilization
        gpu_summary = self.gpu_collector.get_summary()
        report.avg_gpu_utilization = gpu_summary.get('avg_gpu_utilization', 0)
        report.peak_gpu_utilization = gpu_summary.get('peak_gpu_utilization', 0)
        report.avg_memory_used = gpu_summary.get('avg_memory_used', 0)
        report.peak_memory_used = gpu_summary.get('peak_memory_used', 0)
        memory_total = gpu_summary.get('memory_total', 1)
        report.memory_utilization = 100 * report.peak_memory_used / memory_total if memory_total > 0 else 0
        report.avg_power_draw = gpu_summary.get('avg_power_draw', 0)
        report.peak_power_draw = gpu_summary.get('peak_power_draw', 0)
        
        # Efficiency metrics
        if report.avg_power_draw > 0 and report.total_training_time > 0:
            total_energy_wh = report.avg_power_draw * (report.total_training_time / 3600)
            report.samples_per_watt = report.total_samples_processed / total_energy_wh if total_energy_wh > 0 else 0
            
        if report.peak_memory_used > 0:
            report.samples_per_gb_memory = report.total_samples_processed / (report.peak_memory_used / 1024)
            
        report.memory_headroom = memory_total - report.peak_memory_used
        report.max_batch_size_tested = report.batch_size
        
        # Timing events
        if 'checkpoint_save' in self.timing_events:
            report.checkpoint_save_time = np.mean(self.timing_events['checkpoint_save'])
        if 'visualization' in self.timing_events:
            report.visualization_gen_time = np.mean(self.timing_events['visualization'])
        if 'first_result' in self.timing_events:
            report.time_to_first_result = self.timing_events['first_result'][0]
            
        # Estimate experiments per day
        if report.total_training_time > 0 and report.total_epochs > 0:
            time_per_experiment = report.total_training_time  # This experiment's duration
            report.estimated_experiments_per_day = 86400 / time_per_experiment
            
        # Quality metrics
        report.final_loss = self.quality_metrics.get('loss', 0)
        report.final_chamfer_distance = self.quality_metrics.get('chamfer_distance', 0)
        report.final_accuracy_5cm = self.quality_metrics.get('accuracy_5cm', 0)
        
        # Timestamps
        if self.experiment_start:
            report.experiment_start = datetime.fromtimestamp(self.experiment_start).isoformat()
        if self.experiment_end:
            report.experiment_end = datetime.fromtimestamp(self.experiment_end).isoformat()
        report.report_generated = datetime.now().isoformat()
        
        return report
        
    def save_report(self, report: Optional[ProductivityReport] = None, filename: str = "productivity_report"):
        """Save report to JSON and generate formatted text report."""
        if report is None:
            report = self.generate_report()
            
        # Create a comprehensive report dict with device info
        report_dict = asdict(report)
        report_dict['device_info'] = self.device_info
        report_dict['config_info'] = self.config_info
        report_dict['model_info'] = self.model_info
            
        # Save JSON
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save a comparison-friendly summary (single row of key metrics)
        comparison_path = self.output_dir / f"{filename}_comparison.json"
        comparison_data = {
            'device_id': self.device_info.get('hostname', 'unknown'),
            'gpu_name': report.gpu_name,
            'gpu_memory_gb': self.device_info.get('gpu_total_memory_gb', 0),
            'cuda_version': report.cuda_version,
            'pytorch_version': report.pytorch_version,
            'model_name': report.model_name,
            'model_parameters': report.model_parameters,
            'batch_size': report.batch_size,
            'epochs': report.total_epochs,
            'samples_per_second': round(report.samples_per_second, 2),
            'epochs_per_hour': round(report.epochs_per_hour, 2),
            'avg_gpu_utilization': round(report.avg_gpu_utilization, 1),
            'peak_memory_mb': round(report.peak_memory_used, 0),
            'avg_power_watts': round(report.avg_power_draw, 1),
            'samples_per_watt': round(report.samples_per_watt, 1),
            'final_loss': report.final_loss,
            'timestamp': report.report_generated,
        }
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
            
        # Generate formatted text report
        text_report = self._format_report(report)
        text_path = self.output_dir / f"{filename}.txt"
        with open(text_path, 'w') as f:
            f.write(text_report)
            
        # Save raw data for further analysis
        raw_data = {
            'gpu_snapshots': [asdict(s) for s in self.gpu_collector.snapshots],
            'training_snapshots': [asdict(s) for s in self.training_profiler.snapshots],
            'epoch_times': self._epoch_times,
            'timing_events': self.timing_events,
        }
        raw_path = self.output_dir / f"{filename}_raw.json"
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2)
            
        print(f"[ProductivityMetrics] Reports saved to {self.output_dir}")
        return json_path, text_path
        
    def _format_report(self, report: ProductivityReport) -> str:
        """Format report as readable text."""
        lines = [
            "=" * 80,
            "DGX SPARK PRODUCTIVITY METRICS REPORT",
            "Neural 3D Reconstruction Research",
            "=" * 80,
            "",
            f"Generated: {report.report_generated}",
            f"Experiment Duration: {report.experiment_start} to {report.experiment_end}",
            "",
            "-" * 80,
            "SYSTEM CONFIGURATION",
            "-" * 80,
            f"  GPU: {report.gpu_name}",
            f"  Architecture: {report.gpu_architecture}",
            f"  CUDA Version: {report.cuda_version}",
            f"  PyTorch Version: {report.pytorch_version}",
            f"  Driver Version: {report.driver_version}",
            "",
            "-" * 80,
            "MODEL CONFIGURATION", 
            "-" * 80,
            f"  Model: {report.model_name}",
            f"  Parameters: {report.model_parameters:,}",
            f"  Batch Size: {report.batch_size}",
            "",
            "-" * 80,
            "TRAINING THROUGHPUT (Key Productivity Metrics)",
            "-" * 80,
            f"  Total Training Time: {report.total_training_time/3600:.2f} hours ({report.total_training_time:.0f} seconds)",
            f"  Total Epochs: {report.total_epochs}",
            f"  Total Samples Processed: {report.total_samples_processed:,}",
            "",
            f"  >>> Samples/Second: {report.samples_per_second:.1f}",
            f"  >>> Batches/Second: {report.batches_per_second:.2f}",
            f"  >>> Epochs/Hour: {report.epochs_per_hour:.2f}",
            "",
            "-" * 80,
            "GPU UTILIZATION",
            "-" * 80,
            f"  Average GPU Utilization: {report.avg_gpu_utilization:.1f}%",
            f"  Peak GPU Utilization: {report.peak_gpu_utilization:.1f}%",
            f"  Average Memory Used: {report.avg_memory_used:.0f} MB",
            f"  Peak Memory Used: {report.peak_memory_used:.0f} MB",
            f"  Memory Utilization: {report.memory_utilization:.1f}%",
            f"  Memory Headroom: {report.memory_headroom:.0f} MB available",
            "",
            f"  Average Power Draw: {report.avg_power_draw:.1f} W",
            f"  Peak Power Draw: {report.peak_power_draw:.1f} W",
            "",
            "-" * 80,
            "EFFICIENCY METRICS",
            "-" * 80,
            f"  Samples per Watt-hour: {report.samples_per_watt:.1f}",
            f"  Samples per GB Memory: {report.samples_per_gb_memory:.1f}",
            "",
            "-" * 80,
            "EXPERIMENT ITERATION SPEED",
            "-" * 80,
            f"  Time to First Result: {report.time_to_first_result:.1f} seconds",
            f"  Checkpoint Save Time: {report.checkpoint_save_time:.2f} seconds",
            f"  Visualization Generation: {report.visualization_gen_time:.2f} seconds",
            "",
            f"  >>> Estimated Experiments/Day: {report.estimated_experiments_per_day:.1f}",
            "",
            "-" * 80,
            "QUALITY METRICS (Final)",
            "-" * 80,
            f"  Final Loss: {report.final_loss:.6f}",
            f"  Chamfer Distance: {report.final_chamfer_distance:.4f}",
            f"  Accuracy @5cm: {report.final_accuracy_5cm*100:.2f}%",
            "",
            "=" * 80,
            "PRODUCTIVITY SUMMARY",
            "=" * 80,
            "",
            "The DGX Spark enables:",
            f"  • Processing {report.samples_per_second:.0f} training samples per second",
            f"  • Completing {report.epochs_per_hour:.1f} training epochs per hour",
            f"  • Running approximately {report.estimated_experiments_per_day:.0f} full experiments per day",
            f"  • Training {report.model_parameters/1e6:.1f}M parameter models with {report.memory_headroom/1024:.1f} GB memory headroom",
            "",
            "This translates to significantly faster research iteration compared to",
            "consumer-grade GPUs, enabling more rapid hypothesis testing and model development.",
            "",
            "=" * 80,
        ]
        
        return "\n".join(lines)
        
    def _get_gpu_name(self) -> str:
        """Get GPU name from nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return "Unknown GPU"
            
    def _get_driver_version(self) -> str:
        """Get driver version."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return "Unknown"


def create_comparison_baseline() -> Dict[str, Any]:
    """
    Create baseline comparison data for common GPU configurations.
    
    These are approximate values for context - actual performance varies.
    """
    return {
        'rtx_3080': {
            'name': 'NVIDIA RTX 3080',
            'memory_gb': 10,
            'typical_samples_per_second': 800,  # Approximate for similar workload
            'typical_power_watts': 320,
        },
        'rtx_4090': {
            'name': 'NVIDIA RTX 4090', 
            'memory_gb': 24,
            'typical_samples_per_second': 2000,
            'typical_power_watts': 450,
        },
        'a100_40gb': {
            'name': 'NVIDIA A100 40GB',
            'memory_gb': 40,
            'typical_samples_per_second': 3500,
            'typical_power_watts': 400,
        },
        'v100_16gb': {
            'name': 'NVIDIA V100 16GB',
            'memory_gb': 16,
            'typical_samples_per_second': 1200,
            'typical_power_watts': 300,
        },
    }
