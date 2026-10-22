import abc
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.optim as optim

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Log, Static, Tabs, Tab
from textual_plotext import PlotextPlot


class Module(abc.ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Metric(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        pass


class Config(abc.ABC):
    def __init__(self, path: str):
        self.path = path
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.from_dict(json.load(f))
        else:
            self.set_defaults()
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f)

    @abc.abstractmethod
    def from_dict(self, fields: dict):
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass

    @abc.abstractmethod
    def set_defaults(self):
        pass


class DatasetFetcher(abc.ABC):
    @abc.abstractmethod
    def get_train_loader(self, arg: str) -> DataLoader:
        pass

    @abc.abstractmethod
    def get_test_loader(self, arg: str) -> DataLoader:
        pass


class Accuracy(Metric):
    @property
    def name(self) -> str:
        return "Accuracy"

    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        return correct / targets.size(0)


class ErrorRate(Metric):
    @property
    def name(self) -> str:
        return "ErrorRate"

    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        preds = outputs.argmax(dim=1)
        incorrect = (preds != targets).sum().item()
        return incorrect / targets.size(0)


class Confidence(Metric):
    @property
    def name(self) -> str:
        return "Confidence"

    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        probs = torch.softmax(outputs, dim=1)
        max_probs, _ = probs.max(dim=1)
        return max_probs.mean().item()


class TrainingDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #top-bar-container {
        height: 3;
        border-bottom: heavy #313244;
        padding-left: 1;
    }
    #main-container {
        height: 72%;
        layout: horizontal;
    }
    #graph-box {
        width: 65%;
        height: 100%;
        border: none;
        padding: 1;
    }
    #stats-box {
        width: 35%;
        height: 100%;
        border-left: heavy #313244;
        padding: 2;
    }
    #log-box {
        height: 25%;
        border-top: heavy #313244;
        padding: 0 1;
    }
    Static {
        color: #cdd6f4;
    }
    Tabs {
        color: #cdd6f4;
    }
    Tab {
        padding: 0 2;
        color: #6c7086;
    }
    Tab.-active {
        color: #89b4fa;
        text-style: bold;
    }
    """

    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("s", "save_model", "Save Checkpoint"),
        ("left", "prev_metric", "Prev Graph"),
        ("right", "next_metric", "Next Graph"),
        ("up", "scroll_logs_up", "Scroll Log Up"),
        ("down", "scroll_logs_down", "Scroll Log Down"),
    ]

    COLOR_MAP = {
        "Loss": "red",
        "Accuracy": "green",
        "ErrorRate": "yellow",
        "Confidence": "cyan"
    }

    def __init__(
            self,
            model: Module,
            train_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            metrics: List[Metric],
            epochs: int
    ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.epochs = epochs
        self.step_counter = 0

        self.history: Dict[str, List[Tuple[int, float]]] = {"Loss": []}
        for m in self.metrics:
            self.history[m.name] = []

        self.track_keys = ["Loss"] + [m.name for m in self.metrics]
        self.current_graph_index = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="top-bar-container"):
            yield Tabs(*(Tab(key, id=key) for key in self.track_keys), id="metric_tabs")
        with Container(id="main-container"):
            with Container(id="graph-box"):
                yield PlotextPlot(id="metric_plot")
            with Vertical(id="stats-box"):
                yield Static("Training Monitor", id="status_title")
                yield Static("Epoch: 0 / 0", id="epoch_status")
                yield Static("Loss: 0.0000", id="loss_status")
                for metric in self.metrics:
                    yield Static(f"{metric.name}: 0.0000", id=f"metric_{metric.name}")
                yield Static("\n[Controls]\n← / →: Switch Graph\n↑ / ↓: Scroll Logs\ns: Save Checkpoint\nq: Quit",
                             id="controls_help")
        with Container(id="log-box"):
            yield Log(id="logger", highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        tabs = self.query_one("#metric_tabs", Tabs)
        tabs.active = self.track_keys[self.current_graph_index]
        self.update_active_plot()
        self.run_worker(self.train_loop, thread=True)

    def train_loop(self) -> None:
        logger = self.query_one("#logger", Log)
        epoch_lbl = self.query_one("#epoch_status", Static)

        logger.write_line("Initialization complete. Starting training loop...")

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            metric_accumulators = {m.name: 0.0 for m in self.metrics}
            batch_count = 0

            for i, (inputs, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                for m in self.metrics:
                    metric_accumulators[m.name] += m.compute(outputs, targets)
                batch_count += 1

                if i % 40 == 39:
                    self.step_counter += 1
                    avg_loss = running_loss / 40
                    avg_metrics = {k: v / batch_count for k, v in metric_accumulators.items()}

                    self.call_from_thread(self.record_and_update_ui, self.step_counter, avg_loss, avg_metrics)
                    running_loss = 0.0

            epoch_lbl.update(f"Epoch: {epoch + 1} / {self.epochs}")
            logger.write_line(f"Finished Epoch {epoch + 1}/{self.epochs}")

        logger.write_line("✨ Training session successfully finished. Press 's' to save checkpoint or 'q' to quit.")

    def record_and_update_ui(self, step: int, loss: float, metrics: dict) -> None:
        self.history["Loss"].append((step, loss))
        self.query_one("#loss_status", Static).update(f"Loss: {loss:.4f}")

        for name, val in metrics.items():
            self.history[name].append((step, val))
            self.query_one(f"#metric_{name}", Static).update(f"{name}: {val:.4f}")

        self.refresh_plot()

    def refresh_plot(self) -> None:
        active_key = self.track_keys[self.current_graph_index]
        data = self.history.get(active_key, [])

        x_data = [point[0] for point in data]
        y_data = [point[1] for point in data]

        plot_widget = self.query_one("#metric_plot", PlotextPlot)
        plt = plot_widget.plt

        plt.clf()
        plt.title(f"Trajectory: {active_key}")

        if x_data and y_data:
            line_color = self.COLOR_MAP.get(active_key, "blue")
            plt.plot(x_data, y_data, marker="braille", color=line_color)

        plot_widget.refresh()

    def update_active_plot(self) -> None:
        active_key = self.track_keys[self.current_graph_index]
        tabs = self.query_one("#metric_tabs", Tabs)
        tabs.active = active_key
        self.refresh_plot()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab and event.tab.id in self.track_keys:
            self.current_graph_index = self.track_keys.index(event.tab.id)
            self.refresh_plot()

    def action_save_model(self) -> None:
        logger = self.query_one("#logger", Log)
        path = "checkpoints/my-checkpoint.pth"
        try:
            torch.save(self.model.state_dict(), path)
            logger.write_line(f"💾 Checkpoint successfully saved to {path}")
        except Exception as e:
            logger.write_line(f"❌ Error saving checkpoint: {e}")

    def action_prev_metric(self) -> None:
        self.current_graph_index = (self.current_graph_index - 1) % len(self.track_keys)
        self.update_active_plot()

    def action_next_metric(self) -> None:
        self.current_graph_index = (self.current_graph_index + 1) % len(self.track_keys)
        self.update_active_plot()

    def action_scroll_logs_up(self) -> None:
        self.query_one("#logger", Log).scroll_up()

    def action_scroll_logs_down(self) -> None:
        self.query_one("#logger", Log).scroll_down()

    def action_quit_app(self) -> None:
        self.exit()


def run(model: Module, dataset_fetcher: DatasetFetcher, optimizer: optim.Optimizer, criterion,
        metrics: list[Metric], epochs: int, checkpoint_path: str = "checkpoints/my-checkpoint.pth"):
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python script.py [train|eval|infer] [dataset_arg]")
        return

    sub = sys.argv[1]
    dataset_arg = sys.argv[2] if len(sys.argv) > 2 else "default"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    if sub == 'train':
        trainloader = dataset_fetcher.get_train_loader(dataset_arg)

        app = TrainingDashboard(
            model=model,
            train_loader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            epochs=epochs
        )
        app.run()

    elif sub == 'eval':
        testloader = dataset_fetcher.get_test_loader(dataset_arg)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"Evaluation Results -> Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    elif sub == 'infer':
        model.eval()
        with torch.no_grad():
            sample_input = torch.randn(1, 1, 28, 28)
            output = model(sample_input)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
            print(f"Inference -> Predicted Class: {pred_class}, Confidence: {confidence:.4f}")

    else:
        print("Unknown subcommand. Use 'train', 'eval', or 'infer'.")
