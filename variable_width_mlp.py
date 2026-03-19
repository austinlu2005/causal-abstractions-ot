"""Config and model definitions for the shared variable-width addition MLP."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableWidthMLPConfig:
    """Configuration for the variable-width MLP classifier."""

    def __init__(
        self,
        input_dim=40,
        hidden_dims=None,
        num_classes=200,
        dropout=0.0,
        activation="relu",
        include_bias=True,
        squeeze_output=True,
    ):
        """Store the configurable architecture and loss settings for the MLP."""
        if hidden_dims is None:
            hidden_dims = [192, 192, 192, 192]
        self.input_dim = int(input_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.n_layer = len(self.hidden_dims)
        self.h_dim = self.hidden_dims[-1]
        self.num_classes = int(num_classes)
        self.pdrop = float(dropout)
        self.activation_function = str(activation)
        self.include_bias = bool(include_bias)
        self.squeeze_output = bool(squeeze_output)
        self.problem_type = "single_label_classification"

    def to_dict(self):
        """Serialize the config to a plain Python dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_classes": self.num_classes,
            "dropout": self.pdrop,
            "activation": self.activation_function,
            "include_bias": self.include_bias,
            "squeeze_output": self.squeeze_output,
        }


class VariableWidthMLPBlock(nn.Module):
    """One hidden MLP block with linear, activation, and optional dropout."""

    def __init__(self, in_dim, out_dim, activation, dropout, include_bias):
        """Build one hidden linear layer plus activation and dropout."""
        super().__init__()
        self.ff = nn.Linear(in_dim, out_dim, bias=include_bias)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, hidden_states):
        """Apply the feed-forward block to hidden activations."""
        return self.dropout(self.act(self.ff(hidden_states)))


class VariableWidthMLPForClassification(nn.Module):
    """MLP classifier used as the addition backbone model."""

    def __init__(self, config):
        """Build the full MLP classifier from a config object."""
        super().__init__()
        self.config = config
        self.device = torch.device("cpu")
        self.h = nn.ModuleList()

        in_dim = config.input_dim
        for out_dim in config.hidden_dims:
            self.h.append(
                VariableWidthMLPBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=config.activation_function,
                    dropout=config.pdrop,
                    include_bias=config.include_bias,
                )
            )
            in_dim = out_dim

        self.score = nn.Linear(in_dim, config.num_classes, bias=config.include_bias)

    def to(self, *args, **kwargs):
        """Move the model and keep track of the active device."""
        module = super().to(*args, **kwargs)
        if len(args) > 0 and isinstance(args[0], torch.device):
            self.device = args[0]
        elif "device" in kwargs and kwargs["device"] is not None:
            self.device = torch.device(kwargs["device"])
        return module

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        """Run the MLP and optionally return a cross-entropy loss."""
        hidden_states = inputs_embeds if inputs_embeds is not None else input_ids
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1)

        for block in self.h:
            hidden_states = block(hidden_states)

        logits = self.score(hidden_states)
        if self.config.squeeze_output:
            logits = logits.squeeze(1)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.num_classes), labels.view(-1))
            return (loss, logits)

        return (logits,)


def logits_from_output(output):
    """Extract logits from tuple-style or object-style model outputs."""
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "logits"):
        return output.logits
    return output


def load_variable_width_mlp_checkpoint(checkpoint_path, device):
    """Load a saved MLP checkpoint and restore the model config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = VariableWidthMLPConfig(**checkpoint["model_config"])
    model = VariableWidthMLPForClassification(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, checkpoint
