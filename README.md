# HeadGuard-Privacy-Preserving-Federated-Learning-for-Head-Worn-Health-Monitors
HeadGuard is a framework for defending against Membership Inference Attacks (MIA) in Federated Learning for head-worn health monitors (smart earbuds, AR glasses, VR headsets). It uses frequency-domain noise shaping to inject differential privacy noise into physiologically uninformative spectral bands
# HeadGuard

Code for "HeadGuard: Defending Against Membership Inference Attacks in Federated Learning for Head-Worn Health Monitors"

## What this is

Implementation of frequency-domain noise shaping for differential privacy in federated learning. Works on PPG, EOG, and IMU data from head-worn devices.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, SciPy, scikit-learn

Install with:

## Files

- `headguard/core.py` - Main noise shaping and DP logic
- `headguard/attacks.py` - MIA implementations (confidence, loss, metric)
- `headguard/utils.py` - PSD estimation, spectral weights
- `examples/quickstart.py` - Minimal working example
- `configs/default.yaml` - Parameters

## Quick example

```python
from headguard import HeadGuard

hg = HeadGuard(epsilon=4.0, delta=1e-5, sampling_rate=64)
hg.train(model, train_loader)


## Files

- `headguard/core.py` - Main noise shaping and DP logic
- `headguard/attacks.py` - MIA implementations (confidence, loss, metric)
- `headguard/utils.py` - PSD estimation, spectral weights
- `examples/quickstart.py` - Minimal working example
- `configs/default.yaml` - Parameters

## Quick example

```python
from headguard import HeadGuard

hg = HeadGuard(epsilon=4.0, delta=1e-5, sampling_rate=64)
hg.train(model, train_loader)
