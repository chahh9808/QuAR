from .data import load_cifar10
from .utils import load_model
try:
    from .eval import benchmark
except ImportError:
    pass
