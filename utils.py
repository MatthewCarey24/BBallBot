"""Utility functions for BBallBot"""

import numpy as np
from typing import Tuple, TypeVar, Union, Optional, Protocol, runtime_checkable, Any, Sized

@runtime_checkable
class SupportsGetItem(Protocol, Sized):
    def __getitem__(self, key: Union[int, np.ndarray]) -> Any: ...
    def __len__(self) -> int: ...

ArrayLike = TypeVar('ArrayLike', bound=Union[np.ndarray, list, SupportsGetItem])

def split_into_train_and_test(
    data: ArrayLike,
    frac_test: float,
    random_state: Optional[int] = None
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Split data into training and test sets, using the latter portion of the data for testing.
    This is specifically designed for time-series data where we want to test on
    the latter part of the season.
    
    Args:
        data: Array-like data to split
        frac_test: Fraction of data to use for testing (between 0 and 1)
        random_state: Not used, kept for compatibility
        
    Returns:
        Tuple of (training data, test data)
    """
    if not 0 <= frac_test <= 1:
        raise ValueError("frac_test must be between 0 and 1")
    
    n = len(data)
    split_idx = int(n * (1 - frac_test))
    
    if isinstance(data, np.ndarray):
        return data[:split_idx], data[split_idx:]  # type: ignore
    else:
        return (  # type: ignore
            [data[i] for i in range(split_idx)],
            [data[i] for i in range(split_idx, n)]
        )
