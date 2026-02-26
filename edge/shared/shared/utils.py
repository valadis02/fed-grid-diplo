import os
from typing import Iterable, Optional, List

from shared.logging_config import logger

required_columns = [
    'value',  # Raw consumption value
    'value_diff',  # First difference
    # Window 3 (30 minutes)
    'value_rolling_mean_3',
    'value_volatility_3',
    'value_ewm_3',
    # Window 6 (1 hour)
    'value_rolling_mean_6',
    'value_volatility_6',
    'value_ewm_6',
    # Window 12 (2 hours)
    'value_rolling_mean_12',
    'value_volatility_12',
    'value_ewm_12',
    # Window 24 (4 hours)
    'value_rolling_mean_24',
    'value_volatility_24',
    'value_ewm_24',
    'drift_flag',
    'time_since_last_spike'
]

def delete_files_containing(
        directory: str,
        substring: str | None,
        extensions: Optional[Iterable[str]] = None,
        recursive: bool = False
) -> List[str]:
    """
    Delete files in `directory` whose filename contains `substring`. Optionally restrict by file extensions (.keras).
    Returns the list if deleted file paths.
    """

    deleted: List[str] = []
    if not os.path.isdir(directory):
        logger.warning(f"Directory {directory} does not exist. Ignoring.")
        return deleted

    walker = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]

    for root, dirs, files in walker:
        for filename in files:
            if (substring is None or substring in filename) and (extensions is None or any(filename.endswith(ext) for ext in extensions)):
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    deleted.append(filepath)
                except Exception as e:
                    logger.warning(f"Failed to remove file {filepath}: {e}")

    if deleted:
        logger.warning(f"Deleted {len(deleted)} files. Ignoring.")
    else:
        logger.debug(f"No files matched substring=`{substring}` in directory {directory}.")
    return deleted