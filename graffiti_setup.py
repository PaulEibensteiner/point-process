"""``panel serve --setup`` script: start the initial fit before the first request."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graffiti_model import get_job_manager  # noqa: E402

get_job_manager()
