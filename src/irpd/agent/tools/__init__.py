"""The agent's tools. Importing this package registers them in `registry.TOOLS`.

Every tool takes the connection from `holdout.agent_view(day)` and reads only
through it; the day it stands on is the last day `daily_stock` shows.
"""

from irpd.agent.tools import drugs, plan, sql, stock  # noqa: F401
from irpd.agent.tools.plan import reorder_rows, to_policy  # noqa: F401
