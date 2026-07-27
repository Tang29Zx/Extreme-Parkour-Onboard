"""Resolve ROS output topics without importing the hardware runtime."""

from typing import Optional, Tuple


SPORT_STATE_TOPIC = "/api/robot_state/request"
SPORT_MODE_TOPIC = "/api/sport/request"


def resolve_output_topics(
    low_cmd_topic: str,
    dryrun: bool,
    dryrun_suffix: Optional[int] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Return LowCmd and Sport API topics for the selected output mode."""
    if not low_cmd_topic:
        raise ValueError("low_cmd_topic must not be empty.")
    if dryrun:
        if dryrun_suffix is None:
            raise ValueError("dryrun_suffix is required in dry-run mode.")
        return f"{low_cmd_topic}_dryrun_{dryrun_suffix}", None, None
    return low_cmd_topic, SPORT_STATE_TOPIC, SPORT_MODE_TOPIC
