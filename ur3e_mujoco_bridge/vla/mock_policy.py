"""TEST-ONLY policy for exercising the VLA data path without ML dependencies."""

from .action import ProjectAction
from .base_policy import BaseVLAPolicy
from .observation import VLAObservation


class MockPolicy(BaseVLAPolicy):
    """Returns an explicitly non-autonomous hold/open action.

    This does not use object coordinates and must never be described as a VLA.
    """

    def predict(self, observation: VLAObservation) -> list[ProjectAction]:
        del observation
        return [ProjectAction(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
