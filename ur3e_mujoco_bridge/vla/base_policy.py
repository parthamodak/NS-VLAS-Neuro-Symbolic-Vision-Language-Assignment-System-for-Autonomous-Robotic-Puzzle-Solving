"""Backend-independent policy interface."""

from abc import ABC, abstractmethod

from .action import ProjectAction
from .observation import VLAObservation


class BaseVLAPolicy(ABC):
    """A policy returns one or more project actions as a safe-to-queue chunk."""

    @abstractmethod
    def predict(self, observation: VLAObservation) -> list[ProjectAction]:
        """Predict a finite action chunk; validation is performed by safety.py."""
