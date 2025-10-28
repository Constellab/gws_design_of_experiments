# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os

from gws_core import MambaShellProxy, MessageDispatcher


class OptimizationEnvHelper:
    """Helper class to manage the optimization virtual environment."""

    unique_env_name = "OptimizationEnvTask"
    env_file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "optimization_env.yml"
    )

    @classmethod
    def create_proxy(cls, message_dispatcher: MessageDispatcher = None) -> MambaShellProxy:
        """
        Create a MambaShellProxy for the optimization environment.

        Args:
            message_dispatcher: MessageDispatcher instance for logging

        Returns:
            MambaShellProxy instance configured for optimization environment
        """
        return MambaShellProxy(
            env_file_path=cls.env_file_path,
            env_name=cls.unique_env_name,
            message_dispatcher=message_dispatcher
        )
