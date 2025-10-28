# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os

from gws_core import MambaShellProxy, MessageDispatcher


class EconmlEnvHelper:
    """Helper class to manage the econml virtual environment."""

    unique_env_name = "EconmlEnvTask"
    env_file_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "econml_env.yml"
    )

    @classmethod
    def create_proxy(cls, message_dispatcher: MessageDispatcher = None) -> MambaShellProxy:
        """
        Create a MambaShellProxy for the econml environment.

        Args:
            message_dispatcher: MessageDispatcher instance for logging

        Returns:
            MambaShellProxy instance configured for econml environment
        """
        return MambaShellProxy(
            env_file_path=cls.env_file_path,
            env_name=cls.unique_env_name,
            message_dispatcher=message_dispatcher
        )
