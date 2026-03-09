import os

from gws_core import MambaShellProxy, MessageDispatcher


class UMAPEnvHelper:
    """Helper class to manage the UMAP virtual environment."""

    unique_env_name = "UMAPEnvTask"
    env_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "umap_env.yml")

    @classmethod
    def create_proxy(cls, message_dispatcher: MessageDispatcher | None = None) -> MambaShellProxy:
        """
        Create a MambaShellProxy for the UMAP environment.

        Args:
            message_dispatcher: MessageDispatcher instance for logging

        Returns:
            MambaShellProxy instance configured for UMAP environment
        """
        if message_dispatcher is not None:
            return MambaShellProxy(
                env_file_path=cls.env_file_path,
                env_name=cls.unique_env_name,
                message_dispatcher=message_dispatcher,
            )
        return MambaShellProxy(env_file_path=cls.env_file_path, env_name=cls.unique_env_name)
