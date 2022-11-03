"""helper function for the wandb."""
import wandb  # noqa: E402


def wandb_init(project, config):
    """Initialize the wandb."""
    wandb.init(
        reinit=True,
        config=config,
        project=project,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
    )
