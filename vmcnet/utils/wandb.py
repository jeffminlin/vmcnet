"""helper function for the wandb"""
import wandb 

def wandb_init(project, config):
    """initialize the wandb"""
    wandb.init(
                reinit=True,
                config=config,
                project=project,
                settings=wandb.Settings(
                    start_method="thread",
                    _disable_stats=True,
                ),
            )
