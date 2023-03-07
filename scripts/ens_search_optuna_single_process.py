import albumentations as A
import hydra
import optuna
import torch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning import seed_everything


class Objective:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __call__(self, trial: optuna.trial.Trial):
        cfg = self.cfg
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
        # scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.6)
        # scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)

        cfg.module.model.learning_rate = learning_rate
        cfg.module.model.optimizer = optimizer
        cfg.module.model.weight_decay = weight_decay
        # cfg.module.model.scheduler_patience = scheduler_patience
        # cfg.module.model.scheduler_factor = scheduler_factor

        horizontal_flip = trial.suggest_categorical("horizontal_flip", [True, False])
        vertical_flip = trial.suggest_categorical("vertical_flip", [True, False])
        rotate = trial.suggest_categorical("rotate", [True, False])
        affine_transform = trial.suggest_categorical("affine_transform", [True, False])
        random_gamma = trial.suggest_categorical("random_gamma", [True, False])
        gaussian_blur = trial.suggest_categorical("gaussian_blur", [True, False])

        if rotate:
            rotate_limit = trial.suggest_int("rotate_limit", 5, 30)
        if affine_transform:
            translate_percent = trial.suggest_int("translate_percent", 1, 30)
        if random_gamma:
            random_gamma_lb = trial.suggest_int("random_gamma_lb", 60, 95)
            random_gamma_ub = trial.suggest_int("random_gamma_ub", 105, 140)
        if gaussian_blur:
            gaussian_blur_lb = trial.suggest_int("gaussian_blur_lb", 5, 17, step=2)
            gaussian_blur_ub = trial.suggest_int("gaussian_blur_ub", gaussian_blur_lb, 41, step=2)

        t = []
        if horizontal_flip:
            t.append(A.HorizontalFlip(p=0.5))
        if vertical_flip:
            t.append(A.VerticalFlip(p=0.5))
        if rotate:
            t.append(A.SafeRotate(limit=rotate_limit, p=0.2))
        if affine_transform:
            t.append(A.Affine(translate_percent=translate_percent, p=0.5))
        if gaussian_blur:
            t.append(A.GaussianBlur(blur_limit=(gaussian_blur_lb, gaussian_blur_ub), p=0.3))
        if random_gamma:
            t.append(A.RandomGamma(gamma_limit=(random_gamma_lb, random_gamma_ub), p=0.3))

        if cfg.get("seed"):
            seed_everything(cfg.seed, workers=True)

        callbacks = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        logger = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    logger.append(hydra.utils.instantiate(lg_conf))

        dm = hydra.utils.instantiate(cfg.datamodule, transforms=t)
        model = hydra.utils.instantiate(cfg.module.model)

        if cfg.module.get("pretrained"):
            checkpoint = torch.load(cfg.module.pretrained)
            model.load_state_dict(checkpoint["state_dict"])

        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/loss"))

        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            gpus=1,
            # auto_select_gpus=True,
        )
        trainer.fit(model=model, datamodule=dm)
        return trainer.callback_metrics["val/loss"].item()


def main():
    with hydra.initialize(config_path="mt/configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=["experiment=train_yolo", "callbacks=optuna_callbacks", "logger=null"],
        )
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.load_study(
        # direction="minimize",
        pruner=pruner,
        # storage="sqlite:///optim_optuna.db",
        storage="mysql://root@34.116.169.28/example",
        study_name="yolo_optim2",
    )
    # study.optimize(lambda trial: objective(trial, cfg), n_trials=50, n_jobs=4)
    study.optimize(Objective(cfg), n_trials=50)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial: ")
    trial = study.best_trial

    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))


if __name__ == "__main__":
    main()
