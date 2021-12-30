"""Contains arguments on training the model."""
import logging
import os
import yaml
from argparse import ArgumentParser, Namespace
from distutils.util import strtobool
from enum import Enum
from typing import Any, Callable, List
from datetime import datetime
from abc import ABC

import torch

class Args(ABC):
    """ Abstract class for arguments.

    Members:
        initialized (bool): Whether the instance arguments are initialized and valid to use.

    """

    def __init__(self):
        self._initialized: bool = False

    def interpret_args(self, parsed_args: Namespace) -> None:
        """ Interprets the arguments included in the parsed command line arguments.

        Inputs:
            parsed_args (Namespace): The parsed command line arguments.
        """
        self._initialized = True

    def is_initialized(self) -> bool:
        """ Returns whether these arguments have been initialized.

        Returns:
            True if initialized; false otherwise.
        """
        return self._initialized

    def check_initialized(self) -> None:
        """ Raises a ValueError if not initialized. """
        if not self.is_initialized():
            raise ValueError('Arguments are not initialized.')


class TrainingArgs(Args):
    def __init__(self, parser: ArgumentParser):
        super(TrainingArgs, self).__init__()

        # Experiment oprions
        parser.add_argument('--logger_directory',
                            default='learning/logs',
                            type=str,
                            help='Location to save model saves and other information.')

        parser.add_argument('--logging_step',
                            default=50,
                            type=int,
                            help='Location to save model saves and other information.')

        parser.add_argument('--turnoff_wandb',
                            action='store_true')

        parser.add_argument('--debug',
                            action='store_true')

        parser.add_argument('--validate_only',
                            action='store_true')

        parser.add_argument('--no_validation',
                            action='store_true')

        parser.add_argument('--checkpoint_directory',
                            default='checkpoints',
                            type=str,
                            help='the max number of training / validation exmaples')

        parser.add_argument('--validation_step',
                            default=40000,
                            type=int,
                            help='Location to save model saves and other information.')

        parser.add_argument('--checkpoint_step',
                            default=40000,
                            type=int,
                            help='Location to save model saves and other information.')

        parser.add_argument('--experiment_name',
                            default='',
                            type=str,
                            help='The experiment name. A directory will be created under save_directory for it.')

        # Model options
        parser.add_argument('--train_config_file_name',
                            default='',
                            type=str,
                            help='')

        # Training options
        parser.add_argument('--max_epochs',
                            default=20,
                            type=int,
                            help='number of epochs')

        parser.add_argument('--num_workers',
                            default=0,
                            type=int,
                            help='number of epochs')

        parser.add_argument('--validate',
                            action='store_true')

        parser.add_argument('--save_checkpoint',
                            action='store_true')

        # Data options
        parser.add_argument('--dataset_file_name',
                            default="",
                            type=str,
                            help='')

        # Resume training
        parser.add_argument('--resume_training',
                            action='store_true')
        parser.add_argument('--wandb_id',
                            default="",
                            type=str,
                            help='')
        parser.add_argument('--overwrite_wandb_name',
                            default="",
                            type=str,
                            help='')

        self._logger_directory: str = None
        self._logging_step: int = None
        self._turnoff_wandb: bool = None
        self._debug: bool = None
        self._validate_only: bool = None
        self._no_validation: bool = None
        self._save_checkpoint: bool = None
        self._checkpoint_directory: str = None
        self._validation_step:  int = None
        self._checkpoint_step:  int = None
        self._experiment_name: str = None

        self._max_epochs: int = None
        self._num_workers: int = None

        self._train_config_file_name: str = None
        self._dataset_file_name: str = None

        self._resume_training: bool = None
        self._wandb_id: str = None
        self._overwrite_wandb_name: str = None

        now = datetime.now()
        self._now_str = str(now).replace(" ", "-")

    def get_logger_directory(self) -> str:
        self.check_initialized()

        # Create the dir if it does not exist
        full_dir: str = os.path.join(self._logger_directory,
                                     self._experiment_name+"-"+self._now_str)
        os.makedirs(full_dir, exist_ok=True)

        return full_dir

    def get_logging_step(self) -> int:
        return self._logging_step


    def get_turnoff_wandb(self) -> bool:
        self.check_initialized()
        return self._turnoff_wandb

    def get_debug(self) -> bool:
        self.check_initialized()
        return self._debug

    def get_validate_only(self) -> bool:
        self.check_initialized()
        return self._validate_only

    def get_no_validation(self) -> bool:
        self.check_initialized()
        return self._no_validation

    def get_save_checkpoint(self) -> bool:
        self.check_initialized()
        return self._save_checkpoint

    def get_checkpoint_directory(self) -> str:
        self.check_initialized()
        full_dir: str = os.path.join(self._checkpoint_directory, self._experiment_name)

        if not os.path.exists(full_dir):
            print('Created directory: ' + full_dir)
            os.mkdir(full_dir)

        return full_dir

    def get_validation_step(self) -> str:
        return self._validation_step

    def get_checkpoint_step(self) -> str:
        return self._checkpoint_step

    def get_experiment_name(self) -> str:
        return self._experiment_name

    def get_now_str(self) -> str:
        return self._now_str

    def get_wandb_name(self) -> str:
        return self._experiment_name+"-"+self._now_str

    def get_config(self) -> str:
        self.check_initialized()
        return self._config

    def get_train_type(self) -> str:
        self.check_initialized()
        return self._train_type

    def get_max_epochs(self) -> int:
        self.check_initialized()
        return self._max_epochs

    def get_num_workers(self) -> int:
        self.check_initialized()
        return self._num_workers

    def get_resume_training(self) -> bool:
        self.check_initialized()
        return self._resume_training

    def get_wandb_id(self) -> str:
        self.check_initialized()
        return self._wandb_id

    def get_overwrite_wandb_name(self) -> str:
        self.check_initialized()
        return self._overwrite_wandb_name

    def interpret_args(self, parsed_args: Namespace) -> None:
        self._logger_directory = parsed_args.logger_directory
        self._logging_step = parsed_args.logging_step
        self._turnoff_wandb = parsed_args.turnoff_wandb
        self._debug = parsed_args.debug
        self._validate_only = parsed_args.validate_only
        self._no_validation = parsed_args.no_validation
        self._save_checkpoint = parsed_args.save_checkpoint
        self._checkpoint_directory = parsed_args.checkpoint_directory
        self._checkpoint_step = parsed_args.checkpoint_step
        self._validation_step = parsed_args.validation_step
        self._experiment_name = parsed_args.experiment_name

        self._max_epochs = parsed_args.max_epochs
        self._num_workers = parsed_args.num_workers
        self._train_config_file_name = parsed_args.train_config_file_name

        self._resume_training = parsed_args.resume_training
        self._wandb_id = parsed_args.wandb_id
        self._overwrite_wandb_name = parsed_args.overwrite_wandb_name

        # read in training config
        infile = open(self._train_config_file_name, "r")
        self._config = yaml.full_load(infile)

        super(TrainingArgs, self).interpret_args(parsed_args)

    def __str__(self) -> str:
        str_rep = "{}, {}, {}".format(self._save_directory,
                                              self._experiment_name, self._train_config)

        return str_rep

    def __eq__(self, other) -> bool:
        return True
