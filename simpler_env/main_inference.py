import os

import numpy as np

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator


def _configure_tensorflow_memory(tf_memory_limit):
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=tf_memory_limit)],
        )


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        assert args.ckpt_path is not None
        _configure_tensorflow_memory(args.tf_memory_limit)
        from simpler_env.policies.rt1.rt1_model import RT1Inference

        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        _configure_tensorflow_memory(args.tf_memory_limit)
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            from simpler_env.policies.octo.octo_server_model import OctoServerInference

            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            from simpler_env.policies.octo.octo_model import OctoInference

            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif args.policy_model == "stair":
        if args.ckpt_path is None or args.ckpt_path == "None":
            raise ValueError("STAIR evaluation requires --ckpt-path to point to a stage-2 checkpoint")
        from stair.deploy.stair_policy_loader import build_stair_simpler_policy

        model = build_stair_simpler_policy(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_model_type=args.action_model_type,
            num_inference_steps=args.num_inference_steps,
            unnorm_key=args.unnorm_key,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
