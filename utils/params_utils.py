def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelParams", "PipelineParams", "FlowParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    return args