import os
from auxiliaries.misc import logger


def configure_hyperparam_values(args):
    checkpoint_options_file = f'{os.path.split(args.checkpoint)[0]}/finetune_options.txt'

    if os.path.exists(checkpoint_options_file) and not args.ignore_options_file:
        logger.info('Using hyperparameters from checkpoint options file ' 
                    '(user-provided parameters will be ignored if provided).')
        with open(checkpoint_options_file, 'r') as f:
            checkpoint_options = f.read().strip().split(' --')
        for option in checkpoint_options:
            delimiter = option.find(' ')
            if option[:delimiter] in ['fe_classes', 'vit_dim', 'vit_depth', 'heads', 'mlp_dim', 'slices']:
                hp = option[:delimiter]
                value = int(option[delimiter+1:].strip('"'))
                setattr(args, hp, value)

    else:
        if args.ignore_options_file:
            logger.warning('Ignoring checkpoint options file.')
        else:
            logger.info('Could not find checkpoint options file.')
        logger.info('Using user-provided/default hyperparameters instead.')

    result = {'fe_classes': args.fe_classes, 'vit_dim': args.vit_dim, 'vit_depth': args.vit_depth,
              'heads': args.heads, 'mlp_dim': args.mlp_dim, 'slices': args.slices}

    logger.info(f"Hyperparameters: {result}")
    return