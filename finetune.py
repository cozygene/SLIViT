# finetune.py

from auxiliaries.slivit_auxiliaries import *

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    slivit = setup_slivit(args)

    dls, test_loader = setup_dataloaders(args, out_dir)

    learner, best_model_name = create_learner(slivit, dls, out_dir, args)

    try:
        learner.fine_tune(args.epochs, args.lr) if args.fine_tune else learner.fit(args.epochs, args.lr)
        logger.info(f'Best model is stored at:\n{out_dir}/{best_model_name}.pth')
        if len(test_loader):
            evaluate_and_store_results(learner, test_loader, best_model_name, args.meta_data, args.label3d, out_dir)
        else:
            logger.info('No test set provided. Skipping evaluation...')
    except torch.cuda.OutOfMemoryError as e:
        print(f'\n{e.args[0]}\n')
        logger.error('Out of memory error occurred. Exiting...\n')
        wrap_up(out_dir, e.args[0])
        sys.exit(1)
    if args.wandb_name is not None:
        wandb.finish()
    wrap_up(out_dir)
