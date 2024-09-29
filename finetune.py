from auxiliaries.finetune import *

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dls, test_loader, medmnist = setup_dataloaders(args)
    try:
        slivit = SLIViT(backbone=load_backbone(args.fe_classes, args.fe_path),
                        vit_dim=args.vit_dim, vit_depth=args.vit_depth, heads=args.heads, mlp_dim=args.mlp_dim,
                        num_of_patches=args.slices, dropout=args.dropout, emb_dropout=args.emb_dropout)
    except RuntimeError as e:
        logger.error(f"Could not load model:\n{e}\n\nPlease double-check that you have enough GPU memory, "
                     f"the pretrained feature extractor is correctly set up and compatible with the "
                     f"model. This will ensure everything runs smoothly!\n")
        sys.exit(1)

    learner, best_model_name = create_learner(slivit, dls, args, args.out_dir)

    err = None
    try:
        train_and_evaluate(args, learner, best_model_name, test_loader)
    except Exception as e:
        err = e
    wrap_up(args.out_dir, err)
    if args.wandb_name is not None:
        wandb.finish()

