from auxiliaries.slivit_auxiliaries import *

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    dls, test_loader, mnist = setup_dataloaders(args, out_dir)
    try:
        slivit = SLIViT(backbone=load_backbone(args.fe_classes, args.fe_path),
                        fi_dim=args.vit_dim, fi_depth=args.vit_depth, heads=args.heads, mlp_dim=args.mlp_dim,
                        num_vol_frames=args.slices, dropout=args.dropout, emb_dropout=args.emb_dropout)
    except RuntimeError as e:
        logger.error(f"Could not load model:\n{e}\n\nPlease double-check that the pretrained feature extractor is "
                     f"correctly set up and compatible with the model. This will ensure everything runs smoothly!\n")
        sys.exit(1)

    learner, best_model_name = create_learner(slivit, dls, out_dir, args, mnist)

    try:
        train_and_evaluate(learner, out_dir, best_model_name, args, test_loader)
        wrap_up(out_dir)
    except Exception as e:
        wrap_up(out_dir, e)
    finally:
        if args.wandb_name is not None:
            wandb.finish()

