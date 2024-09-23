# finetune.py

from auxiliaries.slivit_auxiliaries import *

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    slivit = SLIViT(backbone=load_backbone(args.fe_classes, args.fe_path),
                    fi_dim=args.vit_dim, fi_depth=args.vit_depth, heads=args.heads, mlp_dim=args.mlp_dim,
                    num_vol_frames=args.slices, dropout=args.dropout, emb_dropout=args.emb_dropout)

    dls, test_loader = setup_dataloaders(args, out_dir)

    learner, best_model_name = create_learner(slivit, dls, out_dir, args)

    try:
        train_and_evaluate_slivit(learner, test_loader, out_dir, best_model_name, args)
        wrap_up(out_dir)
    except Exception as e:
        wrap_up(out_dir, e.args[0])
        sys.exit(1)
    finally:
        if args.wandb_name is not None:
            wandb.finish()
