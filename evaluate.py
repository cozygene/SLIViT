from auxiliaries.slivit_auxiliaries import *

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    dls, test_loader = setup_dataloaders(args, out_dir)

    slivit = SLIViT(backbone=load_backbone(args.fe_classes, args.fe_path),
                    fi_dim=args.vit_dim, fi_depth=args.vit_depth, heads=args.heads, mlp_dim=args.mlp_dim,
                    num_vol_frames=args.slices, dropout=args.dropout, emb_dropout=args.emb_dropout)

    learner, _ = create_learner(slivit, dls, out_dir, args)

    # Evaluate and store results
    evaluate_and_store_results(learner, test_loader, args.checkpoint, args.meta_data, args.label3d, out_dir)

    wrap_up(out_dir)
