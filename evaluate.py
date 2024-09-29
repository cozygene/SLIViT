from auxiliaries.misc import *
from auxiliaries.evaluate import configure_hyperparam_values
from model.slivit import SLIViT
from utils.load_backbone import load_backbone

assert args.checkpoint is not None, 'No checkpoint provided. Please provide a checkpoint to evaluate the model.'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # set hps (including num slices which becomes num_patches) correctly before setting up the dataloaders
    configure_hyperparam_values(args)

    dls, test_loader, medmnist = setup_dataloaders(args)

    try:
        slivit = SLIViT(backbone=load_backbone(args.fe_classes),
                        vit_dim=args.vit_dim, vit_depth=args.vit_depth,
                        heads=args.heads, mlp_dim=args.mlp_dim, num_of_patches=args.slices)

    except RuntimeError as e:
        logger.error(f"Could not load model:\n{e}\n\nPlease double-check that the pretrained feature extractor is "
                     f"correctly set up and compatible with the model. This will ensure everything runs smoothly!\n")
        sys.exit(1)


    learner, _ = create_learner(slivit, dls, args, os.path.split(args.checkpoint)[0])

    # Evaluate and store results
    evaluate_and_store_results(learner, test_loader, args.checkpoint, args.meta, args.label, args.out_dir)

    wrap_up(args.out_dir)
