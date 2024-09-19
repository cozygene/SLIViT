from utils.slivit_auxiliaries import *

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    slivit = setup_slivit(args)

    dls, test_loader = setup_dataloaders(args, out_dir)

    learner, _ = create_learner(slivit, dls, out_dir, args)

    # Evaluate and store results
    evaluate_and_store_results(learner, test_loader, args.checkpoint, args.meta_csv, args.label3d, out_dir)

    wrap_up(out_dir)
