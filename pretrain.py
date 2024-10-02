from auxiliaries.misc import *
from model.feature_extractor import ConvNext, amfic

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dls, empty_test_set = setup_dataloaders(args)  # no test here...

    feature_extractor = ConvNext(amfic.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                                       num_labels=dls.train.dataset.dataset.get_num_classes(),
                                                       ignore_mismatched_sizes=True))

    learner, best_model_name = create_learner(feature_extractor, dls, args, args.out_dir)

    err = None
    try:
        train(args, learner, best_model_name)
    except Exception as e:
        err = e
    wrap_up(args.out_dir, err)
    if args.wandb_name is not None:
        wandb.finish()

