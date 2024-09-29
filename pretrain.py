from torch.utils.data import ConcatDataset

from auxiliaries.slivit_auxiliaries import *
from auxiliaries.misc import get_dataloaders
from model.slivit import ConvNext
from utils.load_backbone import AutoModelForImageClassification as amfic

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    dls, empty_test_set, mnist = setup_dataloaders(args)  # no test here...

    convnext = ConvNext(amfic.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                              # length of the target vector
                                              num_labels=dls.train.dataset.dataset.get_num_classes(),
                                              ignore_mismatched_sizes=True))

    learner, best_model_name = create_learner(convnext, dls, args, args.out_dir, mnist)

    err = None
    try:
        train_and_evaluate(args, learner, best_model_name, empty_test_set)
    except Exception as err:
        pass
    wrap_up(args.out_dir, err)
    if args.wandb_name is not None:
        wandb.finish()

