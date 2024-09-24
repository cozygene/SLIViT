from torch.utils.data import ConcatDataset

from auxiliaries.slivit_auxiliaries import *
from auxiliaries.misc import get_dataloaders
from slivit import ConvNext
from utils.load_backbone import AutoModelForImageClassification as amfic

if args.wandb_name is not None:
    wandb.init(project=args.wandb_name)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    out_dir = init_out_dir(args)

    dls, empty_test_set = setup_dataloaders(args, out_dir)  # no test here...

    convnext = ConvNext(amfic.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                              # length of the target vector
                                              num_labels=dls.train.dataset.dataset.get_num_classes(),
                                              ignore_mismatched_sizes=True))

    learner, best_model_name = create_learner(convnext, dls, out_dir, args)

    try:
        train_and_evaluate(learner, out_dir, best_model_name, args, empty_test_set)
        wrap_up(out_dir)
    except Exception as e:
        wrap_up(out_dir, e)
        raise e
    # finally:
    #     if args.wandb_name is not None:
    #         wandb.finish()

