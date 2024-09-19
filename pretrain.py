from utils.slivit_auxiliaries import *
from utils.load_backbone import AutoModelForImageClassification

#TODO:
def get_pretraining_datasets(args, logger):
    if args.dataset == 'chestmnist':  # TODO: extend to all mnists
        from medmnist import ChestMNIST
        from datasets.ChestMNIST import CMNIST as dataset_class

        # TODO move this whole block into get_datasets()
        train_dataset = ChestMNIST(split="train", download=True)
        valid_dataset = ChestMNIST(split="val", download=True)
        test_dataset = ChestMNIST(split="test", download=True)

        if args.mnist_mocks is not None:
            logger.info(
                '\n\n************ ******************************************************************************')
            logger.info(f'Running a mock version of the dataset with {args.mnist_mocks} samples only!!\n\
******************************************************************************************\n')

            train_dataset = Subset(train_dataset, np.arange(0, args.mnist_mocks))
            valid_dataset = Subset(valid_dataset, np.arange(0, args.mnist_mocks))
            test_dataset = Subset(test_dataset, np.arange(0, args.mnist_mocks))

        train_dataset = dataset_class(train_dataset)
        valid_dataset = dataset_class(valid_dataset)
    elif args.dataset == 'kermany':
        from datasets.KermanyDataset import KermanyDataset as dataset_class
        dataset = dataset_class(args.meta_csv,
                                args.meta_csv,
                                args.data_dir,
                                pathologies=[p for p in args.label2d.split(',')])
        df = pd.read_csv(args.meta_csv)
        splts = [p.split('/')[1] for p in df['path'].values]
        splitter = TrainTestSplitter(test_size=0.1, random_state=42)

        train_indices, valid_indices2 = splitter(dataset)
        valid_dataset = Subset(dataset, valid_indices2)

        train_dataset = Subset(dataset, train_indices)
    else:
        raise NotImplementedError

    return train_dataset, valid_dataset


if __name__ == '__main__':
    batch_size = args.batch
    num_workers = args.cpus
    print(f'Num of cpus is {num_workers}')
    print(f'Batch size is {batch_size}')

    #TODO: consider replacing with fastai's ImageDataLoaders
    '''
    set_seed(42, reproducible=True)
    df = pd.read_csv(csv_path)
    df['is_valid'] = df['Path'].str.contains('test', case=False)
    # Create the DataLoaders using from_df
    dls = ImageDataLoaders.from_df(
        df=df,
        path=data_dir, 
        folder='.', 
        fn_col='Path', 
        label_col=['Normal', 'DME', 'CNV', 'Drusen'], 
        y_block=MultiCategoryBlock(encoded=True, vocab=['Normal', 'DME', 'CNV', 'Drusen']), 
        bs=128,
        valid_col='is_valid',
        item_tfms=Resize((224, 224))  # Resize transformation
        # Uncomment the line below for data augmentation
        # batch_tfms=aug_transforms(mult=2)
    )
    '''
    train_dataset, valid_dataset = get_pretraining_datasets(args, logger)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of train batches is {len(train_loader)}')
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    print(f'# of validation batches is {len(valid_loader)}')
    dls = DataLoaders(train_loader, valid_loader)
    dls.c = 2


    class ConvNext(nn.Module):
        def __init__(self, model):
            super(ConvNext, self).__init__()
            self.model = model

        def forward(self, x):
            x = self.model(x)[0]
            return x


    model = ConvNext(AutoModelForImageClassification.from_pretrained("facebook/convnext-tiny-224", return_dict=False,
                                                                     num_labels=len(train_dataset[0][1]),  # length of the target vector
                                                                     ignore_mismatched_sizes=True))
    model.to(device='cuda')
    learner = Learner(dls, model, model_dir=out_dir,
                      loss_func=torch.nn.BCEWithLogitsLoss())
    # fp16 = MixedPrecision()
    learner = learner.to_fp16()

    learner.metrics = [RocAucMulti(average=None), APScoreMulti(average=None)]

    checkpoint_name = 'convnext_tiny_feature_extractor'
    learner.fit(lr=1e-5, n_epoch=args.epochs, cbs=[SaveModelCallback(fname=checkpoint_name),
                                                   CSVLogger(),
                                                   EarlyStoppingCallback(monitor='valid_loss', min_delta=args.min_delta
                                                                         , patience=args.patience)])

    logger.info(f'Trained feature extractor is saved at:\n{out_dir}/{checkpoint_name}.pth\n')
    logger.info(f'Running configuration is saved at:\n{options_file}\n')

    wrap_up(out_dir)
