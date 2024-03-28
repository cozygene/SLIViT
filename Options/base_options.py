import argparse

class BaseOptions():

    def initialize(self, parser):
        self.parser = parser
        
        parser.add_argument('--data_dir', type=str, default='./', help='Root path to the data')
        parser.add_argument('--meta_csv', type=str,  default='./Dsets/kermany_meta.csv',help='Path to meta csv file')
        parser.add_argument('--out_dir', type=str, default='./', help='Output directory to save the trained model')
        parser.add_argument('--gpu_id', type=int ,default=0, help='GPU ID for training')
        parser.add_argument('--b_size', type=int ,default=16, help='Batch size for training')
        parser.add_argument('--n_cpu', type=int ,default=16, help='# of cpus to use for loading data')
        parser.add_argument('--n_epochs', type=int ,default=20, help='# of training epochs')
        return self.parser
    
    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt
    
    def gather_options(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()