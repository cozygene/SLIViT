import argparse
from Options.base_options import BaseOptions
class TrainOptions():

    def initialize(self, parser):
        self.parser = parser
        parser = BaseOptions.initialize(self, parser) 
        parser.add_argument('--dataset', type=str, default='kermany', help='kermany,chestmnist,custom')
        parser.add_argument('--pathologies',type=str, default='CNV,Drusen,DME,Normal',  help='Comma Spreaded List of Labels to predict for Pre-training')
        parser.add_argument('--min_delta', type=float, default=0.1, help='minimum delta between the last monitor value and the best monitor value for early stopping')
        parser.add_argument('--patience', type=float, default=20,help='patience for early stopping')
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
