import sys
import argparse
import setup
from baddetGMA import BadDetGMA

if __name__ == '__main__':
    # Parse the arguments passed by the user 
    parser = argparse.ArgumentParser(description=sys.argv[0])
    setup.declare_args(parser)
    args = parser.parse_args()
    #setup.get_yaml_args(args)
    setup.set_args(args)

    # Based on the desired attack type, we create a corresponding attack instance:
    #   1. Global Misclassification Attack
    #   2. Regional Misclassification Attack
    #   3. Object Generation Attack
    #   4. Object Disappearance Attack
    if  args.attack_name == "gma":
        attack = BadDetGMA(args)
        attack.prepare_datasets()
    elif args.attack_name == "rma":
        pass
    elif args.attack_name == "oga":
        pass
    elif args.attack_name == "oda":
        pass

    


    