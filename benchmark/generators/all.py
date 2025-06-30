from sets import run_sets_counterfactuals
from tsevo import run_tsevo_counterfactuals
from confetti_generator import run_confetti_counterfactuals

def main():
    run_confetti_counterfactuals(model_name='resnet')
    run_tsevo_counterfactuals(model_name='fcn')
    run_tsevo_counterfactuals(model_name='resnet')
    run_sets_counterfactuals(model_name='fcn')
    run_sets_counterfactuals(model_name='resnet')

if __name__  == "__main__":
    main()