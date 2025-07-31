from generators.sets import run_sets_counterfactuals
from generators.tsevo import run_tsevo_counterfactuals
from generators.confetti_generator import run_confetti_counterfactuals
from generators.comte import run_comte_counterfactuals
from generators.patch_tsinterpret import patch_comte
from data.generate_samples import create_samples

def main():
    #Generate instances to explain
    create_samples("fcn")
    create_samples("resnet")

    #Run all counterfactual generators for both FCN and ResNet models
    run_confetti_counterfactuals(model_name='fcn')
    run_confetti_counterfactuals(model_name='resnet')
    #To run CoMTE, it is necessary to first run a patch script to fix an issue in TSInterpret
    patch_comte()
    run_comte_counterfactuals(model_name='fcn')
    run_comte_counterfactuals(model_name='resnet')
    #Please be aware that both TSEvo and SETS take a long time to run (Almost a day each).
    run_tsevo_counterfactuals(model_name='fcn')
    run_tsevo_counterfactuals(model_name='resnet')
    run_sets_counterfactuals(model_name='fcn')
    run_sets_counterfactuals(model_name='resnet')

if __name__  == "__main__":
    main()