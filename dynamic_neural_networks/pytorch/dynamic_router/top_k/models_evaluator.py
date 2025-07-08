import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch
import re
import numpy as np


from data_transformations import DIR_INFO, DIR_MODELS, transform_data_for_training, ZDCType
from utils import load_train_test_indices
from training_setup import setup_experts, setup_router, setup_router_attention
from utils_eval import (get_mean_std_from_expert_genrations, plot_proton_photonsum_histograms_shared,
                        plot_proton_photonsum_histogreams, make_histograms)
from utils import get_predictions_from_generator_results, calculate_ws_ch_proton_model, sum_channels_parallel,\
    calculate_joint_ws_across_experts, plot_expert_heatmap, plot_expert_specialization

print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)           # Check which CUDA version PyTorch was built with
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA Device 0: {torch.cuda.get_device_name(0)}")

DATA_IMAGES_PATH = "/net/tscratch/people/plgpbedkowski/data/data_proton_photonsum_proton_1_2312.pkl"
DATA_COND_PATH = "/net/tscratch/people/plgpbedkowski/data/data_cond_photonsum_proton_1_2312.pkl"
DATA_POSITIONS_PATH = "/net/tscratch/people/plgpbedkowski/data/data_coord_proton_photonsum_proton_1_2312.pkl"
INPUT_IMAGE_SHAPE = (56, 30)
MIN_PROTON_THRESHOLD = 1
N_COND, NOISE_DIM = 9, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
LR_G = 1e-4
LR_D = 1e-5
LR_A = 1e-4
LR_R = 1e-3
DI_STRENGTH = 0.1
IN_STRENGTH = 1e-3


def evaluate_model(model_dir, model_name, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Evaluating model '{model_name}' located at '{model_dir}'...")

    #
    # LOAD INPUT DATA
    #
    data = pd.read_pickle(DATA_IMAGES_PATH)
    data_cond = pd.read_pickle(DATA_COND_PATH)
    data_posi = pd.read_pickle(DATA_POSITIONS_PATH)
    data_cond_idx = data_cond[data_cond.proton_photon_sum >= MIN_PROTON_THRESHOLD].index
    data_cond = data_cond.loc[data_cond_idx].reset_index(drop=True)
    data = data[data_cond_idx]
    data_posi = data_posi.loc[data_cond_idx].reset_index(drop=True)
    photon_sum_proton_min, photon_sum_proton_max = data_cond.proton_photon_sum.min(), data_cond.proton_photon_sum.max()

    ### TRANSFORM DATA FOR TRAINING ###
    _, x_test, _, x_test_2, _, y_test, _, std_test, \
    _, intensity_test, _, positions_test, _, \
    expert_number_test, scaler_poz, data_cond_names, filepath_models = transform_data_for_training(
        data_cond, data,
        data_posi,
        model_dir,
        ZDCType.PROTON,
        SAVE_EXPERIMENT_DATA=False,
        load_data_file_from_checkpoint=False)

    dir_info = DIR_INFO.format(EXPERIMENT_DIR_NAME=model_dir)
    dir_models = DIR_MODELS.format(EXPERIMENT_DIR_NAME=model_dir)
    #
    # LOAD MODELS FOR EVALUATION
    #
    def get_generators_filenames(input_models_dir, epoch):
        pattern = re.compile(rf"^gen_(\d+)_{epoch}\.pth$")  # epoch is inserted dynamically into the regex
        matching_files = []
        for filename in os.listdir(input_models_dir):
            if pattern.match(filename):
                matching_files.append(filename)
        return matching_files

    generators_filenames = get_generators_filenames(dir_models, epoch)
    n_experts = len(generators_filenames)

    experts, _, _, _, _, _ = setup_experts(n_experts, N_COND, NOISE_DIM, LR_G, LR_D,
                                           LR_A, DI_STRENGTH, IN_STRENGTH, device)
    # router, _ = setup_router_attention(N_COND, n_experts, 4, 128, LR_R, device)
    router, _ = setup_router(N_COND, n_experts, LR_R, device)

    for i, filename in enumerate(generators_filenames):
        gen_file = os.path.join(os.path.join(dir_models, filename))
        experts[i] = torch.load(gen_file, map_location=device)
        # experts[i].load_state_dict(torch.load(gen_file, map_location=device))  # model saved as statedict
    router_file = os.path.join(dir_models, f"router_network_{epoch}.pth")
    print("Router path:", router_file)
    # router.load_state_dict(torch.load(router_file, map_location=device))
    router = torch.load(router_file, map_location=device)

    # Analyze ROUTER
    y_test_tensor = torch.tensor(y_test, requires_grad=False, device=device)

    with torch.no_grad():
        router.eval()
        predicted_expert_one_hot = router(y_test_tensor).cpu().numpy()
        predicted_expert = np.argmax(predicted_expert_one_hot, axis=1)

    expert_counts = np.unique(predicted_expert, return_counts=True)
    expert_number_count_dict = {expert_number: expert_count for (expert_number, expert_count) in
                                zip(expert_counts[0], expert_counts[1])}
    print(expert_number_count_dict)  # number of samples routed to each exper. TODO: Important to log

    #
    # Generate MEAN and STD of the experts on the test data
    #
    y_test_list = []
    indices_experts_list = []
    y_test_tensor_list = []
    for i in range(n_experts):
        indices_experts = np.where(predicted_expert == i)[0].tolist()
        y_test_temp = y_test[indices_experts]
        y_test_list.append(y_test_temp)
        indices_experts_list.append(indices_experts)
        y_test_tensor_list.append(torch.tensor(y_test_temp, requires_grad=False, device=device))

    photonsum_mean_generated_images = []
    photonsum_std_generated_images = []
    photonsum_on_all_generated_images = []
    for i in range(n_experts):
        noise_cond_temp = y_test_tensor_list[i]
        photonsum_mean_generated_images_expert, photonsum_std_generated_images_expert, \
        photonsum_on_all_generated_images_expert = get_mean_std_from_expert_genrations(
            noise_cond_temp, experts[i], device, batch_size=256, noise_dim=NOISE_DIM)
        photonsum_mean_generated_images.append(photonsum_mean_generated_images_expert)
        photonsum_std_generated_images.append(photonsum_std_generated_images_expert)
        photonsum_on_all_generated_images.append(photonsum_on_all_generated_images_expert)

    cond_expert_spec_plot = plot_expert_specialization(y_test, indices_experts_list, epoch, data_cond_names, save_path=dir_info)
    plt.close(cond_expert_spec_plot)

    # Produce plots of Distribution of mean of generated predictions from experts
    photonsum_dist_hist_plot = plot_proton_photonsum_histograms_shared(photonsum_on_all_generated_images, save_path=dir_info)
    plt.close(photonsum_dist_hist_plot)

    print(f"Mean values of generated images: {photonsum_mean_generated_images}")
    print(f"STD values of generated images: {photonsum_std_generated_images}")

    #
    # Calculate original channel values and WS
    #
    all_predictions = []
    ch_gen_experts = []
    x_test_list = []
    for i in range(n_experts):
        x_test_temp = x_test[np.where(predicted_expert == i)[0]]
        x_test_list.append(x_test_temp)

        # noise = torch.randn(len(y_test_tensor_list[i]), NOISE_DIM, device=device)  # TODO: when comparing to different models, the random noise is not the same
        results_temp = get_predictions_from_generator_results(BATCH_SIZE, len(y_test_tensor_list[i]), NOISE_DIM, device,
                                                              y_test_tensor_list[i], experts[i],
                                                              shape_images=INPUT_IMAGE_SHAPE)
        ch_gen_temp = pd.DataFrame(sum_channels_parallel(results_temp)).values if len(results_temp) else np.zeros((0, 5))

        ch_gen_experts.append(ch_gen_temp.copy())
        all_predictions.extend(results_temp)

    all_predictions = np.array(all_predictions)

    # CALCULATE DISTRIBUTION OF CHANNELS IN ORIGINAL TEST DATA #
    org = np.exp(x_test) - 1
    ch_org = np.array(org).reshape(-1, *INPUT_IMAGE_SHAPE)
    del org
    ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

    # calcualte ch_org for each expert
    ch_org_experts = []
    for i in range(n_experts):
        org_temp = np.exp(x_test_list[i]) - 1
        ch_org_expert_temp = np.array(org_temp).reshape(-1, *INPUT_IMAGE_SHAPE)
        del org_temp
        ch_org_expert_temp = pd.DataFrame(sum_channels_parallel(ch_org_expert_temp)).values
        ch_org_experts.append(ch_org_expert_temp.copy())

    for i in range(n_experts):
        fig = make_histograms(y_test_tensor_list[i], experts[i], i, ch_org=ch_org_experts[i], device=device,
                              save_path=dir_info)
        plt.close(fig)

    # CALCULATE WS
    ws_mean, ws_std, ws_mean_exp, ws_std_exp = calculate_joint_ws_across_experts(3,
                                                                                 x_test_list,
                                                                                 y_test_list,
                                                                                 experts,
                                                                                 ch_org,
                                                                                 ch_gen_experts,
                                                                                 NOISE_DIM, device,
                                                                                 batch_size=256,
                                                                                 n_experts=n_experts)
    print(f"WS mean: {ws_mean}, WS std: {ws_std}", "\n",
          "WS mean of experts:", ws_mean_exp, "\n"
          "WS std of experts:", ws_std_exp)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
    # parser.add_argument("--model_dir", type=str, required=True, help="Directory where the model is stored.")
    # parser.add_argument("--model_name", type=str, required=True, help="Name of the model being evaluated.")
    # parser.add_argument("--output_dir", type=str, default="./evaluation_results",
    #                     help="Directory to save evaluation results.")
    # args = parser.parse_args()

    # ATTENTION
    # model_dir = r"C:\Users\PB\Documents\GithubRepos\Generative-DNN-for-Physics-Simulations-CERN\dynamic_neural_networks\pytorch\dynamic_router\experiments\Adaptive load balancing\test_attention_router_differentiation_loss_23_02_2025_16_10_28_876"
    # epoch = 124
    # model_name = "example_model"
    # output_dir = "./evaluation_results"

    # model_dir = r"C:\Users\PB\Documents\GithubRepos\Generative-DNN-for-Physics-Simulations-CERN\dynamic_neural_networks\pytorch\dynamic_router\experiments\Adaptive load balancing\no_entropy_ijcai2025_05_0.001_18_02_2025_08_57_50_1_2312"
    # epoch = 73
    model_name = "example_model"
    output_dir = "evaluation_results"

    model_dir = r"/net/tscratch/people/plgpbedkowski/dynamic_neural_networks/experiments/test_alb_aux_different_gen_train_new_AUX_24_04_2025_00_02_42_472"
    output_dir = os.path.join(model_dir, output_dir)
    epoch = 249

    # Attention router
    # model_dir = r"C:\Users\PB\Documents\GithubRepos\Generative-DNN-for-Physics-Simulations-CERN\dynamic_neural_networks\pytorch\dynamic_router\experiments\Attention_Router\test_attention_router_adaptive_load_balancing_23_02_2025_11_23_56_261"
    # epoch = 111
     # evaluate_model(args.model_dir, args.model_name, args.output_dir)
    evaluate_model(model_dir, model_name, output_dir)
