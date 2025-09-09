import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations
from expertsim.train.utils import calculate_expert_utilization_entropy, calculate_expert_distribution_loss,\
    calculate_adaptive_load_balancing_loss
from expertsim.train.utils import sum_channels_parallel, calculate_joint_ws_across_experts
import pandas as pd
import copy


class MoEWrapper(nn.Module):
    """
    Mixture of Experts wrapper that encapsulates the routing and expert logic.
    Keeps models separate from optimizers for clean architecture. Batch accumulation
    """
    # name = "separate-gen-disc-shared-aux-reg"
    name = "separate-gumbal-gen-disc-shared-aux-reg-masked-router-multiple-aux-reg"
    description = "MoEWrapper where expert is defined as a generator and discriminator." \
                  "Auxiliary regressor is each for every expert "

    def __init__(self, generator_cls, discriminator_cls, aux_reg_cls,
                 router_cls, n_experts: int, cfg,
                 image_shape: tuple = (56, 30)):
        super().__init__()
        self.image_shape = image_shape
        self.generators = nn.ModuleList([copy.deepcopy(generator_cls) for _ in range(n_experts)])
        self.discriminators = nn.ModuleList([copy.deepcopy(discriminator_cls) for _ in range(n_experts)])
        self.aux_regs = nn.ModuleList([copy.deepcopy(aux_reg_cls) for _ in range(n_experts)])
        # self.generators = nn.ModuleList([generator_cls for _ in range(n_experts)])
        # self.discriminators = nn.ModuleList([discriminator_cls for _ in range(n_experts)])
        # self.aux_regs = nn.ModuleList([aux_reg_cls for _ in range(n_experts)])
        self.router = router_cls
        self.n_experts = n_experts
        self.criterion = nn.BCELoss()
        self.noise_dim = cfg.model.noise_dim
        self.cfg = cfg

        self.g_steps = [0 for _ in range(n_experts)]
        self.d_steps = [0 for _ in range(n_experts)]

        for i in range(self.n_experts):
            for j in range(i + 1, self.n_experts):
                for (n1, p1), (n2, p2) in zip(self.generators[i].named_parameters(),
                                              self.generators[j].named_parameters()):
                    assert p1.data_ptr() != p2.data_ptr(), f"Gen params shared between {i} and {j}"

        print("====== Init completed ========")

    def train_step(self, epoch, cond, real_images, true_positions, std, intensity, aux_reg_optimizers,
                   generator_optimizers,
                   discriminator_optimizers, router_optimizer, ema_helper, device):

        B = cond.size(0)
        # Train router network

        # Get predicted experts assignments for samples. Outputs are the probabilities of each expert for each sample.
        # Shape: (batch, self.n_experts)

        tau0 = getattr(self.cfg.model.router, "tau_start")
        tau_min = getattr(self.cfg.model.router, "tau_min")
        tau_decay = getattr(self.cfg.model.router, "tau_decay")  # previously 0.995

        # tau0 = getattr(self.cfg.model.router, "tau_start", 1.2)
        # tau_min = getattr(self.cfg.model.router, "tau_min", 0.6)
        # tau_decay = getattr(self.cfg.model.router, "tau_decay", 0.98)  # per-epoch decay

        # Suggested if your goal is “more exploitation at 50+ epochs”:
        # more sharper routing

        # Compute current temperature
        tau = max(tau_min, tau0 * (tau_decay ** epoch))

        gates_soft, logits = self.router(cond, tau=tau, hard=False)  # gates_soft: [B,E], differentiable
        _, predicted_expert = torch.max(gates_soft, 1)  # (B, 1)

        # calculate the class counts for each expert
        # class_counts = torch.zeros(self.n_experts, dtype=torch.float).to(device)
        # for class_label in range(self.n_experts):
        #     class_counts[class_label] = (predicted_expert == class_label).sum().item()
        # class_counts_adjusted = class_counts / predicted_expert.size(0)  # now this hods for everyexpert the percentage of batch

        gen_losses = []
        disc_losses = []
        div_losses = np.zeros(self.n_experts)
        aux_reg_losses = np.zeros(self.n_experts)
        intensity_losses = np.zeros(self.n_experts)
        mean_intensities_experts = []  # mean intensities for each expert for each batch
        std_intensities_experts = []  # std intensities for each expert for each batch
        mean_intensities_in_batch_expert = torch.zeros((B, 1), device=device)  # this should contain leaf tensors, as we don't want routers from gradient to propagate to generators
        aux_reg_features_experts_in_batch_expert = torch.zeros((B, 64), device=device)

        # Straight-through trick: forward uses one-hot hard; backward uses soft
        # https://www.perplexity.ai/search/is-this-get-features-from-conv-.Dr8Sk4vSRiFe8Hpu3wJTA
        idx = gates_soft.argmax(dim=1)  # [B] expert index per sample (no grad path needed)

        class_counts = torch.bincount(idx, minlength=self.n_experts).to(real_images.dtype)  # [E] on GPU
        class_counts_adjusted = class_counts / B  # keep as GPU tensor throughout

        gates_hard = F.one_hot(idx, num_classes=self.n_experts).float()  # [B, E]
        gates = gates_hard + (gates_soft - gates_soft.detach())  # y = y_hard + (y_soft - y_soft.detach())
        # Use `gates` everywhere you need gradients to flow into the router;

        aux_reg_features_experts = []
        fake_logits_for_router = [None] * self.n_experts  # list of 1D tensors [B_e]
        idx_lists = [None] * self.n_experts  # list of index tensors (global batch indices per expert)
        expert_has_data = [False] * self.n_experts
        all_expert_rewards = []
        expert_masks = []
        # -------------------------
        # Zero ALL grads once at start
        # -------------------------
        for i in range(self.n_experts):
            aux_reg_optimizers[i].zero_grad(set_to_none=True)
            generator_optimizers[i].zero_grad(set_to_none=True)
            discriminator_optimizers[i].zero_grad(set_to_none=True)
        router_optimizer.zero_grad(set_to_none=True)

        for i in range(self.n_experts):
            # # mask = (gates_soft[:, i] > 0).nonzero(as_tuple=True)[0]  # this breaks the computation graph
            mask = (idx == i).nonzero(as_tuple=True)[0]  # hard selection for forward path
            # If no samples selected for this generator
            B_ex = mask.size(0)  # Number of samples assigned to this expert
            if B_ex <= 1:
                gen_losses.append(torch.tensor(0.0, requires_grad=True).to(device))
                disc_losses.append(torch.tensor(0.0, requires_grad=True).to(device))
                mean_intensities_experts.append(torch.tensor(0.0).to(device))
                std_intensities_experts.append(torch.tensor(0.0).to(device))

                feature_shape_aux_conv_channels = 64
                aux_reg_features_experts.append(
                    torch.zeros((1, feature_shape_aux_conv_channels), requires_grad=True).to(device))
                continue

            expert_has_data[i] = True
            #
            # Train discriminator
            #
            # Generator predictions that will be used to train discriminator and generator
            selected_generator = self.generators[i]
            selected_cond = cond[mask]
            noise_1 = torch.randn(B_ex, self.noise_dim, device=real_images.device)
            fake_images = selected_generator(noise_1, selected_cond)

            # Clone or detach tensors to avoid in-place modifications
            selected_discriminator = self.discriminators[i]
            selected_discriminator_optimizer = discriminator_optimizers[i]
            selected_real_images = real_images[mask]
            selected_class_counts = class_counts_adjusted[i]
            idx_lists[i] = mask

            disc_loss = self.discriminator_train_step(i, selected_discriminator, fake_images.detach(),
                                                      selected_discriminator_optimizer,
                                                      self.criterion,
                                                      selected_class_counts,
                                                      selected_real_images,
                                                      selected_cond, B_ex)
            disc_losses.append(disc_loss.detach())

            #
            # Train each generator
            #
            selected_cond = cond[mask]
            selected_true_positions = true_positions[mask]
            selected_intensity = intensity[mask]
            selected_std = std[mask]
            selected_generator = self.generators[i]
            selected_generator_optimizer = generator_optimizers[i]
            selected_discriminator = self.discriminators[i]
            selected_aux_reg = self.aux_regs[i]
            selected_aux_reg_optimizer = aux_reg_optimizers[i]
            selected_class_counts = class_counts_adjusted[i]

            gen_loss, div_loss, intensity_loss, \
            aux_reg_loss, std_intensity, mean_intensity, mean_intensities, aux_reg_features, fake_logits_for_g = self.generator_train_step(
                i,
                fake_images,
                noise_1,
                selected_generator,
                selected_discriminator,
                selected_aux_reg,
                selected_cond,
                selected_generator_optimizer,
                selected_aux_reg_optimizer,
                self.criterion,
                selected_true_positions,
                selected_std,
                selected_intensity,
                selected_class_counts,
                B_ex)
            fake_logits_for_router[i] = fake_logits_for_g.detach()

            aux_reg_features_experts_in_batch_expert[
                mask] = aux_reg_features.detach()
            mean_intensities_in_batch_expert[
                mask] = mean_intensities.detach()  # input the mean intensities for calculated samples.

            # Save statistics
            mean_intensities_experts.append(mean_intensity.detach())
            std_intensities_experts.append(std_intensity.detach())
            gen_losses.append(gen_loss.detach())
            # gen_losses[mask] = gen_loss.detach()
            div_losses[i] = div_loss
            intensity_losses[i] = intensity_loss
            aux_reg_losses[i] = aux_reg_loss

        #
        # Calculate router loss
        #

        if self.n_experts > 1:
            # router_optimizer.zero_grad(set_to_none=True)

            # weighted_gen_loss = (gen_losses.unsqueeze(1) * gates_soft).sum(dim=1).mean()
            # weighted_disc_loss = (disc_losses.unsqueeze(1) * gates_soft).sum(dim=1).mean()
            # gan_loss_scaled = (weighted_gen_loss + weighted_disc_loss) * self.cfg.model.router.gan_strength

            #
            # DISC INFO TYPE 2
            #

            # THIS CODE TAKES LONG TO COMPUTE

            # In router loss calculation:
            # Convert discriminator scores to rewards

            # rewards = torch.zeros(B, device=gates.device)
            # for e in range(self.n_experts):
            #     if expert_has_data[e]:
            #         expert_rewards = fake_logits_for_router[e].view(-1)
            #         all_expert_rewards.append(expert_rewards)
            #         expert_masks.append(idx_lists[e])
            #
            # # Normalize rewards across all experts
            # if all_expert_rewards:
            #     all_rewards_cat = torch.cat(all_expert_rewards)
            #     reward_mean = all_rewards_cat.mean()
            #     reward_std = all_rewards_cat.std() + 1e-8
            #
            #     # Apply normalized rewards
            #     for reward_idx, mask_e in enumerate(expert_masks):
            #         normalized_rewards = (all_expert_rewards[reward_idx] - reward_mean) / reward_std
            #         rewards[mask_e] = normalized_rewards
            #
            # # Policy gradient for routing (REINFORCE)
            # log_probs = torch.log(gates_soft + 1e-8)  # [B, E]
            # selected_log_probs = (log_probs * gates_hard).sum(dim=1)  # [B]
            # policy_loss = -(selected_log_probs * rewards.detach()).mean()
            # L_router_adv = policy_loss * self.cfg.model.router.gan_strength


            # genrator info
            gan_loss_scaled = torch.stack(gen_losses).mean() * self.cfg.model.router.gan_strength
            # gan_loss_scaled = (torch.stack(disc_losses).mean()) * self.cfg.model.router.gan_strength

            expert_entropy_loss = -1 * calculate_expert_utilization_entropy(gates_soft,
                                                                            self.cfg.model.router.util_strength) if self.cfg.model.router.util_strength != 0 else torch.tensor(
                0.0,
                requires_grad=False,
                device=gates_soft.device)

            expert_distribution_loss = calculate_expert_distribution_loss(gates,
                                                                          mean_intensities_in_batch_expert) if self.cfg.model.router.ed_strength != 0. else torch.tensor(
                0.0, requires_grad=False)

            expert_distribution_loss = expert_distribution_loss*self.cfg.model.router.ed_strength

            def compute_cross_generator_similarity(reps_i, reps_j):
                """
                Compute similarity between two generators' representations
                reps_i: (batch_i, 64), reps_j: (batch_j, 64)
                """
                # Normalize representations
                reps_i_norm = F.normalize(reps_i, dim=1)  # (batch_i, 64)
                reps_j_norm = F.normalize(reps_j, dim=1)  # (batch_j, 64)

                # Compute cosine similarity matrix
                similarity_matrix = torch.mm(reps_i_norm, reps_j_norm.t())  # (batch_i, batch_j)

                # Option 1: Mean of all pairwise similarities
                mean_similarity = similarity_matrix.mean()

                # Option 2: Maximum similarity (encourages generators to be different)
                # max_similarity = similarity_matrix.max()

                # Option 3: Top-k mean (focus on most similar pairs)
                # k = min(similarity_matrix.numel(), 10)
                # top_k_similarities = torch.topk(similarity_matrix.flatten(), k).values
                # top_k_mean = top_k_similarities.mean()

                return mean_similarity  # or mean_similarity or top_k_mean

            # def compute_differentiation_loss(aux_reg_features_experts):
            #     """
            #     representations_dict: {generator_id: tensor of shape (batch_size_i, 64)}
            #     Returns diversity loss encouraging different outputs between generators
            #     """
            #     total_loss = 0.0
            #     generator_pairs = 0
            #     num_experts = len(aux_reg_features_experts)
            #
            #     for i, j in combinations(range(num_experts), 2):
            #         gen_i_reps = aux_reg_features_experts[i]
            #         gen_j_reps = aux_reg_features_experts[j]
            #
            #         # Compute inter-generator similarity
            #         inter_gen_similarity = compute_cross_generator_similarity(
            #             gen_i_reps, gen_j_reps
            #         )
            #
            #         # Minimize inter-generator similarity
            #         total_loss += inter_gen_similarity
            #         generator_pairs += 1
            #
            #     return total_loss / generator_pairs if generator_pairs > 0 else 0.0

            # DIFFERENTIATION LOSS
            def compute_differentiation_loss(aux_reg_features_experts):
                """
                Compute differentiation loss for all experts based on the feature vectors from convolution layers.
                param: discriminator_features: List of tensors containing the features of each expert
                return: Differentiation loss
                """
                loss = torch.zeros(1, device=device)
                num_experts = len(aux_reg_features_experts)
                # print("N Experts", num_experts)
                with torch.no_grad():  # Detach computations from graph to save memory
                    feature_means = [feat.mean(0, keepdim=True) for feat in aux_reg_features_experts]
                    # feature_vars = [feat.var(dim=0) for feat in aux_reg_features_experts]

                # test to optimize experts, not router

                # print("shape means", feature_means[0].shape)
                # print("shape vars", feature_vars[0].shape)

                for i, j in combinations(range(num_experts), 2):
                    # Reattach to computation graph only for final loss calculation
                    mean_i = feature_means[i]
                    mean_j = feature_means[j]

                    # var_i = feature_vars[i].detach().requires_grad_(True)
                    # var_j = feature_vars[j].detach().requires_grad_(True)
                    # cos_diss_means = 1- F.cosine_similarity(mean_i, mean_j)
                    # cos_diss_vars = 1- F.cosine_similarity(mean_i, mean_j)
                    # dissimilarity = 0.8*cos_diss_means + 0.2*cos_diss_vars

                    cosine_similarity = F.cosine_similarity(mean_i, mean_j)
                    # -1: vectors dissimilar
                    # 0: vectors orthogonal
                    # 1: vectors
                    dissimilarity = torch.abs(cosine_similarity)
                    loss += dissimilarity
                return loss

            # differentiation_loss = compute_differentiation_loss(aux_reg_features_experts) if self.cfg.model.router.diff_strength != 0. else torch.tensor(0.0)
            # differentiation_loss = -differentiation_loss * self.cfg.model.router.diff_strength

            # differentiation_loss = compute_differentiation_loss(
            #     aux_reg_features_experts) if self.cfg.model.router.diff_strength != 0. else torch.tensor(0.0)
            # differentiation_loss = differentiation_loss * self.cfg.model.router.diff_strength

            # -------------------------------
            # Differentiation loss with soft gates
            # -------------------------------
            # https://www.perplexity.ai/search/deeply-analyze-the-code-here-a-1_ajYfIQT2uSOkObfNupjA
            # mean_intensities_experts is a list length E of scalar tensors (you appended .detach() earlier).
            # For router-only gradient, we can use these detached scalars.
            # We then weight each expert's mean by its average soft gate over the batch to build a
            # differentiable expert-mean proxy that depends on router parameters.
            # eps = 1e-8
            #
            # # Average soft gate per expert across batch, carries gradient to router
            # # shape: [E]
            # expert_gate_avgs = gates_soft.mean(dim=0)  # differentiable wrt router
            #
            # # Build gate-weighted expert means (detach means to avoid second backward through generator)
            # # This yields a tensor per expert whose gradient flows only via gate weights.
            # expert_means_weighted = []
            # for i in range(self.n_experts):
            #     # Broadcast the scalar mean to a tensor and multiply by the average gate
            #     # mean_intensities_experts[i] is a scalar tensor (detached above in your code)
            #     mi = mean_intensities_experts[i]  # router-only path
            #     # If mi might be 0D, ensure it’s a tensor with grad=False but on the right device
            #     mi = mi.to(device)
            #     # Weighted “proxy mean” for the expert: depends on expert_gate_avgs[i]
            #     proxy_mean_i = expert_gate_avgs[i] * mi
            #     expert_means_weighted.append(proxy_mean_i)
            #     # print(f"expert {i}")
            #     # print(expert_gate_avgs[i])
            #     # print(mi)


            differentiation_loss_intensities = sum(
                F.l1_loss(mean_intensities_experts[i].unsqueeze(0),
                          mean_intensities_experts[j].unsqueeze(0))
                for i, j in combinations(range(self.n_experts), 2)
            ) * self.cfg.model.router.diff_strength if self.cfg.model.router.diff_strength != 0. else torch.tensor(0.0)
            # differentiation_loss_stds = sum(
            #     torch.abs((std_intensities_experts[i] - std_intensities_experts[j]))
            #     for i, j in combinations(range(self.n_experts), 2)  # Generate all unique pairs of experts
            # ) if self.cfg.model.router.diff_strength != 0. else torch.tensor(0.0)

            differentiation_loss = -differentiation_loss_intensities * self.cfg.model.router.diff_strength

            alb_loss = calculate_adaptive_load_balancing_loss(gates_soft.sum(dim=0),
                                                              self.cfg.model.router.alb_strength) if self.cfg.model.router.alb_strength != 0. else torch.tensor(
                0.0)

            # Minimum weight for each component
            min_weight = self.cfg.model.router.min_weight
            start_epoch = 0
            end_epoch = self.cfg.model.router.alpha

            # Linear interpolation factor (0 -> prioritize distribution, 1 -> prioritize gen)
            # Linear interpolation factor (0 -> prioritize distribution, 1 -> prioritize gen)
            alpha = min(max((epoch - start_epoch) / (end_epoch - start_epoch), 0.0), 1.0)  # 1 at the beginning

            # Scale weights so they never drop below min_weight
            decreasing_weight = min_weight + (1.0 - min_weight) * alpha  # distribution weight
            increasing_weight = min_weight + (1.0 - min_weight) * (1.0 - alpha)  # generator weight

            # gan_loss_scaled = L_router_adv
            # reverse = increasing esl

            # no alpha
            # Weighted router loss
            router_loss = (
                    expert_distribution_loss +
                    gan_loss_scaled +
                    differentiation_loss +
                    expert_entropy_loss +
                    decreasing_weight*alb_loss
            )
            if epoch < self.cfg.model.router.stop_router_training_epoch:
                # Train Router Network
                router_loss.backward(retain_graph=False)  # delete the graph of the generators that were retained
                router_optimizer.step()
            else:
                logging.info("Router reached epoch training limits. Not stepping now!")
                router_loss = torch.tensor(0.0)
        else:
            gan_loss_scaled = torch.tensor(0.0)
            router_loss = torch.tensor(0.0)
            expert_distribution_loss = torch.tensor(0.0)
            differentiation_loss = torch.tensor(0.0)
            expert_entropy_loss = torch.tensor(0.0)
            alb_loss = torch.tensor(0.0)

        # gen_losses = [gen_loss.item() for gen_loss in gen_losses]
        # disc_losses = [disc_loss.item() for disc_loss in disc_losses]
        # class_counts = [class_count.item() for class_count in class_counts]
        # div_losses = [div_loss.item() for div_loss in div_losses]
        #
        # log_metrics = {
        #     'gen_loss': np.mean(gen_losses),
        #     'disc_loss': np.mean(disc_losses),
        #     'div_loss': np.mean(div_losses),
        #     'intensity_loss': np.mean(intensity_losses),
        #     'aux_reg_loss': np.mean(aux_reg_losses),
        #     'router_loss': router_loss.item(),
        #     'expert_distribution_loss': expert_distribution_loss.item(),
        #     'differentiation_loss': differentiation_loss.item(),
        #     'expert_entropy_loss': expert_entropy_loss.item(),
        #     'adaptive_load_balancing_loss': alb_loss.item(),
        #     'gan_loss': gan_loss_scaled.item(),
        #     **{f"gen_loss_{i}": gen_losses[i] for i in range(self.n_experts)},
        #     **{f"disc_loss_{i}": disc_losses[i] for i in range(self.n_experts)},
        #     **{f"div_loss_experts_{i}": div_losses[i] for i in range(self.n_experts)},
        #     **{f"intensity_loss_experts_{i}": intensity_losses[i] for i in range(self.n_experts)},
        #     **{f"aux_reg_loss_experts_{i}": aux_reg_losses[i] for i in range(self.n_experts)},
        #     **{f"std_intensities_experts_{i}": std_intensities_experts[i].cpu() for i in range(self.n_experts)},
        #     **{f"mean_intensities_experts_{i}": mean_intensities_experts[i].cpu() for i in range(self.n_experts)},
        #     **{f"n_choosen_experts_mean_epoch_{i}": class_counts[i] for i in range(self.n_experts)},
        # }
        #
        # return log_metrics

        log_metrics = {
            'gen_loss': torch.stack([g for g in gen_losses if isinstance(g, torch.Tensor) and g.numel() > 0]).mean(),
            'disc_loss': torch.stack([d for d in disc_losses if isinstance(d, torch.Tensor) and d.numel() > 0]).mean(),
            'div_loss': torch.tensor(np.mean(div_losses), device=device),  # numpy to tensor
            'intensity_loss': torch.tensor(np.mean(intensity_losses), device=device),
            'aux_reg_loss': torch.tensor(np.mean(aux_reg_losses), device=device),
            'router_loss': router_loss,  # Keep as tensor
            'expert_distribution_loss': expert_distribution_loss,
            'differentiation_loss': differentiation_loss,
            'expert_entropy_loss': expert_entropy_loss,
            'adaptive_load_balancing_loss': alb_loss,
            'gan_loss': gan_loss_scaled,
            **{f"gen_loss_{i}": gen_losses[i] for i in range(self.n_experts)},
            **{f"disc_loss_{i}": disc_losses[i] for i in range(self.n_experts)},
            **{f"div_loss_experts_{i}": div_losses[i] for i in range(self.n_experts)},
            **{f"intensity_loss_experts_{i}": intensity_losses[i] for i in range(self.n_experts)},
            **{f"aux_reg_loss_experts_{i}": aux_reg_losses[i] for i in range(self.n_experts)},
            **{f"std_intensities_experts_{i}": std_intensities_experts[i] for i in range(self.n_experts)},
            # Keep as tensor
            **{f"mean_intensities_experts_{i}": mean_intensities_experts[i] for i in range(self.n_experts)},
            # Keep as tensor
            **{f"n_choosen_experts_mean_epoch_{i}": class_counts[i] for i in range(self.n_experts)},  # Keep as tensor
        }

        return log_metrics

    def discriminator_train_step(self, i, disc, fake_images, d_optimizer, criterion, class_counts, real_images, cond, batch_size) -> np.float32:
        """Returns Python float of disc_loss value"""
        # Train discriminator
        # d_optimizer.zero_grad(set_to_none=True)

        # calculate loss for real images
        real_output, real_latent = disc(real_images, cond)

        # calculate loss for generated images
        fake_output, fake_latent = disc(fake_images, cond)

        # Hinge loss for D
        loss_real = F.relu(1.0 - real_output).mean()
        loss_fake = F.relu(1.0 + fake_output).mean()
        disc_loss = loss_real + loss_fake

        w = float(class_counts)
        disc_loss = disc_loss*w
        disc_loss.backward()

        d_optimizer.step()
        return disc_loss

    def generator_train_step(self, i, fake_images, noise_1, generator, discriminator, a_reg, cond, g_optimizer,
                             a_optimizer, criterion, true_positions, std, intensity, class_counts, batch_size):
        # Train Generator
        # g_optimizer.zero_grad(set_to_none=True)
        # a_optimizer.zero_grad(set_to_none=True)

        noise_2 = torch.randn(batch_size, self.noise_dim, device=cond.device)

        # generate fake images
        fake_images_2 = generator(noise_2, cond)

        # validate two images
        fake_output, fake_latent = discriminator(fake_images, cond)  # don't detach, so gradients flow back to generator
        fake_output_2, fake_latent_2 = discriminator(fake_images_2, cond)

        gen_loss = -fake_output.mean()  # hinge generator loss
        # gen_loss = criterion(fake_output, torch.ones_like(fake_output))

        div_loss = self.sdi_gan_regularization(fake_latent, fake_latent_2,
                                               noise_1, noise_2,
                                               std, generator.di_strength)

        intensity_loss, mean_intenisties, std_intensity, mean_intensity = self.intensity_regularization(fake_images,
                                                                                                        intensity,
                                                                                                        generator.in_strength)

        gen_loss = gen_loss + div_loss + intensity_loss

        generated_positions = a_reg(fake_images)

        aux_reg_loss = a_reg.regressor_loss(true_positions, generated_positions) * self.cfg.model.aux_reg.strength

        gen_loss += aux_reg_loss
        w = float(class_counts)
        gen_loss = gen_loss*w
        gen_loss.backward(retain_graph=False)  # only when the router also requires gradients to bass back from it
        g_optimizer.step()
        a_optimizer.step()

        # aux_reg_features = a_reg.feature_extractor(fake_images.detach())
        aux_reg_features = torch.zeros(batch_size, 64, device=fake_images.device)  # Dummy tensor
        return gen_loss, div_loss, intensity_loss, aux_reg_loss, std_intensity, mean_intensity, \
               mean_intenisties, aux_reg_features, fake_output

    @staticmethod
    def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, DI_STRENGTH):
        # Calculate the absolute differences and their means along the batch dimension
        abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=1)
        abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=1)

        # Compute the division term
        div = abs_diff_latent / (abs_diff_noise + 1e-5)

        # Calculate the div_loss
        div_loss = std / (div + 1e-5)

        # Calculate the final div_loss
        div_loss = torch.mean(std) * torch.mean(div_loss)

        return div_loss*DI_STRENGTH

    @staticmethod
    def intensity_regularization(gen_im_proton, intensity_proton, IN_STRENGTH):
        """
        Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
        and the mean and standard deviation of the intensity across the batch.

        Args:
            gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
            intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
            IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.

        Returns:
            torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
                          sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
            torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
            torch.Tensor: The standard deviation of the scaled intensity values across the batch.
            torch.Tensor: The mean of the scaled intensity values across the batch.
        """

        # Sum the intensities in the generated images
        # gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1 #<- this fixed previous bad optimization
        gen_im_proton_rescaled = torch.exp(gen_im_proton) - 1
        # print("Gen shape from model", gen_im_proton_rescaled.shape)
        # Gen shape from model torch.Size([138, 1, 56, 30])
        # After sum: torch.Size([138, 1, 1, 1])
        sum_all_axes_p_rescaled = torch.sum(gen_im_proton_rescaled.clone(), dim=[2, 3],
                                            keepdim=False)  # Sum along all but batch dimension

        # print(sum_all_axes_p_rescaled.shape)  # (batch_size_current, 1)
        # print(sum_all_axes_p_rescaled)
        # REMOVE THIS RESHAPE BECAUSE IT FLATTENS THE DATA FROM ALL EXPERTS
        # sum_all_axes_p_rescaled = sum_all_axes_p_rescaled.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

        # Compute mean and std as PyTorch tensors
        std_intensity_scaled = sum_all_axes_p_rescaled.std()
        mean_intensity_scaled = sum_all_axes_p_rescaled.mean()  # mean intensity of all images
        # print('---------------')
        # print(mean_intensity_scaled.shape)
        # print('---------------')
        # # Ensure intensity_proton is correctly shaped and on the same device
        intensity_proton = intensity_proton.view(-1, 1).to(
            gen_im_proton.device)  # Ensure it is of shape [batch_size, 1]

        # apply the MASK AS WELL FOR EXPERT COMPUTATIONS TO BOTH THE GENERATED AND REAL DATA
        # OR MAYBE CALCULATE THIS N_EXPERT times each for separate expert. TRY TO MAKE THIS PARALLEL

        # print('shape sum_all_axes_p_rescaled', sum_all_axes_p_rescaled.shape)
        # print('shape intensity_proton',intensity_proton.shape)
        assert sum_all_axes_p_rescaled.shape == intensity_proton.shape
        # Calculate MAE loss
        mae_value_p = F.l1_loss(sum_all_axes_p_rescaled, intensity_proton) * IN_STRENGTH

        return mae_value_p, sum_all_axes_p_rescaled, std_intensity_scaled, mean_intensity_scaled

    @torch.no_grad()
    def evaluate(self, epoch, y_test, x_test, true_positions, std, intensity, cfg, device):
        ch_org = np.expm1(x_test)  # Original channels
        ch_org = np.array(ch_org).reshape(-1, *cfg.dataset.input_image_shape)
        ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

        soft_gates, logits = self.router(y_test)
        _, predicted_expert = torch.max(soft_gates, 1)

        indices_experts = [np.where(predicted_expert.cpu().numpy() == i)[0] for i in range(self.n_experts)]

        # Process expert indices dynamically
        ch_org_experts = []
        for i in range(self.n_experts):
            if len(indices_experts[i]) > 0:
                org = np.expm1(x_test[indices_experts[i]])
                ch_org_exp = np.array(org).reshape(-1, *self.image_shape)
                del org
                ch_org_exp = pd.DataFrame(sum_channels_parallel(ch_org_exp)).values
            else:
                ch_org_exp = np.zeros((len(indices_experts[i]), 5))
            ch_org_experts.append(ch_org_exp)

        y_test_experts = [y_test[indices_experts[i]] for i in range(self.n_experts)]
        # Calculate WS distance across all distribution
        ws_mean, ws_std, ws_mean_exp, ws_std_exp = calculate_joint_ws_across_experts(
            min(epoch // 5 + 1, 5),
            [x_test[indices_experts[i]] for i in range(self.n_experts)],
            y_test_experts,
            self.generators,
            ch_org,  # transformed to original input space, no log
            ch_org_experts,
            self.noise_dim, device,
            n_experts=self.n_experts,
            shape_images=cfg.dataset.input_image_shape)

        # Log to WandB tool
        log_data = {
            'ws_mean': ws_mean,
            **{f"ws_mean_{i}": ws_mean_exp[i] for i in range(self.n_experts)},
            'ws_std': ws_std,
            **{f"ws_std_{i}": ws_std_exp[i] for i in range(self.n_experts)},
            'epoch': epoch
        }
        for i in range(self.n_experts):
            log_data[f"ws_mean_{i}"] = ws_mean_exp[i]
            log_data[f"ws_std_{i}"] = ws_std_exp[i]

        return log_data

    def get_expert_assignment_counts(self, expert_assignments: torch.Tensor) -> torch.Tensor:
        """Get count of samples assigned to each expert."""
        class_counts = torch.zeros(self.n_experts, dtype=torch.float, device=expert_assignments.device)
        for expert_idx in range(self.n_experts):
            class_counts[expert_idx] = (expert_assignments == expert_idx).sum().item()
        return class_counts / expert_assignments.size(0)
