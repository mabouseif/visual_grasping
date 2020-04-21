
def get_label_value(self, primitive_action, push_success, grasp_success, change_detected, prev_push_predictions, prev_grasp_predictions, next_color_heightmap, next_depth_heightmap):


    # Compute current reward
    current_reward = 0
    if primitive_action == 'push':
        if change_detected:
            current_reward = 0.5
    elif primitive_action == 'grasp':
        if grasp_success:
            current_reward = 1.0

    # Compute future reward
    if not change_detected and not grasp_success:
        future_reward = 0
    else:
        next_push_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True)
        future_reward = max(np.max(next_push_predictions), np.max(next_grasp_predictions))

    expected_reward = current_reward + self.future_reward_discount * future_reward
    print('Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
    
    return expected_reward, current_reward


# Compute labels and backpropagate
def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):


    # Compute labels
    label = np.zeros((1,320,320))
    action_area = np.zeros((224,224))
    action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
    # blur_kernel = np.ones((5,5),np.float32)/25
    # action_area = cv2.filter2D(action_area, -1, blur_kernel)
    tmp_label = np.zeros((224,224))
    tmp_label[action_area > 0] = label_value
    label[0,48:(320-48),48:(320-48)] = tmp_label

    # Compute label mask
    label_weights = np.zeros(label.shape)
    tmp_label_weights = np.zeros((224,224))
    tmp_label_weights[action_area > 0] = 1
    label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

    # Compute loss and backward pass
    self.optimizer.zero_grad()
    loss_value = 0


    # Do forward pass with specified rotation (to save gradients)
    push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

    if self.use_cuda:
        loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
    else:
        loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
    loss = loss.sum()
    loss.backward()
    loss_value = loss.cpu().data.numpy()

    opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

    push_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx)

    if self.use_cuda:
        loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
    else:
        loss = self.criterion(self.model.output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

    loss = loss.sum()
    loss.backward()
    loss_value = loss.cpu().data.numpy()

    loss_value = loss_value/2

    print('Training loss: %f' % (loss_value))
    self.optimizer.step()