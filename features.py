from data_load import *

def compute_rr_intervals(R_poses):
    features_RR = RR_intervals()
    # pre_R = np.array([], dtype=int)
    post_R = np.array([], dtype=int)
    local_R = np.array([], dtype=int)
    global_R = np.array([], dtype=int)
    # Pre_R and Post_R
    # pre_R = np.append(pre_R, 0)
    post_R = np.append(post_R, R_poses[1] - R_poses[0])
    for i in range(1, len(R_poses) - 1):
        # pre_R = np.append(pre_R, R_poses[i] - R_poses[i - 1])
        post_R = np.append(post_R, R_poses[i + 1] - R_poses[i])
    # pre_R[0] = pre_R[1]
    # pre_R = np.append(pre_R, R_poses[-1] - R_poses[-2])
    post_R = np.append(post_R, post_R[-1])
    # Local_R: AVG from last 10 pre_R values
    for i in range(0, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(-9, 1):
            if j + i >= 0:
                # avg_val = avg_val + pre_R[i + j]
                avg_val = avg_val + post_R[i + j]
                num = num + 1
        local_R = np.append(local_R, avg_val / float(num))
    # Global R AVG: from full past-signal
    # global_R = np.append(global_R, pre_R[0])
    global_R = np.append(global_R, post_R[0])
    for i in range(1, len(R_poses)):
        num = 0
        avg_val = 0
        for j in range(0, i):
            if (R_poses[i] - R_poses[j]) < 108000:
                # avg_val = avg_val + pre_R[j]
                avg_val = avg_val + post_R[j]
                num = num + 1
        # num = i
        global_R = np.append(global_R, avg_val / float(num))
    for i in range(0, len(R_poses)):
        # features_RR.pre_R = np.append(features_RR.pre_R, pre_R[i])
        features_RR.post_R = np.append(features_RR.post_R, post_R[i])
        features_RR.local_R = np.append(features_RR.local_R, local_R[i])
        features_RR.global_R = np.append(features_RR.global_R, global_R[i])

        # features_RR.append([pre_R[i], post_R[i], local_R[i], global_R[i]])
    return features_RR
