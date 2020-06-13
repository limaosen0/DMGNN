import numpy as np
import random
import copy


def read_txt_as_data(filename):
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))
    returnArray = np.array(returnArray)
    return returnArray



def define_actions(action):
    actions = ["walking", "eating", "smoking", "discussion",  "directions",
               "greeting", "phoning", "posing", "purchases", "sitting", "sittingdown", 
               "takingphoto", "waiting", "walkingdog", "walkingtogether"]
    if action in actions:
        return [action]
    elif action == "all":
        return actions
    elif action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]
    else:
        raise( ValueError, "Unrecognized action: %d" % action )



def load_data(data_path, subjects, actions):
    nactions = len(actions)
    sampled_data_set, complete_data = {}, []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            for subact in [1, 2]:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(data_path, subj, action, subact)
                action_sequence = read_txt_as_data(filename)
                t, d = action_sequence.shape
                even_indices = range(0, t, 2)
                sampled_data_set[(subj, action, subact, 'even')] = action_sequence[even_indices, :]
                if len(complete_data) == 0:
                    complete_data = copy.deepcopy(action_sequence)
                else:
                    complete_data = np.append(complete_data, action_sequence, axis=0)
    return sampled_data_set, complete_data



def normalization_stats(complete_data):
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    dimensions_is_zero = []
    dimensions_is_zero.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_nonzero = []
    dimensions_nonzero.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_is_zero] = 1.0

    dim_to_ignore = [0,  1,  2,  3,  4,  5,  18, 19, 20, 33, 34, 35, 48, 49, 50, 63, 64, 65,
                     66, 67, 68, 69, 70, 71, 72, 73, 74, 87, 88, 89,
                     90, 91, 92, 93, 94, 95, 96, 97, 98]
    dim_to_use = [6,  7,  8,  9,  10, 11, 12, 13, 14,
                  15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                  36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53,
                  54, 55, 56, 57, 58, 59, 60 ,61, 62, 75, 76, 77, 78, 79, 80,
                  81, 82, 83, 84, 85, 86]
    return data_mean, data_std, dim_to_ignore, dim_to_use, dimensions_is_zero, dimensions_nonzero



def normalize_data(data_set, data_mean, data_std, dim_to_use):
    data_out = {}
    for key in data_set.keys():
        data_out[key] = np.divide((data_set[key]-data_mean), data_std)
        data_out[key] = data_out[key][:,dim_to_use]
    return data_out



def train_sample(data_set, batch_size, source_seq_len, target_seq_len, input_size):

    all_keys = list(data_set.keys())
    chosen_keys_idx = np.random.choice(len(all_keys), batch_size)
    total_seq_len = source_seq_len + target_seq_len

    encoder_inputs  = np.zeros((batch_size, source_seq_len-1, input_size), dtype=np.float32)
    decoder_inputs  = np.zeros((batch_size, 1, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)

    for i in range(batch_size):
        the_key = all_keys[chosen_keys_idx[i]]
        t, d = data_set[the_key].shape
        idx = np.random.randint(16, t-total_seq_len)
        data_sel = data_set[the_key][idx:idx+total_seq_len,:]

        encoder_inputs[i,:,:]  = data_sel[0:source_seq_len-1,:]
        decoder_inputs[i,:,:]  = data_sel[source_seq_len-1:source_seq_len,:]
        decoder_outputs[i,:,:] = data_sel[source_seq_len:,:]

    rs = int(np.random.uniform(low=0, high=4))
    downsample_idx = np.array([int(i)+rs for i in [np.floor(j*4) for j in range(12)]])

    return encoder_inputs, decoder_inputs, decoder_outputs, downsample_idx



def find_indices_srnn(data_set, action):
    seed = 1234567890
    rng = np.random.RandomState(seed)
    subject = 5
    T1 = data_set[(subject, action, 1, 'even')].shape[0]
    T2 = data_set[(subject, action, 2, 'even')].shape[0]
    prefix, suffix = 50, 100

    idx = []
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    idx.append(rng.randint(16, T1-prefix-suffix))
    idx.append(rng.randint(16, T2-prefix-suffix))
    return idx



def srnn_sample(data_set, action, source_seq_len, target_seq_len, input_size):
    frames = {}
    frames[action] = find_indices_srnn(data_set, action)
    batch_size, subject = 8, 5
    seeds = [(action, (i%2)+1, frames[action][i]) for i in range(batch_size)]

    encoder_inputs = np.zeros((batch_size, source_seq_len-1, input_size), dtype=np.float32)
    decoder_inputs = np.zeros((batch_size, 1, input_size), dtype=np.float32)
    decoder_outputs = np.zeros((batch_size, target_seq_len, input_size), dtype=np.float32)
    for i in range(batch_size):
        _, subsequence, idx = seeds[i]
        idx = idx+source_seq_len
        data_sel = data_set[(subject, action, subsequence, 'even')]
        data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len), :]
        encoder_inputs[i, :, :] = data_sel[0:source_seq_len-1, :]
        decoder_inputs[i, :, :] = data_sel[source_seq_len-1:source_seq_len, :]
        decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]

    rs = int(np.random.uniform(low=0, high=4))
    downsample_idx = np.array([int(i)+rs for i in [np.floor(j*4) for j in range(12)]])

    return encoder_inputs, decoder_inputs, decoder_outputs, downsample_idx



def expmap2rotmat(r):
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta+np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
    r0x = r0x-r0x.T
    R = np.eye(3,3)+np.sin(theta)*r0x+(1-np.cos(theta))*(r0x).dot(r0x)
    return R



def rotmat2euler(R):
    if R[0,2]==1 or R[0,2]==-1:
        e3 = 0
        dlta = np.arctan2(R[0,1], R[0,2])
        if R[0,2]==-1:
            e2 = np.pi/2
            e1 = e3+dlta
        else:
            e2 = -1*np.pi/2
            e1 = -1*e3+dlta
    else:
        e2 = -1*np.arcsin(R[0,2])
        e1 = np.arctan2(R[1,2]/np.cos(e2), R[2,2]/np.cos(e2))
        e3 = np.arctan2(R[0,1]/np.cos(e2), R[0,0]/np.cos(e2))
    eul = np.array([e1, e2, e3])
    return eul



def unnormalize_data(data, data_mean, data_std, dim_ignore, dim_use, dim_zero):
    t, d = data.shape[0], data_mean.shape[0]   # t = 25, d = 99
    orig_data = np.zeros((t, d), dtype=np.float32)    # [25, 99]
    mask = np.ones((t, d), dtype=np.float32)
    dim_use, dim_zero = np.array(dim_use), np.array(dim_zero)                       #  66
    orig_data[:,dim_use] = data
    mask[:,dim_zero] = 0
    orig_data = np.multiply(orig_data, mask)

    std_mat = np.repeat(data_std.reshape((1,d)), t, axis=0)
    mean_mat = np.repeat(data_mean.reshape((1,d)), t, axis=0)
    orig_data = np.multiply(orig_data, std_mat)+mean_mat
    return orig_data



def get_srnn_gts(actions, data_set, data_mean, data_std, dim_to_ignore, 
                 source_seq_len, target_seq_len, input_size, to_euler=True):
    srnn_gts = {}
    for action in actions:
        srnn_gt = []
        encoder_inputs, decoder_inputs, targets = srnn_sample(data_set, action, source_seq_len, target_seq_len, input_size)
        for i in np.arange(targets.shape[0]):
            target = targets[i,:,:]
            if to_euler:
                for j in np.arange(target.shape[0]):
                    for k in np.arange(3, 97, 3):
                        target[j,k:k+3] = rotmat2euler(expmap2rotmat(target[j,k:k+3]))
            srnn_gt.append(target)
        srnn_gts[action] = srnn_gt
    return srnn_gts


def quat2expmap(q):
    if (np.abs(np.linalg.norm(q)-1)>1e-3):
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta, coshalftheta = np.linalg.norm(q[1:]), q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2*np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta+2*np.pi, 2*np.pi)

    if theta > np.pi:
        theta =  2*np.pi-theta
        r0 = -r0
    r = r0 * theta
    return r


def rotmat2quat(R):
    rotdiff = R-R.T;
    r = np.zeros(3)
    r[0] = -rotdiff[1,2]
    r[1] =  rotdiff[0,2]
    r[2] = -rotdiff[0,1]
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)
    sintheta = np.linalg.norm(r)/2
    costheta = (np.trace(R)-1)/2;
    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta/2)
    q[1:] = r0*np.sin(theta/2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def _some_variables():
    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                    17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1
    offset = np.array([0.000000,     0.000000,     0.000000,
                       -132.948591,  0.000000,     0.000000,
                       0.000000,    -442.894612,   0.000000,
                       0.000000,    -454.206447,   0.000000,
                       0.000000,     0.000000,     162.767078,
                       0.000000,     0.000000,     74.999437,
                       132.948826,   0.000000,     0.000000,
                       0.000000,    -442.894413,   0.000000,
                       0.000000,    -454.206590,   0.000000,
                       0.000000,     0.000000,     162.767426,
                       0.000000,     0.000000,     74.999948,
                       0.000000,     0.100000,     0.000000,
                       0.000000,     233.383263,   0.000000,
                       0.000000,     257.077681,   0.000000,
                       0.000000,     121.134938,   0.000000,
                       0.000000,     115.002227,   0.000000,
                       0.000000,     257.077681,   0.000000,
                       0.000000,     151.034226,   0.000000,
                       0.000000,     278.882773,   0.000000,
                       0.000000,     251.733451,   0.000000,
                       0.000000,     0.000000,     0.000000,
                       0.000000,     0.000000,     99.999627,
                       0.000000,     100.000188,   0.000000,
                       0.000000,     0.000000,     0.000000,
                       0.000000,     257.077681,   0.000000,
                       0.000000,     151.031437,   0.000000,
                       0.000000,     278.892924,   0.000000,
                       0.000000,     251.728680,   0.000000,
                       0.000000,     0.000000,     0.000000,
                       0.000000,     0.000000,     99.999888,
                       0.000000,     137.499922,   0.000000,
                       0.000000,     0.000000,     0.000000]).reshape(-1,3)
    rotInd = [[5,  6,  4],  [8,  9,  7],  [11, 12, 10], [14, 15, 13], [17, 18, 16], [],
              [20, 21, 19], [23, 24, 22], [26, 27, 25], [29, 30, 28],               [],
              [32, 33, 31], [35, 36, 34], [38, 39, 37], [41, 42, 40],               [],
              [44, 45, 43], [47, 48, 46], [50, 51, 49], [53, 54, 52], [56, 57, 55], [],
              [59, 60, 58],                                                         [],
              [62, 63, 61], [65, 66, 64], [68, 69, 67], [71, 72, 70], [74, 75, 73], [],
              [77, 78, 76],                                                         []]
    expmapInd = np.split(np.arange(4,100)-1,32)
    return parent, offset, rotInd, expmapInd


def revert_coordinate_space(channels, R0, T0):
    n, d = channels.shape
    channels_rec = copy.copy(channels)
    R_prev, T_prev = R0, T0
    rootRotInd = np.arange(3,6)

    for ii in range(n):
        R_diff = expmap2rotmat(channels[ii, rootRotInd])
        R = R_diff.dot(R_prev)
        channels_rec[ii, rootRotInd] = rotmat2expmap(R)
        T = T_prev + ((R_prev.T).dot(np.reshape(channels[ii,:3],[3,1]))).reshape(-1)
        channels_rec[ii,:3] = T
        T_prev, R_prev = T, R
    return channels_rec


def fkl(angles, parent, offset, rotInd, expmapInd):
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):
        if not rotInd[i] :
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[ rotInd[i][0]-1 ]
            yangle = angles[ rotInd[i][1]-1 ]
            zangle = angles[ rotInd[i][2]-1 ]

        r = angles[expmapInd[i]]
        thisRotation = expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i,:], (1,3))+thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i,:]+thisPosition).dot(xyzStruct[parent[i]]['rotation'])+xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    xyz = xyz[:,[0,2,1]]
    return np.reshape(xyz,[-1])


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot