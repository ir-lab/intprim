import numpy as np
import numpy.random
from .. import basis_model
from .. import bayesian_interaction_primitives as bip

def spatial_robustness():
    np.random.seed(213413414)

    # Initialize a BIP object with 2 DOF named "X" and "Y" which are approximated by 8 basis functions.
    primitive = bip.BayesianInteractionPrimitive(2, ['X', 'Y'], 8)

    # A single example of a handwriting trajectory
    xdata = np.array([
        2.52147861,  2.68261873,  2.84009521,  2.99269205,  3.13926385,
        3.27876056,  3.41025573,  3.5329778 ,  3.64634321,  3.74998937,
        3.8438048 ,  3.92795314,  4.00288777,  4.0693539 ,  4.12837543,
        4.18122498,  4.22937664,  4.27444203,  4.31809201,  4.36196737,
        4.40758299,  4.4562309 ,  4.50888808,  4.56613502,  4.62809093,
        4.69437067,  4.76406782,  4.83576665,  4.90758435,  4.97724312,
        5.04216954,  5.099617  ,  5.14680484,  5.18106677,  5.1999997 ,
        5.20160394,  5.18440564,  5.14755368,  5.09088427,  5.01494897,
        4.92100438,  4.8109641 ,  4.68731662,  4.55301474,  4.41134412,
        4.26577973,  4.11983926,  3.97694226,  3.84028296,  3.71272292,
        3.59670796,  3.4942117 ,  3.4067061 ,  3.33515726,  3.28004369,
        3.24139282,  3.21883106,  3.21164261,  3.21883242,  3.23918946,
        3.27134723,  3.31383944,  3.36515007,  3.42375745,  3.48817336,
        3.55697803,  3.62885243,  3.70260907,  3.77722187,  3.85185522,
        3.92589153,  3.99895578,  4.07093474,  4.14198835,  4.21255021,
        4.2833145 ,  4.35520693,  4.42933819,  4.50693958,  4.5892814 ,
        4.67757669,  4.7728736 ,  4.87594169,  4.98715824,  5.10640159,
        5.23295916,  5.36545793,  5.50182437,  5.63928031,  5.7743792 ,
        5.90308534,  6.02089593,  6.12300271,  6.20448725,  6.26054043,
        6.28669463,  6.27905489,  6.23451447,  6.15094057,  6.02731681
    ])
    ydata = np.array([
        2.60877965,  2.76485925,  2.91587601,  3.06074461,  3.19850088,
        3.32832259,  3.44955237,  3.56172269,  3.66458245,  3.75812375,
        3.84260667,  3.9185795 ,  3.98689125,  4.04869382,  4.10543106,
        4.1588132 ,  4.21077576,  4.26342334,  4.31895999,  4.37960871,
        4.44752397,  4.52470161,  4.61289081,  4.71351323,  4.82759375,
        4.95570667,  5.09794052,  5.25388323,  5.42262803,  5.60279957,
        5.79259769,  5.98985598,  6.19211079,  6.39667626,  6.60072087,
        6.80134129,  6.99563046,  7.18073763,  7.35391969,  7.51258424,
        7.6543261 ,  7.77695956,  7.87854902,  7.95744025,  8.0122939 ,
        8.0421214 ,  8.0463223 ,  8.0247204 ,  7.97759496,  7.90570262,
        7.81028529,  7.69306011,  7.55618819,  7.40222104,  7.23402506,
        7.05468668,  6.86740265,  6.67536129,  6.48162182,  6.28899902,
        6.09996034,  5.916542  ,  5.74028898,  5.57222266,  5.41283782,
        5.26212897,  5.11964415,  4.98456294,  4.85579367,  4.73208409,
        4.61213865,  4.49473531,  4.37883468,  4.26367447,  4.14884334,
        4.0343288 ,  3.9205359 ,  3.80827461,  3.69871613,  3.59332021,
        3.49373739,  3.40169213,  3.31885379,  3.24670384,  3.18640788,
        3.13870115,  3.10379544,  3.08131435,  3.07026211,  3.06902906,
        3.07543489,  3.08680804,  3.10009753,  3.11201102,  3.11917145,
        3.1182827 ,  3.10629444,  3.08055594,  3.03894936,  2.97999426
    ])

    phase_velocities = []
    translation_variance = 4.0
    train_noise_variance = 1.0

    # Add 30 demonstrations which are generated from the writing sample
    for demo in range(30):
        # Apply a random translation
        translation_noise = np.random.normal(0.0, translation_variance)
        train_trajectory = np.array([xdata + translation_noise, ydata + translation_noise])
        # Apply an additional element-wise noise on top, then add this trajectory as a demonstration.
        primitive.add_demonstration(np.random.normal(train_trajectory, train_noise_variance))
        # Keep track of the phase velocity for each full demonstration. We'll use it later when testing.
        phase_velocities.append(1.0 / train_trajectory.shape[1])

    test_trajectory = np.array([
        xdata - translation_variance * 0.8,
        # Note that we're translating by the variance to simulate an extreme case, rather than sampling from a normal distribution.
        ydata - translation_variance * 0.8
    ])

    test_noise_variance = 1.0
    test_noise = np.random.normal(np.zeros(test_trajectory.shape), test_noise_variance)

    for test_noise_weight in np.linspace(0.01, 1.0, 10):
        test_trajectory_noisy = test_trajectory + (test_noise * test_noise_weight)

        test_trajectory_partial = np.array(test_trajectory_noisy, copy = True)
        test_trajectory_partial[0, :] = 0.0;

        # Generating the observation noise matrix.
        # The variance for the X DOF is a very high value so that the observed x-values are not heavily weighted in conditional inference.
        # This is desirable because we don't have observed x-values, we want to generate them based on the observed y-values.
        # This mimics an HRI scenario in which we only observe one agent's actions and want to generate the other agent's actions in response.
        observation_noise = np.array([[100000.0, 0.0], [0.0, test_noise_variance * test_noise_weight]])

        observable_samples = 30
        primitive.initialize_filter(phase_velocity = np.mean(phase_velocities), phase_var = np.var(phase_velocities))
        gen_trajectory, phase = primitive.generate_probable_trajectory_recursive(test_trajectory_partial[:, :observable_samples], observation_noise, num_samples = 100 - observable_samples)

        mean_trajectory = primitive.get_mean_trajectory()

        primitive.plot_partial_trajectory(gen_trajectory, test_trajectory_noisy[:, :observable_samples], mean_trajectory)
