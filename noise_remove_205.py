from my_func import *

df = pd.read_csv('205.csv')
data = df['\'MLII\''].values
data_to_process = data[105000:115000]
data_norm = data_to_process - np.mean(data_to_process)

# noise-tolerance (no strict fidelity enforcement)
tau = 1
# no DC part imposed
DC = 0
# initialize omegas uniformly
init = 1
# Error allowed
tol = 1e-6

for Alpha in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
              17000, 18000]:
    for k in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        alpha = Alpha
        K = k
        IMF_VMD, hat, omega = VMD(data_norm, alpha, tau, K, DC, init, tol)
        corrected_baseline = np.sum(IMF_VMD[1:], axis=0)
        plt.close()
        fig = plt.figure(figsize=[18, 10])
        plt.plot(data_norm, label='Clean ECG')
        plt.plot(IMF_VMD[0], label='Baseline ECG')
        plt.plot(corrected_baseline, label='Corrected Baseline')
        plt.legend()
        plt.title(str(alpha) + '  ' + str(k))
        plt.savefig('Images' + str(alpha) + '_' + str(k) + '.png', dpi=250)
