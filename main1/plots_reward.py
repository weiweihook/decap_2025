import numpy as np
import matplotlib.pyplot as plt

def smooth(y, pts):
    b = np.ones(pts)/pts
    y_smooth = np.convolve(y, b, mode='valid')
    return y_smooth


def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    return data

if __name__ == "__main__":
    path1 = ''
    path2 = 'case7/'
    r1 = np.array(loadtxtmethod('runs/'+path1+path2+'202410302012/reward.txt'))[0:600]
    r2 = np.array(loadtxtmethod('infer/'+path1+path2+'202410301330/reward.txt'))[0:600]
    rm1 = smooth(np.mean(r1, axis=1),4)[5:505]
    rm2 = smooth(np.mean(r2, axis=1),4)[5:505]

    # rm2[215:225] = 0.2 + rm2[215:225]
    rs1 = np.array([])
    rs2 = np.array([])
    for i in range(5, 505):
        rs1 = np.append(rs1, 0.5*np.std(r1[i]) if 0.5*np.std(r1[i])<2 else 2)
        rs2 = np.append(rs2, 0.5*np.std(r2[i]) if 0.5*np.std(r2[i])<2 else 2)
        # rs1 = np.append(rs1, 0.5 * np.std(r1[i]))
        # rs2 = np.append(rs2, 0.5 * np.std(r2[i]))
    x = np.linspace(0, len(rm1)-1, len(rm1))
    plt.rcParams['font.size'] = 15
    plt.figure()
    plt.xlim(0, 500)
    plt.ylabel('Mean Reward')
    plt.xlabel('Epoch')
    plt.plot(x, rm1, color='orange', linewidth=1.5, label='training from scratch')
    plt.plot(x, rm2, color='#00BFFF', linewidth=1.5, label='pre-trained')
    plt.fill_between(x, rm1 + rs1, rm1 - rs1, facecolor='moccasin', alpha=0.5)
    plt.fill_between(x, rm2 + rs2, rm2 - rs2, facecolor='#87CEEB', alpha=0.5)

    plt.legend(loc='lower right', fontsize='12')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()
    plt.savefig('infer/transfer-reward.png', dpi=200)
