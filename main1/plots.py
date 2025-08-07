import os.path

import numpy as np
import matplotlib.pyplot as plt

def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=np.float32)
    return data

def smooth(y, pts):
    b = np.ones(pts)/pts
    y_smooth = np.convolve(y, b, mode='same')
    return y_smooth
if __name__ == "__main__":
    path1 = 'runs/'
    path2 = 'case4/'
    path3 = '202508061025/'

    # loss
    l1 = np.array(loadtxtmethod(path1 + path2 + path3 + '/loss.txt'))
    ls1 = smooth(l1, 5)
    x1 = np.linspace(0, len(ls1)-1, len(ls1))
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.xlim(0, 600)
    plt.xlabel('Epoch')
    p1 = ax.plot(x1, ls1, color='orange', linewidth=2.5, label='training from scratch')
    plt.title('Loss', fontsize=30)
    plt.savefig(path1+path2+path3+'/loss.png', dpi=200, bbox_inches='tight')

    # policy loss
    l1 = np.array(loadtxtmethod(path1 + path2 + path3 + '/pgloss.txt'))
    ls1 = smooth(l1, 5)
    x1 = np.linspace(0, len(ls1)-1, len(ls1))
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.xlim(0, 600)
    plt.xlabel('Epoch')
    p1 = ax.plot(x1, ls1, color='orange', linewidth=2.5, label='training from scratch')
    plt.title('Policy Loss', fontsize=30)
    plt.savefig(path1+path2+path3+'/policyloss.png', dpi=200, bbox_inches='tight')


    # value loss
    l1 = np.array(loadtxtmethod(path1 + path2 + path3 + '/vloss.txt'))
    ls1 = smooth(l1, 5)
    x1 = np.linspace(0, len(ls1)-1, len(ls1))
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.xlim(0, 600)
    plt.xlabel('Epoch')
    p1 = ax.plot(x1, ls1, color='orange', linewidth=2.5, label='training from scratch')
    plt.title('Value Loss', fontsize=30)
    plt.savefig(path1+path2+path3+'/valueloss.png', dpi=200, bbox_inches='tight')

    # entropy loss
    l1 = np.array(loadtxtmethod(path1 + path2 + path3 + '/entloss.txt'))
    ls1 = smooth(l1, 5)
    x1 = np.linspace(0, len(ls1)-1, len(ls1))
    plt.rcParams['font.size'] = 20
    fig, ax = plt.subplots()
    plt.xlim(0, 600)
    plt.xlabel('Epoch')
    p1 = ax.plot(x1, ls1, color='orange', linewidth=2.5, label='training from scratch')
    plt.title('Entropy Loss', fontsize=30)

    plt.savefig(path1+path2+path3+'/entloss.png', dpi=200, bbox_inches='tight')


