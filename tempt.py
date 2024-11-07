import math
def trade_off(acc:list, wga:list):
    baseline_acc = [98.1, 95.5, 91.9, 82.1]
    baseline_wga = [75.3, 45.9, 66.7, 68.4]
    for i in range(len(acc)):
        td = (wga[i] - baseline_wga[i]) / (baseline_acc[i] - acc[i])
        print(td, end="||")
    
    
jtt_acc = [93.3, 88.0, 78.6, 91.1];jtt_wga = [86.7, 81.1, 78.6, 69.3]
cnc_acc = [90.9, 89.9, 0, 81.7]; cnc_wga = [88.5, 88.5, 0, 68.9]
ssa_acc = [92.2, 92.8, 79.9, 76.6]; ssa_wga = [89.0, 89.8, 76.6, 69.9]
dfr_acc = [94.2, 91.3, 82.1, 87.2];dfr_wga = [92.9, 82.1, 74.7, 70.1]
self_acc = [94.0, 91.7, 79.1, 81.2];self_wga = [93.0, 83.9, 79.1, 70.7]

our_acc = [97.1, 91.2, 90.1, 0];our_wga = [93.3, 81.0, 72.02, 0]

trade_off(our_acc, our_wga)