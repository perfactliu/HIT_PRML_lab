import matplotlib.pyplot as plt
shot_zero = [0]
shot_num = [1, 2, 4]
acc_zero = [(0.7706+0.7696+0.7551)/3]
acc_num = [(0.5880+0.6148+0.6003)/3, (0.6336+0.6438+0.5939)/3, (0.6486+0.6454+0.6717)/3]
plt.figure()
plt.scatter(shot_zero, acc_zero, color='purple', marker='*', label='Clip zero-shot')
plt.plot(shot_num, acc_num, color='blue', marker='o', label='Clip + CoOp')
plt.title('Caltech 101')
plt.xlabel('shot number')
plt.ylabel('accuracy')
plt.legend()
plt.show()
print(acc_zero,acc_num)
