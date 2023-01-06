import numpy as np
import datetime

def print_losses(total_epoch, epoch, iter, iter_per_epoch, losses, print_keys=False):
    

    if print_keys:
        header_str = 'epoch [%d/%d]\t\t\tloss\t' % (epoch,total_epoch)

        for key, value in losses.items():
            if key != 'loss':
                if len(key) < 5:
                    key_str = key + ' ' * (5 - len(key))
                    header_str += '\t\t%s' % (key_str)
                else:
                    header_str += '\t\t%s' % (key[0:5])

        print(header_str)

    loss_str = 'epoch [%d/%d] %05d/%05d: \t%.4f\t' % (epoch,total_epoch, iter, iter_per_epoch, np.mean(losses['loss']))

    for key, value in losses.items():
        if key != 'loss':
            loss_str += '\t\t%.4f' % (np.mean(value))

    now = datetime.datetime.now()
    nowTime = now.strftime('%H:%M:%S')   # 12:11:32
    loss_str += '\t\t%s' % (nowTime)
    print(loss_str)


