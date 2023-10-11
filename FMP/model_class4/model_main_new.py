# This is the GPU version.
# PT 2.0.1 on Python 3.8

import os
import time

from model_resnet18 import *
from model_resnet34 import *
# Udpate it!
from tool_mat_reader import *

DEBUG = True
LEVEL = 18
EPOCH = 10
CHECKPOINTSTEP = 1


if LEVEL == 34:
    model = ResNet34().cuda()
elif LEVEL == 18:
    model = ResNet18().cuda()
else:
    raise ValueError("The LEVEL configuration is incorrect.")

# WARNING: Update the total number of test data!
TEST_TOTAL = 2048

# Define the gpu ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Using GPU '0'; 0,1 for KSF.
# Print all the devices --------------------------------------------------
print("Devices: {}".format(t.cuda.device_count()))
# Using cuda backend -----------------------------------------------------
t.backends.cudnn.enable = True
# Clear the cache --------------------------------------------------------
t.cuda.empty_cache()
# Reset the peek memory --------------------------------------------------
t.cuda.reset_peak_memory_stats()
# Set the random seeds for reproducibility -------------------------------
t.cuda.manual_seed_all(12345)
# Reset the RAM statistics -----------------------------------------------
t.cuda.reset_accumulated_memory_stats()
# Set the random seeds for reproducibility -------------------------------
np.random.seed(12345)

if DEBUG:
    print("debug mode")
else:
    print("release mode")

    def print(*args, **kwargs):
        # Close all print function
        pass


def key_info_print(*args, **kwargs):
    """Print the key information
    """
    __builtins__.print(*args, **kwargs)


if __name__ == '__main__':
    # Get current time -------------------------------------------------------
    time_stamp = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    # Print the current time -------------------------------------------------
    key_info_print(time_stamp)
    # Begin ------------------------------------------------------------------
    optimizer = t.optim.Adam(
        model.parameters(),
        weight_decay=1e-6,
        lr=1e-3,  # 1e-4 ~ 85%
    )
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=int(EPOCH / 2),
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss().cuda()
    # Create the folder -------------------------------------------------------
    path = "../model"
    os.makedirs(path, exist_ok=True)
    # Training ----------------------------------------------------------------
    for epo in range(EPOCH):
        model.train()
        loss = 0
        for bid, (img, lab) in enumerate(tra):
            optimizer.zero_grad()
            out = model(img.cuda())
            #
            loss = criterion(out.cuda(), lab.long().cuda())
            loss.backward()
            optimizer.step()
        # Update the learning rate to reduce the pob. of overfitting
        scheduler.step()
        # Not the checkpoint, print the loss only
        key_info_print(
            "Epoch [{}/{}] \t Learning rate: {:.5f} \t Loss: {:.8f}".format(
                epo + 1, EPOCH,
                optimizer.state_dict()['param_groups'][0]['lr'], loss.item()))
        if (epo + 1) % CHECKPOINTSTEP == 0:
            model.eval()
            # Evaluate the model ------------------------------------------------
            with t.no_grad():
                total_cor = 0
                for bid, (ima_tes, lab_tes) in enumerate(tes):
                    # Move to Cuda (GPU)
                    ima_tes = ima_tes.cuda()
                    lab_tes = lab_tes.cuda()
                    # Prediction
                    out_tes = model(ima_tes)
                    _, pre = t.max(out_tes, dim=1)
                    print("The prediction results:\n{}".format(pre))
                    print("The ground truth:\n{}".format(lab_tes))
                    print((pre.int() == lab_tes.int()).sum().item())
                    total_cor += (pre.int() == lab_tes.int()).sum().item()
            print("Number of correct results: {}".format(str(total_cor)))
            key_info_print("Total accuracy is: {}%".format(total_cor /
                                                           TEST_TOTAL * 100))
            # Save the model ----------------------------------------------------
            if total_cor / TEST_TOTAL * 100 > 85:  # Target 85%
                os.makedirs('../model', exist_ok=True,)
                t.save(model.state_dict(),
                       "../model/model_{}_{}_{}.pth".format(str(tra_label.shape), LEVEL, epo + 1))
