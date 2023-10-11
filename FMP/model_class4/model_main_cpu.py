import os
import time

from model_resnet18 import *
from model_resnet34 import *
from tool_mat_reader import *


DEBUG = True
LEVEL = 18
EPOCH = 10
CHECKPOINTSTEP = 1

t.backends.cudnn.enable = True #  Do not comment it out 

if LEVEL == 34:
    # model = models.resnet34(pretrained=False).cpu()
    model = ResNet34().cpu()
elif LEVEL == 18:
    # model = models.resnet18(pretrained=False).cpu()
    model = ResNet18().cpu()

# Define the gpu ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Print all the devices --------------------------------------------------
print("Devices: {}".format(t.cuda.device_count()))
# Clear the cache --------------------------------------------------------
t.cuda.empty_cache()
# Reset the peek memory --------------------------------------------------
t.cuda.reset_peak_memory_stats()
# Set the random seeds for reproducibility -------------------------------
t.cuda.manual_seed_all(12345)
# Set the random seeds for reproducibility -------------------------------
np.random.seed(12345)
# Allocate the model to all available GPUs -------------------------------
# model = t.nn.DataParallel(model)

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
    print("Torch on CUDA: ", t.cuda.is_available())
    # Begin ------------------------------------------------------------------
    optimizer = t.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=int(EPOCH / 2),
        gamma=0.5,
    )
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MultiLabelMarginLoss().cpu()
    # Create the folder -------------------------------------------------------
    path = "../model"
    os.makedirs(path, exist_ok=True)
    # Training ----------------------------------------------------------------
    for epo in range(EPOCH):
        model.train()
        loss = 0
        for bid, (img, lab) in enumerate(tra):
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lab.long())
            loss.backward()
            optimizer.step()
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
                total_num = 2048
                for bid, (ima_tes, lab_tes) in enumerate(tes):
                    out_tes = model(ima_tes)
                    _, pre = t.max(out_tes, dim=1)
                    print(pre)
                    print(lab_tes)
                    print((pre.int() == lab_tes.int()).sum().item())
                    total_cor += (pre.int() == lab_tes.int()).sum().item()
                print("Number of correct results: {}".format(
                    str(total_cor)))
            key_info_print("Total accuracy is: {}%".format(
                total_cor/total_num*100))

            # Save the model ----------------------------------------------------
            if total_cor / total_num * 100 > 85:
                t.save(model.state_dict(),
                       "../model/model_{}_{}.pth".format(LEVEL, epo + 1))
