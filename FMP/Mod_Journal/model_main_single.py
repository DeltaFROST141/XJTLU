import os
import time
import gc

from model_resnet18 import *
from model_resnet34 import *
from tool_mat_reader_single import *

DEBUG = False
LEVEL = 18

if LEVEL == 34:
    model = ResNet34()
elif LEVEL == 18:
    model = ResNet18()

# Define the gpu ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
model = t.nn.DataParallel(model)
# Model setting ----------------------------------------------------------
epoch = 50
num_middle_check = 5
checkpoint = int(epoch / num_middle_check)
if checkpoint <= 0:
    checkpoint = 1

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
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # Print the current time -------------------------------------------------
    key_info_print(time_stamp)
    # Begin ------------------------------------------------------------------
    model = model.cuda()
    tra_data = tra
    optimizer = t.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )
    scheduler = t.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=int(epoch / 2),
        gamma=0.1,
    )
    criterion = nn.BCELoss()
    criterion = criterion.cuda()
    # Create the folder -------------------------------------------------------
    path = "../model"
    os.makedirs(path, exist_ok=True)
    loss_list = []
    # Training ----------------------------------------------------------------
    for epo in range(epoch):
        model.train()
        loss = 0
        for bid, (images, labels) in enumerate(tra_data):
            optimizer.zero_grad()
            outputs = model(images.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
        loss_list.append(loss.item())
        if (epo + 1) % 1 == 0:
            # Not the checkpoint, print the loss only
            key_info_print(
                "Epoch [{}/{}] \t Learning rate: {:.5f} \t Loss: {:.8f}".
                format(epo + 1, epoch,
                       optimizer.state_dict()['param_groups'][0]['lr'],
                       loss.item()))
        if (epo + 1) % checkpoint == 0:
            model.eval()
            # Evaluate the model ------------------------------------------------
            with t.no_grad():
                for bid, (images_tes, labels_tes) in enumerate(tes):
                    outputs_tes = model(images_tes.cuda())
                    # Move the data to CPU
                    outputs_tes = outputs_tes.cpu().detach().numpy()
                    labels_tes = labels_tes.cpu().detach().numpy()
                    # Calculate the accuracy
                    diff = np.mean(np.abs(np.subtract(outputs_tes,
                                                      labels_tes)))
                    key_info_print(
                        "Epoch [{}/{}] \t Accuracy on Test: {:.2f}%".format(
                            epo + 1, epoch, (1 - diff) * 100))
                #         if diff < 0.00002:
                #             # Save the model
                #             t.save(
                #                 model.state_dict(),
                #                 path + "/{}_{:.2f}.pth".format(
                #                     time.strftime("%m%d%H%M%S", time.localtime()),
                #                     (1 - diff) * 100),
                #             )
                #             key_info_print("Model saved!")
                gc.collect()
                t.cuda.empty_cache()
            model.train()
        scheduler.step()
    # Save the loss list ------------------------------------------------------
    np.savetxt(
        path + "/loss_list_{}.txt".format(time_stamp),
        loss_list,
        fmt="%.8f",
    )
