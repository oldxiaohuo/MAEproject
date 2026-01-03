import os
import shutil

class mae_config():
    
    model = "MAE"
    suffix = "Whole_body_bone"
    exp_name = model + "-" + suffix

    # data
    data = "data/mhd_folder"
    input_H = 1024
    input_W = 512


    # model pre-training
    weights = None
    lr = 5e-5
    nb_epoch = 1000
    patience = 50
    batch_size = 20
    optimizer = "sgd"
    workers = 4
    max_queue_size = workers * 4
    save_samples = "png"

    # image deformation


    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    