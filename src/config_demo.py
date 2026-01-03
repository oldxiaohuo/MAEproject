import os
import shutil

class models_genesis_config:
    model = "Unet3D"
    suffix = "genesis_chest_ct"
    exp_name = model + "-" + suffix
    
    # data
    data = "/mnt/dataset/shared/zongwei/LUNA16/Self_Learning_Cubes"
    train_fold=[0,1,2,3,4]
    valid_fold=[5,6]
    test_fold=[7,8,9]
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1        #这个参数是什么意思
    
    '''
    一般情况下，verbose 参数用于控制程序的输出详细程度：
verbose = 1 意味着会输出基本的信息，比如训练过程中的进度、损失值等。
如果 verbose 设置为 0，则通常表示静默模式，减少或不输出这些信息。
在该项目中，verbose 可能被用来控制训练或预处理时的日志、进度条或者调试信息的显示。
具体表现可以在训练相关的脚本或函数中看到它的使用（比如传递给训练函数或日志模块），让用户了解训练过程或调试。
    '''
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 6
    optimizer = "sgd"
    workers = 10
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    model_path = "pretrained_weights"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    '''
    方法没有参数，除了 self（指代类的实例）。
    dir(self) 会返回 self 对象的所有属性名列表，包括方法名、变量名等。
    '''
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")