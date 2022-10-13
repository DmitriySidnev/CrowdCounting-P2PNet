from subprocess import run, DEVNULL, CalledProcessError  # nosec

#from mo import mo_onnx
import torch

from models import build_model


class Args:
    def __init__(self):
        self.row = 4
        self.line = 4
        self.weight_path = './0005_best_mae_54.96.pth'
        self.backbone = 'vgg16_bn'


device = torch.device('cpu')
args = Args()
model = build_model(args)
model.to(device)
checkpoint = torch.load(args.weight_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

input_shape = [1, 3, 768, 1280]
data = torch.rand(input_shape)

input_names = ['data']
output_names = ['pred_logits', 'pred_points']

output_file_path = 'model.onnx'

#dynamic_axes = {'data': {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}}

with torch.no_grad():
    torch.onnx.export(
        model,
        data,
        output_file_path,
        verbose=True,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        #dynamic_axes=dynamic_axes,
        opset_version=10,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    )

norm_mean = [0.485, 0.456, 0.406]  # should be fixed if necessary
norm_std = [0.229, 0.224, 0.225]  # should be fixed if necessary

mean_values = str([s for s in norm_mean])
scale_values = str([s for s in norm_std])

#mo_onnx.main()
"""
command_line = ['mo',
                f'--input_model={output_file_path}',
                f'--mean_values={mean_values}',
                f'--scale_values={scale_values}',
                '--output_dir=./ir',
                '--data_type=FP32',
                #'--reverse_input_channels',  # should be uncomment if necessary
                f'--input_shape={input_shape}']
run(command_line, shell=False, check=True)

command_line = ['benchmark_app', '-m=ir/model.xml']
run(command_line, shell=False, check=True)
"""
