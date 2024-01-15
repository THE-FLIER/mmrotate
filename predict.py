from mmdet.apis import init_detector, inference_detector
import mmrotate

config_file = './configs/oriented_reppoints/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py'
checkpoint_file = './runs/oriented/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')
inference_detector(model, 'demo/demo.jpg')