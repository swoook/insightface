import torch

from argparse import ArgumentParser
import time

from mmdet.apis import init_detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--runmode', type=str, 
    choices=['infer', 'fps'], default='infer', 
    help='infer: infer from single image and write a result as a .jpg file \n fps: measure a FPS of given model')
    
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


class Inspector():
    def __init__(self, args):
        self.model = init_detector(args.config, args.checkpoint, device=args.device)
        self.device = args.device
        

    @torch.no_grad()
    def measure_fps(self):
        num_repet = 10000
        input_data = torch.Tensor(1, 3, 640, 640).to(self.device)
        print("start warm up")
        for _ in range(num_repet): self.model.feature_test(input_data)
        print("warm up done")

        if next(self.model.parameters()).is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_repet):
            self.model.feature_test(input_data)
        if next(self.model.parameters()).is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_time = (t1 - t0) / num_repet
        print('inference time: {} (s)'.format((inference_time)))


def main(args):
    inspector = Inspector(args)
    if args.runmode == 'fps':
        inspector.measure_fps()


if __name__ == "__main__":
    args = parse_args()
    main(args)