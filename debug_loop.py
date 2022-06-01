from src.utils.segmentation.losses import *
import gc
from tqdm import tqdm
import kornia.morphology as KM
from memory_profiler import profile

@profile
def open_close_image(data, o_size, c_size):
    kernel_open = torch.ones((o_size, o_size))
    kernel_close = torch.ones((c_size, c_size))
    r = KM.closing(data.unsqueeze(1), kernel_open)
    r = KM.opening(r, kernel_close)
    r = r.squeeze(1)
    return r

@profile
def values():

    # results = torch.zeros((10,10))
    results = []
    masks = torch.load('masks.pt')
    data_sm = torch.load('prediction_data.pt')
    masks = masks[:2]
    data_sm = data_sm[:2]

    for i in tqdm(range(10)):
        res = []
        for j in range(10):
            o_size = i * 4 + 1
            c_size = j * 4 + 1

            with torch.no_grad():
                kernel_open = torch.ones((o_size, o_size))
                kernel_close = torch.ones((c_size, c_size))
                r = KM.closing(data_sm.unsqueeze(1), kernel_open)
                r = KM.opening(r, kernel_close)
                r = r.squeeze(1)
                result = iou_binary(r, masks).mean().item()

            # r = open_close_image(data_sm, o_size, c_size)
            # res.append(result)
            del result
            del r
            gc.collect()
        results.append(res)
    return results


if __name__ == '__main__':
    results = values()
    torch.save(torch.tensor(results), 'results.pt')