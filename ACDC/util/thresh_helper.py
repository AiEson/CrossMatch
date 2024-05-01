import torch
import torch.distributed as dist


class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):
        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
        self.gpu_num = dist.get_world_size()

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape
        pred_gather = torch.zeros([n * self.gpu_num, c, h, w]).cuda()
        dist.all_gather_into_tensor(pred_gather, pred)
        pred = pred_gather
        if ignore_mask is not None:
            ignore_mask_gather = torch.zeros([n * self.gpu_num, h, w]).cuda().long()
            dist.all_gather_into_tensor(ignore_mask_gather, ignore_mask)
            ignore_mask = ignore_mask_gather
        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        cls_num = self.nclass
        new_global = 0.0
        for cls in range(self.nclass):
            cls_map = mask_pred == cls
            if ignore_mask is not None:
                cls_map *= ignore_mask != 1
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        return_dict["new_global"] = new_global / cls_num

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)
        if update_g:
            self.thresh_global = (
                self.momentum * self.thresh_global
                + (1 - self.momentum) * thresh["new_global"]
            )

    def get_thresh_global(self):
        return self.thresh_global


class DropRateController:
    def __init__(self, init_rate=0.0, momentum=0.99) -> None:
        self.rate = init_rate
        self.momentum = momentum

    def new_drop_rate(self, loss_fp, loss_x):
        # loss_fp_gather = torch.zeros([1]).cuda()
        # dist.all_reduce(loss_fp_gather, loss_fp)
        # loss_x_gather = torch.zeros([1]).cuda()
        # dist.all_reduce(loss_x_gather, loss_x)
        if type(loss_fp) == torch.Tensor:
            loss_fp = loss_fp.item()
        if type(loss_x) == torch.Tensor:
            loss_x = loss_x.item()

        error_rate = loss_x - loss_fp
        # print("error_rate: ", error_rate)
        return error_rate

    def drop_rate_update(self, loss_fp, loss_x, scale_factor=2.0):
        error_rate = self.new_drop_rate(loss_fp, loss_x) * scale_factor
        out = self.momentum * self.rate + (1 - self.momentum) * (error_rate + self.rate)
        self.rate = min(0.999, max(0.001, out))
        return self.rate
