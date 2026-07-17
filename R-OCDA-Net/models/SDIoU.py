import math
import torch

def sd_bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    SDIoU: bool = True,
    eps: float = 1e-7,
    delta: float = 0.5,
):
    """
    Calculate IoU/GIoU/DIoU/CIoU/SDIoU for box1 (1,4) against box2 (n,4).
    Args:
        box1: Tensor, shape (1,4)
        box2: Tensor, shape (n,4)
        xywh: True if inputs are (x,y,w,h), else (x1,y1,x2,y2)
        GIoU/DIoU/CIoU/SDIoU: select metric (prefer setting one to True)
        eps: numerical stability
        delta: SDIoU dynamic term upper bound (0~1 recommended)
    Returns:
        Tensor of shape broadcasted to box2 (n,1) matching input dtype/device
    """
    device = box1.device
    dtype = box1.dtype

    if xywh:
        x1, y1, w1, h1 = box1.chunk(4, -1)
        x2, y2, w2, h2 = box2.chunk(4, -1)
        w1 = w1.clamp(min=0)
        h1 = h1.clamp(min=0)
        w2 = w2.clamp(min=0)
        h2 = h2.clamp(min=0)

        w1h1 = w1 * h1
        w2h2 = w2 * h2

        w1_ = w1 * 0.5
        h1_ = h1 * 0.5
        w2_ = w2 * 0.5
        h2_ = h2 * 0.5

        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        w1 = (b1_x2 - b1_x1).clamp(min=0)
        h1 = (b1_y2 - b1_y1).clamp(min=0)
        w2 = (b2_x2 - b2_x1).clamp(min=0)
        h2 = (b2_y2 - b2_y1).clamp(min=0)

        w1h1 = w1 * h1
        w2h2 = w2 * h2

    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h

    union = w1h1 + w2h2 - inter + eps

    iou = inter / union

    if not (CIoU or DIoU or GIoU or SDIoU):
        return iou

    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)

    if CIoU or DIoU or SDIoU:
        c2 = cw.pow(2) + ch.pow(2) + eps

        b1_cx = (b1_x1 + b1_x2) * 0.5
        b1_cy = (b1_y1 + b1_y2) * 0.5
        b2_cx = (b2_x1 + b2_x2) * 0.5
        b2_cy = (b2_y1 + b2_y2) * 0.5
        rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)

        if CIoU or SDIoU:
            v = (4.0 / (math.pi ** 2)) * (
                torch.atan((w2 + eps) / (h2 + eps)) - torch.atan((w1 + eps) / (h1 + eps))
            ).pow(2)
            with torch.no_grad():
                alpha = v / (v + (1.0 - iou + eps))

            if SDIoU:
                delta_t = torch.as_tensor(delta, device=device, dtype=dtype)
                beta = (w2h2 * delta_t) / 81.0
                beta = torch.minimum(beta, delta_t)

                return (
                    delta_t
                    - beta
                    + (1 - delta_t + beta) * (iou - v * alpha)
                    - (1 + delta_t - beta) * (rho2 / c2)
                )

            return iou - (rho2 / c2 + v * alpha)

        return iou - rho2 / c2

    c_area = cw * ch + eps
    return iou - (c_area - union) / c_area
