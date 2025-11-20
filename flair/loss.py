import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import math

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features




def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)



def get_multi_positive_mps(target, k):
    """
    :param target: tensor of shape (b, b*k), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b*k), for each row i, the col [i*k, (i+1)*k] should be ones
    """
    for i in range(target.shape[0]):
        target[i, i * k:(i + 1) * k] = 1
    return target



def get_multi_positive_tcs(target, k):
    """
    :param target: tensor of shape (b, b+k-1), all with values -1 at each entry
    :param k
    :return: tensor of shape (b, b+k-1), for each row i, the col [i, i+k) should be ones
    """
    for i in range(target.shape[0]):
        target[i, i: i + k] = 1
    return target




def get_mps_logits(image_features, text_features, logit_scale, logit_bias=None):
    logits = logit_scale * image_features @ text_features.T  # if multi-cap: (B, B*K)
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_mps_ground_truth(device, dtype, target_shape, negative_only=False,
                                        num_captions=4):
    dim0, dim1 = target_shape  # (B, B*K)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B*K)
    if not negative_only:
        labels = get_multi_positive_mps(target=labels, k=num_captions)
    return labels

def get_intra_logits(image_features, text_features, logit_scale, logit_bias=None):
    """
    image_features: (B, K, D),
    text_features: (B, K, D).
    Target: (B, K, K)
    """
    logits = logit_scale * torch.einsum('bkd,bjd->bkj', image_features, text_features)
    # logits = logit_scale * image_features @ text_features.T  
    if logit_bias is not None:
        logits += logit_bias
    return logits

def get_tcs_ground_truth(device, dtype, target_shape, negative_only=False, num_captions=4):
    dim0, dim1 = target_shape  # (B, B+K-1)
    labels = -torch.ones((dim0, dim1), device=device, dtype=dtype)  # (B, B+K-1)
    if not negative_only:
        labels = get_multi_positive_tcs(target=labels, k=num_captions)
    return labels

def get_tcs_logits(features_0, features_1, logit_scale, logit_bias=None):
    logits = logit_scale * torch.einsum('bij,bij->bi', features_0, features_1)
    if logit_bias is not None:
        logits += logit_bias
    return logits


class FlairLoss(nn.Module):
    """
    Implementation of FLAIR loss in: https://arxiv.org/pdf/2412.03561
    When setting added_mps_loss=False, this class is simply text-conditioned sigmoid loss;
    When added_mps_loss=True, this class is 'text-conditioned sigmod loss + multi-positive sigmoid loss'
    """

    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
            num_cap_per_img=8,
            added_mps_loss=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}
        self.num_cap_per_img = num_cap_per_img
        self.added_mps_loss = added_mps_loss


    def _loss_with_attn_pool(self, image_features, image_tokens, text_features, logit_scale,
                             logit_bias=None, negative_only=False, visual_proj=None, g_text_features=None):

        local_image_features = visual_proj(text_features, image_tokens, image_tokens)  # (B, B+K-1, D)

        local_image_features = F.normalize(local_image_features, dim=-1)
        global_text_features = F.normalize(text_features, dim=-1)

        i2t_logits = get_tcs_logits(local_image_features, global_text_features, logit_scale, logit_bias)

        i2t_labels = get_tcs_ground_truth(device=text_features.device,
                                        dtype=text_features.dtype,
                                        target_shape=i2t_logits.size(),
                                        negative_only=negative_only,
                                        num_captions=self.num_cap_per_img)

        tcs_loss = -F.logsigmoid(i2t_labels * i2t_logits).sum() / text_features.shape[1] # text-conditioned sigmoid loss


        if self.added_mps_loss:
            g_image_features = F.normalize(image_features, dim=-1)  #(B, D)
            g_text_features = F.normalize(g_text_features, dim=-1)  #(B*K, D)
            mps_logits = get_mps_logits(image_features=g_image_features, text_features=g_text_features,
                                                  logit_scale=logit_scale, logit_bias=logit_bias)
            g2g_labels = get_mps_ground_truth(device=g_text_features.device,
                                              dtype=g_text_features.dtype,
                                              target_shape=mps_logits.size(),
                                              negative_only=negative_only,
                                              num_captions=self.num_cap_per_img)
            mps_loss = -F.logsigmoid(g2g_labels * mps_logits).sum() / g_text_features.shape[0] # multi-positive sigmoid loss

            loss = (tcs_loss + mps_loss) / 2
        else:
            loss = tcs_loss


        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, image_tokens=None,
                visual_proj=None, output_dict=False):
        '''
        expected shape: text_features: (B*K, D), image_embeddings: (B, L, D)
        '''
        if self.added_mps_loss:
            g_text_features = text_features  # (B*K, D)
        else:
            g_text_features = None
        

        # We don't change the shape of image tokens anywhere before the loss function.
        batch_size = image_tokens.shape[0]
        num_captions = self.num_cap_per_img
        caption_indices = torch.arange(batch_size * num_captions).view(batch_size, num_captions).to(
            text_features.device)

        text_features = downsample_text_features(text_features=text_features, batch_size=batch_size,
                                                 caption_indices=caption_indices,
                                                 num_captions=num_captions)

        loss = self._loss_with_attn_pool(image_features=image_features,
                                         image_tokens=image_tokens,
                                         text_features=text_features,
                                         visual_proj=visual_proj,
                                         logit_scale=logit_scale,
                                         logit_bias=logit_bias,
                                         g_text_features=g_text_features)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features_to_left = g_text_features

                num_bidir, remainder = divmod(self.world_size - 1, 2)

                g_text_features_recv = None  # predefine it to be None

                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_bidir_with_grad(
                            left_rank,
                            right_rank,
                            g_text_features_to_left,
                            g_text_features_to_right,
                        )
                        for j in range(len(text_features_recv)):
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=text_features_recv[j],
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=g_text_features_recv[j]
                            )
                    else:
                        for f in text_features_recv:
                            loss += self._loss_with_attn_pool(
                                image_features=image_features,
                                image_tokens=image_tokens,
                                text_features=f,
                                visual_proj=visual_proj,
                                logit_scale=logit_scale,
                                logit_bias=logit_bias,
                                negative_only=True,
                                g_text_features=None)
                    text_features_to_left, text_features_to_right = text_features_recv
                    if self.added_mps_loss:
                        g_text_features_to_left, g_text_features_to_right = g_text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    if self.added_mps_loss:
                        g_text_features_recv = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=g_text_features_recv
                        )
                    else:
                        loss += self._loss_with_attn_pool(
                            image_features=image_features,
                            image_tokens=image_tokens,
                            text_features=text_features_recv,
                            visual_proj=visual_proj,
                            logit_scale=logit_scale,
                            logit_bias=logit_bias,
                            negative_only=True,
                            g_text_features=None)
            else:
                text_features_to_right = text_features
                if self.added_mps_loss:
                    g_text_features_to_right = g_text_features

                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    if self.added_mps_loss:
                        g_text_features_from_left = neighbour_exchange_with_grad(
                            left_rank, right_rank, g_text_features_to_right)
                    else:
                        g_text_features_from_left = None

                    loss += self._loss_with_attn_pool(
                        image_features=image_features,
                        image_tokens=image_tokens,
                        text_features=text_features_from_left,
                        visual_proj=visual_proj,
                        logit_scale=logit_scale,
                        logit_bias=logit_bias,
                        negative_only=True,
                        g_text_features=g_text_features_from_left)

                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss




def downsample_text_features(text_features, batch_size, caption_indices, num_captions):
    device = text_features.device
    own_caption_indices = caption_indices  # Shape: (B, K)

    mask = torch.ones(batch_size, batch_size, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)

    other_image_indices = torch.arange(batch_size, device=device).unsqueeze(0).expand(batch_size, batch_size)
    other_image_indices = other_image_indices[mask].view(batch_size, batch_size - 1)
    random_offsets = torch.randint(0, num_captions, (batch_size, batch_size - 1), device=device)  # (B, B-1)
    other_caption_indices = caption_indices[other_image_indices, random_offsets]  # sampled indices (B, B-1)

    combined_indices = torch.cat([own_caption_indices, other_caption_indices], dim=1)
    combined_indices, _ = combined_indices.sort(dim=1)
    flat_combined_indices = combined_indices.view(-1)  # flatten to take the text_features out

    downsampled_text_features = text_features[flat_combined_indices]

    embed_dim = text_features.shape[-1]  # Reshape to (B, K + B - 1, D)
    downsampled_text_features = downsampled_text_features.view(batch_size, num_captions + batch_size - 1, embed_dim)
    return downsampled_text_features
