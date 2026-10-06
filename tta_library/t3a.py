"""T3A (Iwasawa and Matsuo, NeurIPS 2021), the training-free classifier-adjustment baseline.

Adapted from the official ZOA release's port, tta_library/t3a.py
(https://github.com/mala-lab/ZOA, MIT, (c) 2025 DengZeshuai), about 82% unchanged. The edits
handle the quantized head and dispatch between ViT and ResNet backbones.
"""
import torch
import torch.nn as nn

# NOTE (this port): on ImageNet-R (args.corruption == 'rendition'),
# self.imagenet_mask reduces logits to 200 classes, but warmup_labels are
# built in the full 1000-class space. The mask path is therefore class-space
# inconsistent. Safe for ImageNet-C corruptions; ImageNet-R uses the mask-aware path below.
def get_vit_featurer(network):
    network.head = nn.Identity()
    if hasattr(network, 'head_dist'):
        network.head_dist = None
    return network

def get_resnet_featurer(network):
    network.fc = nn.Identity()
    if hasattr(network, 'head_dist'):
        network.head_dist = None
    return network

class T3A(torch.nn.Module):
    """
    Test Time Template Adjustments (T3A)
    """
    def __init__(self, args, model, num_classes, filter_K):
        super().__init__()
        if args.arch == 'resnet50' or args.arch == 'resnet50_gn':
            self.classifier = model.fc
            self.featurizer = get_resnet_featurer(model)
        elif isinstance(model.head, nn.Linear):
            self.classifier = model.head
            self.featurizer = get_vit_featurer(model)
        else:
            raise NotImplementedError
        
        warmup_supports = self.classifier.weight.data
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = filter_K
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x, adapt=True):
        z = self.featurizer(x)
        if adapt:
            # online adaptation
            p = self.classifier(z)
            if self.imagenet_mask is not None:
                # ImageNet-R (or any 200-class subset): the masked argmax position
                # is the R-class index (0..199). Convert it to the full ImageNet-1K
                # class ID via the True-positions of the mask, so labels live in
                # the same 1000-class space as the model head, consistent with the
                # output convention used by NoAdapt/NEO/Ours/FOZO/ZOA.
                if not torch.is_tensor(self.imagenet_mask):
                    _mask_t = torch.as_tensor(self.imagenet_mask, dtype=torch.bool, device=p.device)
                else:
                    _mask_t = self.imagenet_mask.to(p.device)
                _mask_indices = _mask_t.nonzero(as_tuple=True)[0]  # (200,) ImageNet-1K IDs
                p_masked = p[:, _mask_t]                            # (B, 200)
                _argmax_in_masked = p_masked.argmax(1)              # (B,) in [0,199]
                _true_class_ids = _mask_indices[_argmax_in_masked]  # (B,) ImageNet-1K IDs
                yhat = torch.nn.functional.one_hot(_true_class_ids,
                                                   num_classes=self.num_classes).float()
                ent = softmax_entropy(p_masked)
            else:
                yhat = torch.nn.functional.one_hot(p.argmax(1),
                                                   num_classes=self.num_classes).float()
                ent = softmax_entropy(p)

            # prediction
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ent = torch.cat([self.ent, ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s)))).cuda()

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels

    def predict(self, x, adapt=False):
        return self(x, adapt)

    def reset(self):
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: disenable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            # m.requires_grad_(False)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(False)
    return model