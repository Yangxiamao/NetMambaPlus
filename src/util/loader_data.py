import math
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
import os
import json
import numpy as np
import itertools
import torch
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.common import MTU, PAD_ID
from typing import Union
from copy import deepcopy

##### dataset for raw bytes #####

def string_list_to_arr(string_list, num_string=5, string_len=320, pad_val=0):
    arr = []
    for string in string_list[:num_string]:
        string = string.strip()
        if len(string) == 0:
            arr.append([pad_val] * string_len)
        else:
            tmp_arr = [int(x) for x in string.split()]
            tmp_arr += [pad_val] * (string_len - len(tmp_arr))
            arr.append(tmp_arr[:string_len])
    arr += [[pad_val] * string_len] * (num_string - len(arr)) # (num_string, string_len)
    return list(itertools.chain(*arr)) # (num_string * string_len,)


def string_list_to_arr_with_mask(string_list, num_string=5, header_len=80, payload_len=240, pad_val=0):
    """
    Return:
    - arr: List[int], shape (num_string * (header_len + payload_len),)
    - pad_mask: List[int], shape (num_string * (header_len + payload_len),), 1 for pad, 0 for real data
    """
    arr = []
    pad_mask = np.zeros((num_string, header_len + payload_len), dtype=np.uint8)
    for idx, string in enumerate(string_list[:num_string]):
        header_str, payload_str = string
        header_str = header_str.strip()
        payload_str = payload_str.strip()
        if len(header_str) == 0:
            header_arr = [pad_val] * header_len
            pad_mask[idx, :header_len] = 1
        else:
            header_arr = [int(x) for x in header_str.split()]
            if len(header_arr) < header_len:
                pad_mask[idx, len(header_arr):header_len] = 1
                header_arr += [pad_val] * (header_len - len(header_arr))
            else:
                header_arr = header_arr[:header_len]
        if len(payload_str) == 0:
            payload_arr = [pad_val] * payload_len
            pad_mask[idx, header_len:] = 1
        else:
            payload_arr = [int(x) for x in payload_str.split()]
            if len(payload_arr) < payload_len:
                pad_mask[idx, header_len + len(payload_arr):] = 1
                payload_arr += [pad_val] * (payload_len - len(payload_arr))
            else:
                payload_arr = payload_arr[:payload_len]
        arr.append(header_arr + payload_arr)
    if len(arr) < num_string:
        for idx in range(len(arr), num_string):
            pad_mask[idx, :] = 1
            arr.append([pad_val] * (header_len + payload_len))
    return list(itertools.chain(*arr)), pad_mask.reshape(-1).tolist()  
    # (num_string * (header_len + payload_len),), (num_string * (header_len + payload_len),)


def sample_data(data: list, ratio: float):
    if ratio >= 1.0:
        return data
    import random
    random.seed(0)
    random.shuffle(data)
    print(f"Sample {ratio * 100:.2f}% of the dataset")
    return data[:int(len(data) * ratio)]


def normalize_bytes(raw_bytes, transform=None):
    if transform is not None:
        bytes = np.expand_dims(np.array(raw_bytes), axis=0).astype(np.uint8)
        bytes = transform(Image.fromarray(bytes))
    else:
        bytes = torch.tensor(raw_bytes)
        mean, std = torch.tensor([0.5]), torch.tensor([0.5])
        bytes = (bytes - mean) / std
    return bytes


class ByteDataset(Dataset):
    def __init__(self, path, num_packet=5, num_packet_byte=320, transform=None, ratio=1.0):
        assert os.path.exists(path), f"File not found: {path}"
        self.transform = transform
        self.ratio = ratio
        with open(path) as f:
            self.data = json.load(f)
        self.data = sample_data(self.data, ratio)
        self.idx2label = {}
        for item in self.data:
            item["data"] = string_list_to_arr(item["data"], num_string=num_packet, string_len=num_packet_byte)
            if item["label"] not in self.idx2label:
                self.idx2label[item["label"]] = item["name"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = item["label"]
        data = normalize_bytes(item["data"], transform=self.transform)
        return data, label
    
    def __str__(self):
        return super().__str__() + f" with {len(self.data)} items ({self.ratio * 100:.2f}%)"

##### dataset for sequence features #####

def str_to_arr(string, max_len=50, pad_value=0.):
    arr = [float(x) for x in string.split(" ")]
    arr = arr[:max_len] + [pad_value] * (max_len - len(arr))
    return arr

def str_to_arr_with_mask(string, max_len=50, pad_value=0.):
    """
    Return:
    - arr: List[float], shape (max_len,)
    - pad_mask: List[int], shape (max_len,), 1 for pad, 0 for real data
    """
    arr = [float(x) for x in string.split(" ")]
    pad_mask = [0] * min(len(arr), max_len) + [1] * (max_len - len(arr))
    arr = arr[:max_len] + [pad_value] * (max_len - len(arr))
    return arr, pad_mask


##### IAT (inter-arrival time) value encoding #####
# Both `sizes` and `intervals` are fed to FixedCosineEmbed, which is the standard
# sinusoidal encoding (frequencies 1 ~ 1e-4) designed for *integer positions* in
# roughly [0, 10000]. Shifted sizes land in [0, 2*MTU] and fit that domain well.
#
# `sigmoid_log` (the original upstream encoding) maps [0, inf) into [0.5, 1.0] via
# f(x) = (1+x)/(2+x). Its slope at x->0 is only 1/4, so the sub-millisecond region is
# almost completely flattened: moving the encoded value by 0.01 requires the IAT to
# differ by ~41 ms. On the MASQUE-vs-Direct dataset the class-discriminative signal
# sits at the *hundred-microsecond* scale, which after encoding spans 6.4e-06 --
# 0.0014% of the observed value range. All 20 tokens then receive near-identical
# embeddings and the timing channel is effectively dead before layer 1.
#
# `log10_decade` instead maps the IAT to log10 seconds scaled to 100 units per decade:
#     v = clip((log10(max(x, 1e-7)) + 7) * 100, 0, 1000)
# so 1us -> 100, 10us -> 200, 100us -> 300, 1ms -> 400, ... 10s -> 800, pad -> 1000.
# The realised range on our data is [0, 845] (matching the integer domain the cosine
# embedding expects) and the class gap grows to 0.76% of the span (~540x).
INTERVAL_ENCODINGS = ("sigmoid_log", "log10_decade")

IAT_LOG_FLOOR = 1e-7      # 0.1 us; also where exact-zero IATs (first packet) land
IAT_LOG_OFFSET = 7.0      # so that IAT == IAT_LOG_FLOOR maps to 0
IAT_LOG_SCALE = 100.0     # units per decade
IAT_PAD_VALUE = 1000.0    # out-of-range sentinel for padded positions


def encode_intervals(intervals, encoding="sigmoid_log"):
    """Encode raw IATs (seconds, may contain 0 and float('inf') for padding).

    `intervals` is expected to already be padded with float('inf').
    """
    if encoding == "sigmoid_log":
        return [1 / (1 + 1 / (1 + x)) for x in intervals] # sigmoid(log(1 + x))
    if encoding == "log10_decade":
        out = []
        for x in intervals:
            if not math.isfinite(x): # padded position
                out.append(IAT_PAD_VALUE)
                continue
            v = (math.log10(max(x, IAT_LOG_FLOOR)) + IAT_LOG_OFFSET) * IAT_LOG_SCALE
            out.append(min(max(v, 0.0), IAT_PAD_VALUE))
        return out
    raise ValueError(f"Unknown interval encoding: {encoding}")


def process_uni_flow_data(data: list,
                          num_packet, header_len, payload_len, seq_len,
                          interval_encoding="sigmoid_log"):
    idx2label = {}
    for idx, item in enumerate(data):
        byte_data, byte_pad_mask = string_list_to_arr_with_mask(item["data"], num_string=num_packet, 
                                                    header_len=header_len, payload_len=payload_len)
        sizes, size_pad_mask = str_to_arr_with_mask(item["sizes"], max_len=seq_len, pad_value=PAD_ID) # sizes of uni-flows
        sizes = [0 if x < 0 else MTU if x > MTU else x for x in sizes] # shift to [0, MTU]
        iats, iat_pad_mask = str_to_arr_with_mask(item["intervals"], max_len=seq_len, pad_value=float("inf")) # intervals of uni-flows
        iats = encode_intervals(iats, interval_encoding)
        if item["label"] not in idx2label:
            idx2label[item["label"]] = item["name"]
        data[idx] = {
            "bytes": byte_data,
            "byte_pad_mask": byte_pad_mask,
            "sizes": sizes,
            "size_pad_mask": size_pad_mask,
            "intervals": iats,
            "iat_pad_mask": iat_pad_mask,
            "label": item["label"],
            "env_idx": item.get("env_idx", -1), # for ByteSizeIntervalEnvDataset
        }
    return data, idx2label


def process_bi_flow_data(data: list, num_packet, num_packet_byte, seq_len, size_key,
                         interval_encoding="sigmoid_log"):
    idx2label = {}
    for idx, item in enumerate(data):
        item["data"] = string_list_to_arr(item["data"], num_string=num_packet, string_len=num_packet_byte)

        if size_key == "sizes":
            sizes = str_to_arr(item["sizes"], max_len=seq_len, pad_value=PAD_ID) # sizes of uni-flows
            sizes = [0 if x < 0 else MTU if x > MTU else x for x in sizes] # shift to [0, MTU]
        elif size_key == "signed_sizes":
            sizes = str_to_arr(item["signed_sizes"], max_len=seq_len, pad_value=PAD_ID+MTU) # sizes of bi-flows
            sizes = [0 if (x+MTU) < 0 else 2*MTU if (x+MTU) > 2*MTU else (x+MTU) for x in sizes] # shift to [0, 2 * MTU]
        else:
            raise ValueError(f"Unknown size key: {size_key}")

        intervals = str_to_arr(item["intervals"], max_len=seq_len, pad_value=float("inf")) # intervals of uni-flows
        intervals = encode_intervals(intervals, interval_encoding)
        if item["label"] not in idx2label:
            idx2label[item["label"]] = item["name"]
        data[idx] = {
            "bytes": item["data"],
            "sizes": sizes,
            "intervals": intervals,
            "label": item["label"],
            "env_idx": item.get("env_idx", -1), # for ByteSizeIntervalEnvDataset
        }
    return data, idx2label


##### dataset for multimodal features #####
class ByteSizeIntervalDataset(Dataset):
    def __init__(self, path_or_data: Union[str, list], num_packet=5, num_packet_byte=320, transform=None, ratio=1.0,
                 seq_len=20, class_idx=None, size_key="sizes", interval_encoding="sigmoid_log"):
        super().__init__()
        if isinstance(path_or_data, str):
            path = path_or_data
            assert os.path.exists(path), f"File not found: {path}"
            with open(path) as f:
                self.data = json.load(f)
        elif isinstance(path_or_data, list):
            self.data = path_or_data
        else:
            raise ValueError(f"Unknown type of path_or_data: {type(path_or_data)}")

        self.transform = transform
        self.ratio = ratio
        self.data = sample_data(self.data, ratio)
        assert size_key in ["sizes", "signed_sizes"], f"Unknown size key: {size_key}"
        assert interval_encoding in INTERVAL_ENCODINGS, f"Unknown interval encoding: {interval_encoding}"
        self.data, self.idx2label = process_bi_flow_data(self.data, num_packet, num_packet_byte, seq_len, size_key,
                                                        interval_encoding=interval_encoding)
        
        if class_idx is not None:
            self.data = [item for item in self.data if item["label"] == class_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # type: ignore
        item = self.data[idx]
        sizes = torch.tensor(item["sizes"])
        intervals = torch.tensor(item["intervals"])
        label = item["label"]
        bytes = normalize_bytes(item["bytes"], transform=self.transform)
        return bytes, sizes, intervals, torch.tensor(label, dtype=torch.long)


##### data loader functions #####

def build_dataset(args, data_path, ratio=1.0, class_idx=None):
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # out shape (1, H, W)
        transforms.ToTensor(), # divide by 255
        transforms.Normalize(mean, std),
    ])
    if args.dataset_type == "byte":
        dataset = ByteDataset(data_path, transform=transform, 
                              num_packet=args.num_packet, 
                              num_packet_byte=args.num_packet_byte,
                              ratio=ratio)
    elif args.dataset_type == "byte_size_interval":
        dataset = ByteSizeIntervalDataset(data_path, transform=transform,
                                num_packet=args.num_packet, 
                                num_packet_byte=args.num_packet_byte,
                                seq_len=args.seq_len, class_idx=class_idx,
                                ratio=ratio, size_key=args.size_key,
                                interval_encoding=getattr(args, "interval_encoding", "sigmoid_log"))
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    return dataset


def get_data_loader(args, data_path, data_ratio=1.0, 
                    batch_size=None, random_sampler=False,
                    class_idx=None):
    dataset = build_dataset(args, data_path, ratio=data_ratio, class_idx=class_idx)
    print(f"Dataset: {dataset}, type: {args.dataset_type}, size_key: {args.size_key}, "
          f"interval_encoding: {getattr(args, 'interval_encoding', 'sigmoid_log')}, ratio: {data_ratio}")
    if random_sampler:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    data_loader = DataLoader(
        dataset, sampler=sampler, # shuffle is replaced by sampler
        batch_size=batch_size or args.batch_size, # use args.batch_size if batch_size is None
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=False,
    )
    return data_loader, dataset.idx2label


def get_num_sample_per_cls(dataloader, num_classes, dataset_type):
    num_sample_per_cls = torch.zeros(num_classes)
    if dataset_type == "byte_size_interval":
        label_generator = (labels for _, _, _, labels in dataloader)
    elif dataset_type in ["byte", "seq"]:
        label_generator = (labels for _, labels in dataloader)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    for labels in label_generator:
        for label in labels:
            num_sample_per_cls[label] += 1
    return num_sample_per_cls
