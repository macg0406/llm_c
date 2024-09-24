#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import struct

import torch
import safetensors

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


class ModelWriter:
    def __init__(self, output_filename, config, dtype="fp32", group_size=128):
        self.output_filename = output_filename
        self.writer = open(output_filename, "wb")
        self.fileopen = True
        self.config = config
        self.dtype = dtype
        self.dim = config["hidden_size"]
        self.hidden_dim = config["intermediate_size"]
        self.n_layers = config["num_hidden_layers"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_len = config["max_position_embeddings"]
        self.head_size = self.dim // self.n_heads
        self.tie_word_embeddings = config["tie_word_embeddings"]
        self.group_size = group_size
        self.writer.write(struct.pack('I', 0x616b3432))
        self.writer.write(struct.pack('i', 2))
        header = struct.pack('iiiiiii', self.dim, self.hidden_dim, self.n_layers, self.n_heads,
                                    self.n_kv_heads, self.vocab_size, self.max_seq_len)
        self.writer.write(header)
        self.writer.write(struct.pack('B', int(self.tie_word_embeddings)))
        self.writer.write(struct.pack('i', int(group_size)))
        pad = 256 - self.writer.tell() # pad rest with zeros; tell returns current pos
        assert pad >= 0
        self.writer.write(b'\0' * pad)

    def close(self):
        if self.fileopen:
            self.writer.flush()
            self.writer.close()
            self.fileopen = False
            print(f"Write {self.output_filename} success.")

    def __del__(self):
        self.close()

    def permute_reverse(self, tensor):
        if tensor.dim() == 1:
            dim1, dim2 = tensor.shape[0], 1
        elif tensor.dim() == 2:
            dim1, dim2 = tensor.shape
        else:
            dim1, dim2 = tensor.shape[-2], tensor.shape[-1]
        return tensor.view(-1, 2, self.head_size // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    def write_tensor_fp32(self, tensor):
        last_dim = tensor.shape[-1]
        new_tensor = tensor.view(-1, last_dim)
        for i in range(new_tensor.shape[0]):
            d = new_tensor[i].detach().cpu().to(torch.float32).numpy()
            # d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
            b = struct.pack(f'{last_dim}f', *d)
            self.writer.write(b)

    def write_tensor_int8(self, tensor):
        # d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
        # b = struct.pack(f'{len(d)}b', *d)
        # self.writer.write(b)
        last_dim = tensor.shape[-1]
        new_tensor = tensor.view(-1, last_dim)
        for i in range(new_tensor.shape[0]):
            d = new_tensor[i].detach().cpu().to(torch.int8).numpy()
            b = struct.pack(f'{last_dim}b', *d)
            self.writer.write(b)

    def write_tensor(self, tensor, dtype=None, permute_reverse=False, chunk_size=100):
        if permute_reverse:
            tensor = self.permute_reverse(tensor)
        if not dtype:
            dtype = self.dtype
        if dtype == "int8":
            self.write_tensor_int8(tensor)
        else:
            self.write_tensor_fp32(tensor)

    def write_tensor_with_quant(self, tensor, dtype=None, permute_reverse=False, group_size=-1):
        if group_size == 0:
            group_size = self.group_size
        elif group_size < 0:
            group_size = tensor.shape[-1]
        if permute_reverse:
            tensor = self.permute_reverse(tensor)
        if not dtype:
            dtype = self.dtype
        if dtype != "int8":
            raise ValueError("Only support int8 dtype.")
        else:
            q, s, err = quantize_q80(tensor, group_size)
            self.write_tensor_int8(q)
            self.write_tensor_fp32(s)


def export_fp32(input_path, output_path):
    with open("models/qwen/Qwen2-0.5B-Instruct/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    # print(json.dumps(config, indent=4))
    model_writer = ModelWriter("qwen2-05B-it-q8-v0.bin", config, dtype="int8")
    with safetensors.safe_open("models/qwen/Qwen2-0.5B-Instruct/model.safetensors", framework="pt", device="cpu") as fin:
        model_writer.write_tensor(fin.get_tensor('model.embed_tokens.weight'))
        for i in range(config["num_hidden_layers"]):
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.input_layernorm.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.q_proj.weight'), permute_reverse=True)
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.k_proj.weight'), permute_reverse=True)
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.v_proj.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.q_proj.bias'), permute_reverse=True)
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.k_proj.bias'), permute_reverse=True)
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.v_proj.bias'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.self_attn.o_proj.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.post_attention_layernorm.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.mlp.gate_proj.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.mlp.up_proj.weight'))
            model_writer.write_tensor(fin.get_tensor(f'model.layers.{i}.mlp.down_proj.weight'))
        model_writer.write_tensor(fin.get_tensor('model.norm.weight'))
        if not model_writer.tie_word_embeddings:
            model_writer.write_tensor(fin.get_tensor('lm_head.weight'))
        model_writer.close()


def main():
    pass


if __name__ == "__main__":
    main()